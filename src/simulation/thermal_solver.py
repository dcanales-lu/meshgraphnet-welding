"""2D transient thermal finite-element solver for welding simulation.

This module is the *data-generation engine* of ``meshgraphnet_welding``. It
solves the transient heat-conduction equation on an arbitrary 2D triangular
mesh while a **Goldak double-ellipsoid** heat source travels along a prescribed
weld path, and exports a sequence of snapshots that the ``src/data`` pipeline
turns into PyG graphs for the MeshGraphNet surrogate.

Physical model
--------------
The domain is interpreted as the **top-down surface of a plate** (the x-y plane
seen from above). The torch moves in-plane along the trajectory ``gamma(s)``.
We solve the through-thickness-integrated heat equation

    rho * Cp(T) * dT/dt = div(k grad T) + Q_vol

with ``Cp(T)`` the *apparent* heat capacity (latent heat of fusion smeared over
the solidus-liquidus interval), and ``Q_vol`` the volumetric source obtained by
projecting the 3D Goldak ellipsoid onto the surface (see :func:`goldak_flux`).

Boundary conditions on tagged boundary markers may be Dirichlet (fixed
temperature) or Robin (convection ``h(T - T_inf)`` plus optional linearized
radiation ``eps*sigma*(T^4 - T_inf^4)``).

Time integration uses the theta-method (default backward Euler, ``theta=1``).
Because the apparent heat capacity (and radiation) make the system nonlinear,
each step is solved with a Picard fixed-point iteration; for a purely linear
problem this collapses to a single solve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import dot, grad

#: Stefan-Boltzmann constant [W m^-2 K^-4].
SIGMA_SB = 5.670374419e-8


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class MaterialProperties:
    """Thermophysical properties (mild-steel defaults, SI units, kelvin).

    ``thickness`` folds the surface heat flux [W/m^2] into a volumetric source
    [W/m^3] for the thickness-integrated 2D model.
    """

    rho: float = 7850.0          # density [kg/m^3]
    cp: float = 600.0            # specific heat [J/(kg K)]
    k: float = 30.0             # thermal conductivity [W/(m K)]
    latent_heat: float = 2.7e5   # latent heat of fusion L [J/kg]
    T_solidus: float = 1723.0    # solidus temperature [K]
    T_liquidus: float = 1773.0   # liquidus temperature [K]
    T_ambient: float = 300.0     # initial / ambient temperature [K]
    thickness: float = 5.0e-3    # plate thickness h [m]

    def cp_apparent(self, T: np.ndarray) -> np.ndarray:
        """Apparent heat capacity ``cp + L * d(f_liquid)/dT``.

        The liquid-fraction derivative is modelled as a normalized Gaussian
        bump centred on the melting range so that its integral over temperature
        equals one (total latent heat ``L`` is released across the interval).
        Smooth and differentiable, which keeps the Picard iteration stable.
        """
        Tm = 0.5 * (self.T_solidus + self.T_liquidus)
        width = max(self.T_liquidus - self.T_solidus, 1.0)
        s = width / 4.0
        dfl_dT = np.exp(-(((T - Tm) / s) ** 2)) / (s * np.sqrt(np.pi))
        return self.cp + self.latent_heat * dfl_dT


@dataclass
class GoldakParams:
    """Goldak double-ellipsoid parameters.

    ``power`` is the gross arc power [W]; the net power deposited is
    ``efficiency * power``. ``f_r`` defaults to ``2 - f_f`` (the Goldak
    constraint ``f_f + f_r = 2``).
    """

    power: float = 2000.0    # gross power [W]
    efficiency: float = 0.8  # arc efficiency [-]
    a: float = 3.0e-3        # depth semi-axis [m] (kept for documentation)
    b: float = 3.0e-3        # half-width (transverse, along normal) [m]
    c_f: float = 3.0e-3      # front semi-axis (along tangent) [m]
    c_r: float = 6.0e-3      # rear semi-axis (along tangent) [m]
    f_f: float = 0.6         # front heat fraction [-]

    @property
    def f_r(self) -> float:
        return 2.0 - self.f_f

    @property
    def net_power(self) -> float:
        return self.efficiency * self.power


# A boundary spec is either a Dirichlet or Robin descriptor keyed by marker.
@dataclass
class Dirichlet:
    """Fixed-temperature boundary condition [K]."""

    value: float


@dataclass
class Robin:
    """Convection (+ optional linearized radiation) boundary condition.

    Flux out of the domain is ``h_conv (T - T_inf) + eps*sigma*(T^4 - T_inf^4)``.
    """

    h_conv: float = 15.0
    T_inf: float = 300.0
    emissivity: float = 0.0


BoundarySpec = Union[Dirichlet, Robin]


@dataclass
class SolverConfig:
    """Time-stepping and nonlinear-iteration controls."""

    dt: float = 0.02
    t_end: float = 2.0
    theta: float = 1.0          # 1.0 = backward Euler, 0.5 = Crank-Nicolson
    max_picard: int = 20
    picard_tol: float = 1e-4
    snapshot_every: int = 1
    verbose: bool = True
    #: Time [s] at which the heat source switches OFF. For ``t > source_end_time``
    #: the Goldak load is zero (post-weld cooling tail) and the torch is parked at
    #: its position at ``source_end_time``. ``None`` keeps the source on for the
    #: whole horizon (the original behaviour).
    source_end_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
class Trajectory:
    """Base class for weld-path trajectories ``gamma(s)`` (s = time [s])."""

    def position(self, s: float) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def tangent(self, s: float) -> np.ndarray:
        """Unit tangent via central finite differences (override if analytic)."""
        ds = 1e-6
        d = self.position(s + ds) - self.position(s - ds)
        norm = np.linalg.norm(d)
        return d / norm if norm > 0 else np.array([1.0, 0.0])

    def normal(self, s: float) -> np.ndarray:
        """Unit normal: tangent rotated +90 degrees."""
        t = self.tangent(s)
        return np.array([-t[1], t[0]])


class LinearTrajectory(Trajectory):
    """Straight path from ``start`` to ``end`` travelled at constant ``speed``."""

    def __init__(self, start, end, speed: float):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.speed = float(speed)
        d = self.end - self.start
        self._length = np.linalg.norm(d)
        self._dir = d / self._length if self._length > 0 else np.array([1.0, 0.0])

    def position(self, s: float) -> np.ndarray:
        travelled = min(self.speed * s, self._length)
        return self.start + travelled * self._dir

    def tangent(self, s: float) -> np.ndarray:
        return self._dir.copy()


class ParametricTrajectory(Trajectory):
    """Arbitrary ``gamma(s)`` callable, optional analytic derivative ``dgamma``."""

    def __init__(
        self,
        fn: Callable[[float], np.ndarray],
        dfn: Optional[Callable[[float], np.ndarray]] = None,
    ):
        self._fn = fn
        self._dfn = dfn

    def position(self, s: float) -> np.ndarray:
        return np.asarray(self._fn(s), dtype=float)

    def tangent(self, s: float) -> np.ndarray:
        if self._dfn is not None:
            d = np.asarray(self._dfn(s), dtype=float)
            norm = np.linalg.norm(d)
            return d / norm if norm > 0 else np.array([1.0, 0.0])
        return super().tangent(s)


# ---------------------------------------------------------------------------
# Goldak heat source (2D surface projection)
# ---------------------------------------------------------------------------
def goldak_flux(
    x: np.ndarray,
    y: np.ndarray,
    pos: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    params: GoldakParams,
    thickness: float,
) -> np.ndarray:
    """Volumetric heat-source density [W/m^3] for the 2D top-down model.

    The 3D Goldak ellipsoid integrated analytically over the through-thickness
    direction yields the surface flux ``q2D = 6 f Q / (pi b c) *
    exp(-3 xi^2/c^2 - 3 eta^2/b^2)`` [W/m^2], where ``xi`` is the along-tangent
    coordinate and ``eta`` the along-normal coordinate relative to the source
    centre. The front semi-axis ``c_f`` is used ahead of the torch (``xi >= 0``)
    and the rear ``c_r`` behind it. Dividing by ``thickness`` gives the
    volumetric source used in the thickness-integrated PDE.

    All array arguments broadcast, so this works for both plain point arrays and
    skfem quadrature-point arrays of shape ``(n_elems, n_qp)``.
    """
    dx = x - pos[0]
    dy = y - pos[1]
    xi = dx * tangent[0] + dy * tangent[1]
    eta = dx * normal[0] + dy * normal[1]

    front = xi >= 0.0
    c = np.where(front, params.c_f, params.c_r)
    f = np.where(front, params.f_f, params.f_r)

    q2d = (
        6.0 * f * params.net_power / (np.pi * params.b * c)
        * np.exp(-3.0 * xi**2 / c**2 - 3.0 * eta**2 / params.b**2)
    )
    return q2d / thickness


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------
def rectangular_plate(
    width: float, height: float, nx: int, ny: int
) -> MeshTri:
    """Rectangular plate mesh ``[0,width] x [0,height]`` with tagged edges.

    Boundary markers ``left``, ``right``, ``bottom``, ``top`` are attached for
    use in :class:`BoundaryConditions`.
    """
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, width, nx + 1), np.linspace(0.0, height, ny + 1)
    )
    return mesh.with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0.0),
            "right": lambda x: np.isclose(x[0], width),
            "bottom": lambda x: np.isclose(x[1], 0.0),
            "top": lambda x: np.isclose(x[1], height),
        }
    )


def load_mesh(path: Union[str, Path]) -> MeshTri:
    """Load an arbitrary 2D triangular mesh via meshio (``MeshTri.load``)."""
    return MeshTri.load(str(path))


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Sequence of solver snapshots, ready for ML ingestion or visualization."""

    coords: np.ndarray                 # (N, 2) node coordinates
    cells: np.ndarray                  # (M, 3) triangle connectivity
    times: np.ndarray                  # (S,) snapshot times
    temperature: np.ndarray            # (S, N) temperature field
    boundary_masks: dict               # marker -> (N,) bool node mask
    source_position: np.ndarray        # (S, 2)
    source_tangent: np.ndarray         # (S, 2)
    source_normal: np.ndarray          # (S, 2)
    source_power: np.ndarray           # (S,) net deposited power
    metadata: dict = field(default_factory=dict)  # process/Goldak/BC params

    # -- persistence -------------------------------------------------------
    def save_npz(self, path: Union[str, Path]) -> Path:
        """Save all fields to a compressed ``.npz`` (ML source of truth)."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "coords": self.coords,
            "cells": self.cells,
            "times": self.times,
            "temperature": self.temperature,
            "source_position": self.source_position,
            "source_tangent": self.source_tangent,
            "source_normal": self.source_normal,
            "source_power": self.source_power,
            "boundary_marker_names": np.array(
                list(self.boundary_masks.keys()), dtype=object
            ),
        }
        for name, mask in self.boundary_masks.items():
            arrays[f"bmask_{name}"] = mask
        arrays["metadata_json"] = np.array(json.dumps(self.metadata))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "SimulationResult":
        """Inverse of :meth:`save_npz`."""
        data = np.load(Path(path), allow_pickle=True)
        names = list(data["boundary_marker_names"])
        masks = {str(n): data[f"bmask_{n}"] for n in names}
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        return cls(
            coords=data["coords"],
            cells=data["cells"],
            times=data["times"],
            temperature=data["temperature"],
            boundary_masks=masks,
            source_position=data["source_position"],
            source_tangent=data["source_tangent"],
            source_normal=data["source_normal"],
            source_power=data["source_power"],
            metadata=metadata,
        )

    def save_xdmf(self, path: Union[str, Path]) -> Path:
        """Write an XDMF time-series (``.xdmf`` + ``.h5``) for ParaView.

        Requires ``h5py`` (meshio's XDMF backend).

        .. note::
            meshio's :class:`TimeSeriesWriter` creates the HDF5 payload in the
            *current working directory* using a bare ``<stem>.h5`` name, and
            references it from the ``.xdmf`` by that same basename. If the
            ``.xdmf`` is written to a different directory, that relative
            reference dangles and ParaView crashes on load. We therefore move
            the ``.h5`` next to the ``.xdmf`` so the (basename) reference
            resolves — keeping the pair self-contained and relocatable.
        """
        import shutil

        import meshio

        path = Path(path).with_suffix(".xdmf")
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.column_stack([self.coords, np.zeros(len(self.coords))])

        origin = Path.cwd()
        with meshio.xdmf.TimeSeriesWriter(str(path)) as writer:
            writer.write_points_cells(points, [("triangle", self.cells)])
            for i, t in enumerate(self.times):
                writer.write_data(float(t), point_data={"T": self.temperature[i]})

        # Relocate the HDF5 file next to the .xdmf (see note above).
        h5_name = Path(writer.h5_filename).name
        produced = origin / h5_name
        target = path.with_name(h5_name)
        if produced.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(str(produced), str(target))
        return path


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class TransientThermalSolver:
    """Backward-Euler / theta-method transient thermal FEM solver.

    Parameters
    ----------
    mesh:
        A :class:`skfem.MeshTri` with named boundary markers.
    material:
        :class:`MaterialProperties`.
    goldak:
        :class:`GoldakParams`.
    trajectory:
        A :class:`Trajectory` describing the torch path.
    boundary_conditions:
        Mapping ``marker -> Dirichlet | Robin``.
    config:
        :class:`SolverConfig`.
    """

    def __init__(
        self,
        mesh: MeshTri,
        material: MaterialProperties,
        goldak: GoldakParams,
        trajectory: Trajectory,
        boundary_conditions: dict,
        config: Optional[SolverConfig] = None,
    ):
        self.mesh = mesh
        self.mat = material
        self.goldak = goldak
        self.traj = trajectory
        self.bcs = boundary_conditions
        self.cfg = config or SolverConfig()

        self.basis = Basis(mesh, ElementTriP1())
        self.coords = mesh.p.T.copy()           # (N, 2)
        self.cells = mesh.t.T.copy()            # (M, 3)
        self.N = self.basis.N

        # Split BCs by type.
        self._dirichlet = {m: s for m, s in self.bcs.items() if isinstance(s, Dirichlet)}
        self._robin = {m: s for m, s in self.bcs.items() if isinstance(s, Robin)}

        # Static stiffness matrix K = int k grad u . grad v.
        self._K = self._assemble_stiffness()

        # Convection part of the Robin operator (constant) and its RHS load.
        self._R_conv, self._conv_rhs = self._assemble_convection()

        # Facet bases for any radiative Robin markers (recomputed each Picard it).
        self._rad_markers = {
            m: s for m, s in self._robin.items() if s.emissivity > 0.0
        }
        self._rad_fbases = {
            m: FacetBasis(mesh, ElementTriP1(), facets=m) for m in self._rad_markers
        }

        # Dirichlet dof indices and prescribed values.
        self._D, self._x_dirichlet = self._build_dirichlet()

        # Per-marker boolean node masks for output.
        self.boundary_masks = self._build_boundary_masks()

        # Nonlinear if latent heat is released or radiation is present.
        self._nonlinear = self.mat.latent_heat > 0.0 or bool(self._rad_markers)

    # -- assembly ----------------------------------------------------------
    def _assemble_stiffness(self):
        k = self.mat.k

        @BilinearForm
        def stiff(u, v, w):
            return k * dot(grad(u), grad(v))

        return stiff.assemble(self.basis)

    def _assemble_mass(self, T: np.ndarray):
        """Mass matrix with apparent heat capacity evaluated at ``T``."""
        rho = self.mat.rho
        cp_apparent = self.mat.cp_apparent

        @BilinearForm
        def mass(u, v, w):
            return rho * cp_apparent(w["T"]) * u * v

        return mass.assemble(self.basis, T=self.basis.interpolate(T))

    def _assemble_convection(self):
        """Constant convection matrix and RHS load summed over Robin markers."""
        R = None
        rhs = np.zeros(self.N)
        for marker, spec in self._robin.items():
            fb = FacetBasis(self.mesh, ElementTriP1(), facets=marker)
            h = spec.h_conv
            T_inf = spec.T_inf

            @BilinearForm
            def conv_mat(u, v, w, h=h):
                return h * u * v

            @LinearForm
            def conv_rhs(v, w, h=h, T_inf=T_inf):
                return h * T_inf * v

            Rm = conv_mat.assemble(fb)
            R = Rm if R is None else R + Rm
            rhs += conv_rhs.assemble(fb)
        if R is None:
            from scipy.sparse import csr_matrix

            R = csr_matrix((self.N, self.N))
        return R, rhs

    def _assemble_radiation(self, T: np.ndarray):
        """Linearized radiation operator/load at temperature ``T``.

        Uses the effective coefficient ``h_rad = eps*sigma*(T^2+T_inf^2)(T+T_inf)``
        so that ``h_rad (T - T_inf) = eps*sigma*(T^4 - T_inf^4)``.
        """
        from scipy.sparse import csr_matrix

        R = csr_matrix((self.N, self.N))
        rhs = np.zeros(self.N)
        for marker, spec in self._rad_markers.items():
            fb = self._rad_fbases[marker]
            eps = spec.emissivity
            T_inf = spec.T_inf

            @BilinearForm
            def rad_mat(u, v, w, eps=eps, T_inf=T_inf):
                Ts = w["T"]
                h_rad = eps * SIGMA_SB * (Ts**2 + T_inf**2) * (Ts + T_inf)
                return h_rad * u * v

            @LinearForm
            def rad_rhs(v, w, eps=eps, T_inf=T_inf):
                Ts = w["T"]
                h_rad = eps * SIGMA_SB * (Ts**2 + T_inf**2) * (Ts + T_inf)
                return h_rad * T_inf * v

            R = R + rad_mat.assemble(fb, T=fb.interpolate(T))
            rhs += rad_rhs.assemble(fb, T=fb.interpolate(T))
        return R, rhs

    def _source_is_on(self, s: float) -> bool:
        """Whether the heat source is active at time ``s`` (off during cooling)."""
        end = self.cfg.source_end_time
        return end is None or s <= end

    def _traj_time(self, s: float) -> float:
        """Trajectory query time, clamped to the weld end during cooling.

        Once the source is off the torch is parked at its ``source_end_time``
        position, so co-moving coordinates stay finite instead of extrapolating
        the path off the plate.
        """
        end = self.cfg.source_end_time
        return s if end is None else min(s, end)

    def _assemble_source(self, s: float):
        """Goldak source load vector at trajectory time ``s``.

        Returns ``(load, pos, tangent, normal)``. During the post-weld cooling
        tail (``s > source_end_time``) the load is the zero vector (torch off);
        the parked-torch frame is still returned for snapshot recording.
        """
        sq = self._traj_time(s)
        pos = self.traj.position(sq)
        tangent = self.traj.tangent(sq)
        normal = self.traj.normal(sq)

        if not self._source_is_on(s):
            return np.zeros(self.N), pos, tangent, normal

        params = self.goldak
        thickness = self.mat.thickness

        @LinearForm
        def source(v, w):
            x, y = w.x
            q = goldak_flux(x, y, pos, tangent, normal, params, thickness)
            return q * v

        return source.assemble(self.basis), pos, tangent, normal

    def _build_dirichlet(self):
        D_parts = []
        x = np.zeros(self.N)
        for marker, spec in self._dirichlet.items():
            dofs = self.basis.get_dofs(marker).flatten()
            D_parts.append(dofs)
            x[dofs] = spec.value
        D = np.unique(np.concatenate(D_parts)) if D_parts else np.array([], dtype=int)
        return D, x

    def _build_boundary_masks(self):
        masks = {}
        for marker in self.bcs:
            mask = np.zeros(self.N, dtype=bool)
            mask[self.basis.get_dofs(marker).flatten()] = True
            masks[marker] = mask
        return masks

    # -- time stepping -----------------------------------------------------
    def _step(self, T_n: np.ndarray, t_np1: float, t_n: float) -> np.ndarray:
        """Advance one step from ``t_n`` to ``t_np1`` (theta-method + Picard)."""
        theta = self.cfg.theta
        dt = t_np1 - t_n

        b_np1, _, _, _ = self._assemble_source(t_np1)
        b_n, _, _, _ = self._assemble_source(t_n)

        T_iter = T_n.copy()
        max_it = self.cfg.max_picard if self._nonlinear else 1
        for _ in range(max_it):
            M = self._assemble_mass(T_iter)
            A = self._K + self._R_conv
            rhs_robin = self._conv_rhs.copy()
            if self._rad_markers:
                R_rad, rhs_rad = self._assemble_radiation(T_iter)
                A = A + R_rad
                rhs_robin = rhs_robin + rhs_rad

            lhs = M / dt + theta * A
            rhs = (
                M / dt @ T_n
                - (1.0 - theta) * (A @ T_n)
                + theta * b_np1
                + (1.0 - theta) * b_n
                + rhs_robin
            )

            T_new = solve(*condense(lhs, rhs, x=self._x_dirichlet, D=self._D))

            denom = np.linalg.norm(T_new)
            change = np.linalg.norm(T_new - T_iter) / denom if denom > 0 else 0.0
            T_iter = T_new
            if change < self.cfg.picard_tol:
                break
        return T_iter

    def run(self) -> SimulationResult:
        """Integrate over the time horizon and return collected snapshots."""
        n_steps = int(round(self.cfg.t_end / self.cfg.dt))
        T = np.full(self.N, self.mat.T_ambient)

        times, temps = [], []
        src_pos, src_tan, src_nor, src_pow = [], [], [], []

        def record(s: float, Tcur: np.ndarray):
            sq = self._traj_time(s)  # parked at the weld end during cooling
            times.append(s)
            temps.append(Tcur.copy())
            src_pos.append(self.traj.position(sq))
            src_tan.append(self.traj.tangent(sq))
            src_nor.append(self.traj.normal(sq))
            src_pow.append(self.goldak.net_power if self._source_is_on(s) else 0.0)

        record(0.0, T)

        iterator = range(1, n_steps + 1)
        if self.cfg.verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="thermal solve", unit="step")
            except ImportError:
                pass

        for n in iterator:
            t_n = (n - 1) * self.cfg.dt
            t_np1 = n * self.cfg.dt
            T = self._step(T, t_np1, t_n)
            if n % self.cfg.snapshot_every == 0 or n == n_steps:
                record(t_np1, T)

        return SimulationResult(
            coords=self.coords,
            cells=self.cells,
            times=np.array(times),
            temperature=np.array(temps),
            boundary_masks=self.boundary_masks,
            source_position=np.array(src_pos),
            source_tangent=np.array(src_tan),
            source_normal=np.array(src_nor),
            source_power=np.array(src_pow),
            metadata=self._build_metadata(),
        )

    def _build_metadata(self) -> dict:
        """Process/Goldak/BC parameters embedded in the raw output so that the
        graph-data pipeline can reconstruct every node feature from a single
        self-contained file."""
        g = self.goldak
        boundary_specs = {}
        for marker, spec in self.bcs.items():
            if isinstance(spec, Dirichlet):
                boundary_specs[marker] = {"type": "dirichlet", "value": float(spec.value)}
            elif isinstance(spec, Robin):
                boundary_specs[marker] = {
                    "type": "robin",
                    "h_conv": float(spec.h_conv),
                    "T_inf": float(spec.T_inf),
                    "emissivity": float(spec.emissivity),
                }
        return {
            "goldak": {
                "power": float(g.power),
                "efficiency": float(g.efficiency),
                "net_power": float(g.net_power),
                "a": float(g.a),
                "b": float(g.b),
                "c_f": float(g.c_f),
                "c_r": float(g.c_r),
                "f_f": float(g.f_f),
                "f_r": float(g.f_r),
            },
            "thickness": float(self.mat.thickness),
            "T_ambient": float(self.mat.T_ambient),
            "boundary_specs": boundary_specs,
        }

# RunPod MCP setup — checklist (do when moving off local training)

How to connect Claude Code to RunPod via MCP so the agent can create/manage pods,
volumes, and launch cloud training by natural language. **Written 2026-06-21;
deferred until the local v2 training finishes** (a Claude Code restart is needed
to activate MCP, and restarting now could orphan the running local training).

There are two RunPod MCP servers.

## 1. Docs server — ALREADY ADDED ✅ (no auth)

`runpod-docs` (HTTP, `https://docs.runpod.io/mcp`) was added to the user config
(`C:\Users\diego\.claude.json`) and shows **Connected**. It lets the agent search
RunPod docs. Its tools become available **after the next Claude Code restart**.

To re-add if ever lost:
```
claude mcp add runpod-docs --scope user --transport http https://docs.runpod.io/mcp
```

## 2. API server — TODO (needs Node.js + your API key)

The API server (create/stop pods, volumes, endpoints, registries) runs via
`npx @runpod/mcp-server`, which requires **Node.js** — currently NOT installed.

Steps:

1. **Install Node.js LTS:** https://nodejs.org  (or `winget install OpenJS.NodeJS.LTS`).
   Verify in a terminal: `node --version` and `npx --version`.

2. **Get a RunPod API key:** runpod.io → Settings → API Keys.

3. **Add the server — run this in YOUR OWN terminal, NOT in the Claude chat**
   (so the secret key never enters the conversation / agent context):
   ```
   claude mcp add runpod --scope user -e RUNPOD_API_KEY=<your_key> -- npx -y @runpod/mcp-server@latest
   ```

4. **Verify:** `claude mcp list` should show `runpod` connected.

## 3. Activation — restart timing ⚠️

MCP servers load at session start, so the new tools appear only **after restarting
Claude Code**. **Do NOT restart while the local training is running in the
background** — it may orphan/kill the training process. Restart only when:
- the local v2 run has given the results we want (or we accept stopping it), AND
- we're ready to move work to the cloud.

## 4. Security

- **Never paste the RunPod API key into the Claude chat.** Run the `claude mcp add
  runpod ...` command yourself in a separate terminal. The key lives only in your
  local `.claude.json`.

## 5. Once MCP is active — the RunPod plan for this project

With the API MCP connected (new session), the agent can drive RunPod by natural
language. The intended flow for the v2 training:

1. **Create a network volume ≥300 GB** mounted at `/workspace` (the v2 dataset is
   ~6.8 GB raw + ~274 GB processed `.pt` cache; see the cache-size caveat below).
2. **Deploy a pod** with a big GPU (40–80 GB VRAM: A100/L40S/H100) and ≥16 vCPUs.
3. **Get the code + data on the volume:** `git clone` the repo, then regenerate
   the dataset there (it is NOT in git — too large; the generator is seeded):
   ```
   uv sync --frozen
   uv run python -m src.simulation.generate_dataset --num_train 300 --num_val 60 --workers <cores>
   ```
4. **Launch training detached + auto-shutdown** (see `RUNPOD.md` and
   `scripts/runpod_train.sh`):
   ```
   tmux new -s train
   bash scripts/runpod_train.sh config.generic_full_v2_runpod.json
   ```
   `config.generic_full_v2_runpod.json` is tuned for a big GPU: batch 16 (raise to
   32 on 48–80 GB), full dataset per epoch (`max_windows_per_epoch=0`),
   `num_workers=16`, `progress_bar=false`. The wrapper auto-stops the pod on exit.
5. **Retrieve** `checkpoints_genfull_v2/best_model.pt` via `runpodctl send`.

### Cache-size optimization (worth doing before heavy cloud use)
The processed `.pt` cache is ~274 GB because `graph_builder` stores the mesh
topology (`edge_index`/`edge_attr`) redundantly in every per-timestep graph (77%
waste). An on-the-fly / shared-topology dataset (load mesh once per sim, build
node features per step) would cut it to ~60 GB and speed up I/O — meaningful on
paid cloud storage. See the note in the project memory.

## Related files
- `RUNPOD.md` — full no-Docker RunPod workflow (uv, tmux, auto-shutdown, resume).
- `scripts/runpod_train.sh` — train + auto-stop wrapper.
- `config.generic_full_v2_runpod.json` — big-GPU training config.

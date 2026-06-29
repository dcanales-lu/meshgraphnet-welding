"""Generate an English audio of the congress talk from the tagged script.

Strips the intent tags (``[informative]`` ...) and the ``<!-- Slide N -->``
comments from ``docs/talk_script_en_tagged.md`` and synthesizes speech with
Microsoft Edge's free neural voices (no API key) via the ``edge-tts`` package.

Run (needs internet)::

    uv run python scripts/make_talk_audio.py                 # default voice
    uv run python scripts/make_talk_audio.py --voice en-GB-RyanNeural --rate -10%
    uv run python scripts/make_talk_audio.py --list-voices   # show English voices

Output: ``docs/talk_audio_en.mp3`` (+ a ``.txt`` of the clean spoken text).
"""
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import edge_tts

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "talk_script_en_tagged.md"
OUT_MP3 = REPO / "docs" / "talk_audio_en.mp3"
OUT_TXT = REPO / "docs" / "talk_audio_en.txt"

TAG = re.compile(r"\[[a-zA-Z]+\]\s*")          # [informative], [explanation], ...
COMMENT = re.compile(r"<!--.*?-->", re.S)       # <!-- Slide N -->


def clean_speech() -> str:
    md = SRC.read_text(encoding="utf-8")
    # keep only the body, from the first slide marker onward (drop the header)
    i = md.find("<!-- Slide 1")
    body = md[i:] if i >= 0 else md
    body = COMMENT.sub("", body)                # drop slide-marker comments
    paras = []
    for block in body.split("\n\n"):
        line = TAG.sub("", block).strip()       # drop intent tags
        if line and not line.startswith(("#", ">", "-", "|", "*", "---")):
            paras.append(" ".join(line.split()))
    # blank line between paragraphs -> a natural pause per slide
    return "\n\n".join(paras)


async def synth(text: str, voice: str, rate: str) -> None:
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(str(OUT_MP3))


async def list_voices() -> None:
    vs = await edge_tts.list_voices()
    for v in sorted(vs, key=lambda x: x["ShortName"]):
        if v["Locale"].startswith("en"):
            print(f"{v['ShortName']:28s} {v['Gender']:7s} {v['Locale']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-US-GuyNeural",
                    help="edge-tts voice (try en-GB-RyanNeural, en-US-AndrewNeural)")
    ap.add_argument("--rate", default="-8%", help="speech rate, e.g. -8%% (slower) or +5%%")
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        asyncio.run(list_voices())
        return

    text = clean_speech()
    OUT_TXT.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"Clean speech: {words} words -> {OUT_TXT}")
    asyncio.run(synth(text, args.voice, args.rate))
    print(f"Wrote {OUT_MP3}  (voice {args.voice}, rate {args.rate})")


if __name__ == "__main__":
    main()

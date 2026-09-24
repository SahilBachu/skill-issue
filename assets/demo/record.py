"""Record the README demo GIF from real command runs.

    python assets/demo/record.py

1. Builds a sandbox (temp SKILL_ISSUE_HOME and CLAUDE_CONFIG_DIR) with a few real public skills.
2. Starts the daemon with the default gate and waits until the models are loaded.
3. Runs each command in SCRIPT for real and captures its output and wall time.
4. Renders a terminal animation (typed command, then the real output after the real delay).

Nothing is scripted except the commands themselves: every line of output and every timing in the
GIF comes from the run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SKILLS = [  # real skills from the pinned corpus (pool ids)
    "openai/skills:skills/.curated/gh-fix-ci",
    "openai/skills:skills/.curated/vercel-deploy",
    "openai/skills:skills/.curated/jupyter-notebook",
    "openai/skills:skills/.curated/linear",
    "huggingface/skills:skills/huggingface-datasets",
    "huggingface/skills:skills/hf-mem",
    "anthropics/skills:skills/webapp-testing",
    "anthropics/skills:skills/mcp-builder",
    "obra/superpowers:skills/systematic-debugging",
    "obra/superpowers:skills/test-driven-development",
    "anthropics/skills:skills/slack-gif-creator",
    "anthropics/skills:skills/frontend-design",
]
SCRIPT = [
    ("skill-issue index", []),
    ('skill-issue route "ci is red on my PR again, can you figure out why"', []),
    ('skill-issue route "rename getUserData to fetchUser everywhere in src/"', []),
    ('skill-issue route "will Qwen3-32B in bf16 fit on two 24GB cards?"', []),
]

W, H = 1100, 560
BG, FG, DIM, ACC, OK, PROMPT = "#0d1117", "#e6edf3", "#7d8590", "#5eead4", "#7ee787", "#818cf8"
FONT_SIZE = 17
LINE_H = 25


def font() -> ImageFont.FreeTypeFont:
    import matplotlib

    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(path), FONT_SIZE)


def sandbox() -> tuple[dict[str, str], Path]:
    from bench.data import pool

    tmp = Path(tempfile.mkdtemp(prefix="skill-issue-demo-"))
    home, claude = tmp / "si", tmp / "claude"
    home.mkdir()
    pl = pool()
    for sid in SKILLS:
        s = pl.skills[pl.index[sid]]
        d = claude / "skills" / s.name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {s.name}\ndescription: {json.dumps(' '.join(s.description.split()))}\n---\n\n{s.body}\n",
            encoding="utf-8",
        )
    (home / "config.toml").write_text("[sources]\nagents = false\nclaude_plugins = false\n", encoding="utf-8")
    env = {**os.environ, "SKILL_ISSUE_HOME": str(home), "CLAUDE_CONFIG_DIR": str(claude), "PYTHONIOENCODING": "utf-8"}
    return env, tmp


def run(cmd: str, env: dict[str, str], cwd: Path) -> tuple[list[str], float]:
    exe = [sys.executable, "-m", "skillissue", *__import__("shlex").split(cmd)[1:]]
    t = time.perf_counter()
    out = subprocess.run(exe, env=env, cwd=cwd, capture_output=True, text=True, encoding="utf-8").stdout
    return out.rstrip("\n").splitlines(), time.perf_counter() - t


def color_for(line: str) -> str:
    if line.startswith("->"):
        return OK if "no skill" not in line else DIM
    if line.startswith(("mode=", "  ")):
        return DIM
    return FG


def render(transcript: list[tuple[str, list[str], float]], out: Path) -> None:
    f = font()
    frames: list[Image.Image] = []
    durations: list[int] = []
    lines: list[tuple[str, str]] = []  # (text, color)

    def snap(ms: int) -> None:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, W - 1, 34], radius=0, fill="#161b22")
        for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
            d.ellipse([16 + i * 22, 11, 28 + i * 22, 23], fill=c)
        d.text((W // 2, 17), "skill-issue demo (real output, real timings)", fill=DIM, font=f, anchor="mm")
        visible = lines[-((H - 60) // LINE_H) :]
        for i, (text, color) in enumerate(visible):
            if text.startswith("$ "):
                d.text((24, 50 + i * LINE_H), "$", fill=PROMPT, font=f)
                d.text((24 + f.getlength("$ "), 50 + i * LINE_H), text[2:], fill=FG, font=f)
            else:
                d.text((24, 50 + i * LINE_H), text, fill=color, font=f)
        frames.append(img)
        durations.append(ms)

    snap(600)
    for cmd, output, secs in transcript:
        lines.append(("$ ", FG))
        for i in range(0, len(cmd), 3):  # typing
            lines[-1] = ("$ " + cmd[: i + 3], FG)
            snap(45)
        lines[-1] = ("$ " + cmd, FG)
        snap(max(250, int(secs * 1000)))  # the command's real wall time
        for o in output:
            lines.append((o, color_for(o)))
        lines.append(("", FG))
        snap(2600)
    snap(1500)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def main() -> None:
    env, tmp = sandbox()
    daemon = subprocess.Popen(
        [sys.executable, "-m", "skillissue", "daemon", "run"],
        env=env,
        cwd=tmp,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        while (
            '"ready"'
            not in subprocess.run(
                [sys.executable, "-m", "skillissue", "daemon", "status"], env=env, capture_output=True, text=True
            ).stdout
        ):
            time.sleep(1)
        run('skill-issue route "warm up"', env, tmp)
        transcript = []
        for cmd, _ in SCRIPT:
            output, secs = run(cmd, env, tmp)
            print("$", cmd, f"({secs * 1000:.0f} ms)")
            print("\n".join(output))
            transcript.append((cmd, output, secs))
        (Path(__file__).parent / "transcript.json").write_text(json.dumps(transcript, indent=1), encoding="utf-8")
        render(transcript, ROOT / "assets" / "demo.gif")
        print("wrote assets/demo.gif")
    finally:
        daemon.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

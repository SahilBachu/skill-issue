"""Turn bench/results/*.json into charts (assets/charts/*.svg, light + dark) and markdown tables.

    python -m bench.report

The README embeds the charts with <picture> so GitHub picks the variant for the viewer's theme,
and includes the tables between <!-- results:NAME --> markers, which this script rewrites."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter

from bench.corpus import ROOT
from bench.data import RESULTS

CHARTS = ROOT / "assets" / "charts"
README = ROOT / "README.md"
DOC = ROOT / "docs" / "results.md"

# Validated reference palette (dataviz skill): fixed slot order, light and dark steps.
THEME = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#0d1117", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}  # fmt: skip
MARKERS = ["o", "s", "D", "^"]
FONT = ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"]


def load(name: str) -> dict[str, Any] | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def style(ax: Any, t: dict[str, Any]) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=10, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t: dict[str, Any], w: float = 7.6, h: float = 4.2) -> tuple[Any, Any]:
    plt.rcParams["font.family"] = FONT
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    return fig, ax


def save(fig: Any, name: str, mode: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / f"{name}-{mode}.svg", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def scaling_chart(
    systems: list[tuple[str, str]], metric: str, name: str, title: str, lower_better: bool = False
) -> None:
    data = [(label, load(s)) for s, label in systems]
    data = [(label, d) for label, d in data if d]
    if not data:
        return
    for mode, t in THEME.items():
        fig, ax = figure(t)
        ends = []
        for i, (label, d) in enumerate(data):
            pts = sorted((v["catalog_size"], v[metric]) for v in d["test"].values())
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            c = t["series"][i]
            ax.plot(
                xs,
                ys,
                color=c,
                linewidth=2,
                marker=MARKERS[i],
                markersize=7,
                markeredgecolor=t["surface"],
                markeredgewidth=1.5,
                label=label,
            )
            ends.append([ys[-1], label, c])
        # Direct labels at the right end, nudged apart so they never overlap.
        ends.sort(key=lambda e: e[0])
        gap = 0.045
        for k in range(1, len(ends)):
            if ends[k][0] - ends[k - 1][0] < gap:
                ends[k][0] = ends[k - 1][0] + gap
        xmax = max(v["catalog_size"] for _, d in data for v in d["test"].values())
        for y, label, c in ends:
            ax.annotate(
                label,
                xy=(xmax, y),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=t["ink2"],
            )
            ax.plot([xmax * 1.02], [y], marker="s", markersize=6, color=c, clip_on=False)
        ax.set_xscale("log")
        ticks = sorted({v["catalog_size"] for _, d in data for v in d["test"].values()})
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([f"{x:,}" for x in ticks])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1)
        ax.set_xlabel("skills in catalog (log scale)", color=t["muted"], fontsize=10)
        ax.set_title(title, loc="left", color=t["ink"], fontsize=13, pad=14, fontweight="semibold")
        leg = ax.legend(loc="lower left" if not lower_better else "upper left", frameon=False, fontsize=9.5)
        for text in leg.get_texts():
            text.set_color(t["ink2"])
        save(fig, name, mode)


def latency_chart() -> None:
    d = load("gate_latency-cuda")
    if not d:
        return
    rows = [r for r in d["results"] if r["candidates"] == 12 and r["body_chars"] == 300]
    label = {
        "laya": "Laya (421M)",
        "bge-m3": "bge-reranker-v2-m3 (568M)",
        "gte-mb": "gte-reranker-modernbert-base (150M)",
        "minilm": "ms-marco-MiniLM-L6 (22M)",
    }
    rows.sort(key=lambda r: r["p50_ms"])
    for mode, t in THEME.items():
        fig, ax = figure(t, h=2.9)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ys = range(len(rows))
        ax.barh(list(ys), [r["p50_ms"] for r in rows], height=0.56, color=t["series"][0])
        for y, r in zip(ys, rows):
            ax.annotate(
                f"{r['p50_ms']:.0f} ms",
                xy=(r["p50_ms"], y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=10,
                color=t["ink2"],
            )
        ax.set_yticks(list(ys))
        ax.set_yticklabels([label.get(r["model"], r["model"]) for r in rows], color=t["ink2"], fontsize=10)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel(
            "gate latency, p50, 12 candidates (" + str(d.get("gpu") or d["device"]) + ")", color=t["muted"], fontsize=10
        )
        ax.set_title("Gate cost per prompt", loc="left", color=t["ink"], fontsize=13, pad=12, fontweight="semibold")
        save(fig, "latency", mode)


def calibration_chart(system: str, size: str = "100") -> None:
    d = load(system)
    if not d:
        return
    rel = d["test"][size]["reliability"]
    for mode, t in THEME.items():
        fig, ax = figure(t, w=4.6, h=4.4)
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
        ax.plot([0, 1], [0, 1], color=t["axis"], linewidth=1.2, linestyle=(0, (4, 4)), label="perfect calibration")
        xs = [b["confidence"] for b in rel]
        ys = [b["accuracy"] for b in rel]
        sizes = [max(30, min(260, b["n"] / 3)) for b in rel]
        ax.plot(xs, ys, color=t["series"][0], linewidth=2)
        ax.scatter(
            xs, ys, s=sizes, color=t["series"][0], edgecolor=t["surface"], linewidth=1.5, zorder=3, label=d["label"]
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("predicted probability", color=t["muted"], fontsize=10)
        ax.set_ylabel("observed rate of relevant skills", color=t["muted"], fontsize=10)
        ece = d["test"][size]["ece"]
        ax.set_title(
            f"Calibration (ECE {ece:.3f})", loc="left", color=t["ink"], fontsize=13, pad=12, fontweight="semibold"
        )
        leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
        for text in leg.get_texts():
            text.set_color(t["ink2"])
        save(fig, "calibration", mode)


def agent_chart() -> None:
    rows = []
    for size in (100, 1000):
        for arm in ("vanilla", "router"):
            files = sorted(
                (RESULTS / "agent").glob(f"{arm}-N{size}-n*.json"), key=lambda p: int(p.stem.split("-n")[-1])
            )
            if files:
                d = json.loads(files[-1].read_text(encoding="utf-8"))
                rows.append((size, arm, d["metrics"]["exact_match"], d["n"]))
    if not rows:
        return
    sizes = sorted({r[0] for r in rows})
    arms = [("vanilla", "Claude Code alone"), ("router", "Claude Code + skill-issue")]
    for mode, t in THEME.items():
        fig, ax = figure(t, w=6.0, h=3.8)
        width = 0.36
        for i, (arm, label) in enumerate(arms):
            xs, ys = [], []
            for j, s in enumerate(sizes):
                v = next((r[2] for r in rows if r[0] == s and r[1] == arm), None)
                if v is not None:
                    xs.append(j + (i - 0.5) * (width + 0.02))
                    ys.append(v)
            bars = ax.bar(xs, ys, width=width, color=t["series"][i], label=label)
            for b, v in zip(bars, ys):
                ax.annotate(
                    f"{v:.0%}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    color=t["ink2"],
                )
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([f"{s:,} skills" for s in sizes], color=t["ink2"])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1.08)
        ax.set_title(
            "Right skills loaded (exact match), Claude Code headless",
            loc="left",
            color=t["ink"],
            fontsize=12,
            pad=12,
            fontweight="semibold",
        )
        leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncols=2, bbox_to_anchor=(1, 1.02))
        for text in leg.get_texts():
            text.set_color(t["ink2"])
        save(fig, "agent", mode)


# ---- tables -----------------------------------------------------------------------------


def pct(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{x:.1%}"


def main_table(systems: list[str], size: str) -> str:
    head = "| System | Exact match | Top-1 | Recall@10 | None acc. | False inj. | F1 | ECE |\n|---|---|---|---|---|---|---|---|"
    rows = []
    for s in systems:
        d = load(s)
        if not d or size not in d["test"]:
            continue
        m = d["test"][size]
        rows.append(
            f"| {d['label']} | {pct(m['exact_match'])} | {pct(m['top1'])} | {pct(m['recall@10'])} | {pct(m['none_accuracy'])} | "
            f"{pct(m['false_injection_rate'])} | {pct(m['f1'])} | {m['ece']:.3f} |"
        )
    return "\n".join([head, *rows])


def scaling_table(systems: list[str]) -> str:
    sizes: list[int] = []
    rows = []
    for s in systems:
        d = load(s)
        if not d:
            continue
        pts = sorted((v["catalog_size"], v["exact_match"]) for v in d["test"].values())
        sizes = [p[0] for p in pts]
        rows.append(f"| {d['label']} | " + " | ".join(pct(p[1]) for p in pts) + " |")
    head = "| System | " + " | ".join(f"{x:,}" for x in sizes) + " |\n|---|" + "---|" * len(sizes)
    return "\n".join([head, *rows])


def hand_table(systems: list[str]) -> str:
    head = "| System | Exact match | Hit rate | None acc. | False inj. |\n|---|---|---|---|---|"
    rows = []
    for s in systems:
        d = load(s)
        if not d or "handwritten" not in d:
            continue
        m = d["handwritten"]
        rows.append(
            f"| {d['label']} | {pct(m['exact_match'])} | {pct(m['hit_rate'])} | {pct(m['none_accuracy'])} | {pct(m['false_injection_rate'])} |"
        )
    return "\n".join([head, *rows])


def agent_table() -> str:
    head = "| Catalog | Setup | n | Exact match | Hit rate | None acc. | False inj. | Median time | Reported cost / prompt |\n|---|---|---|---|---|---|---|---|---|"
    rows = []
    for p in sorted((RESULTS / "agent").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["n"] < 50:
            continue  # pilots
        m = d["metrics"]
        setup = "Claude Code alone" if d["arm"] == "vanilla" else "Claude Code + skill-issue"
        rows.append(
            f"| {d['size']:,} | {setup} | {d['n']} | {pct(m['exact_match'])} | {pct(m['hit_rate'])} | {pct(m['none_accuracy'])} | "
            f"{pct(m['false_injection_rate'])} | {m.get('latency_ms_p50', 0) / 1000:.1f} s | ${d['reported_cost_usd_mean']:.3f} |"
        )
    return "\n".join([head, *rows])


def skillret_table() -> str:
    d = load("skillret")
    if not d:
        return ""
    names = {
        "bm25": "BM25",
        "dense": "Embeddings (bge-small)",
        "hybrid": "Hybrid (skill-issue retrieval)",
        "skillrouter-retrieval": "SkillRouter SR-Emb-0.6B (retrieval only)",
    }
    head = f"| System ({d['n_queries']:,} queries, {d['n_skills']:,} skills) | nDCG@10 | Recall@1 | Recall@10 | MRR@10 |\n|---|---|---|---|---|"
    rows = []
    for k, m in d["systems"].items():
        label = names.get(k) or (load(k) or {}).get("label", k)
        rows.append(
            f"| {label} | {m['ndcg@10']:.3f} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} |"
        )
    return "\n".join([head, *rows])


def latency_table() -> str:
    head = "| Gate | Candidates | Hook p50 | Hook p95 | Process floor | Daemon RAM | GPU memory |\n|---|---|---|---|---|---|---|"
    rows = []
    for p in sorted((RESULTS / "hook_latency").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        gpu = f"{d['daemon_gpu_mb']:.0f} MB" if d.get("daemon_gpu_mb") else "n/a"
        rows.append(
            f"| {d['gate']} {Path(d['model']).name or ''} ({d['device']}) | {d['max_candidates']} | {d['p50_ms']:.0f} ms | {d['p95_ms']:.0f} ms | "
            f"{d['process_floor_p50_ms']:.0f} ms | {d['daemon_rss_mb']:.0f} MB | {gpu} |"
        )
    return "\n".join([head, *rows])


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- results:{name} -->)(.*?)(<!-- /results:{name} -->)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


MAIN = [
    "bm25",
    "dense",
    "hybrid",
    "skillrouter",
    "minilm-zs",
    "gte-mb-zs",
    "bge-m3-zs",
    "laya-zs",
    "minilm-ft",
    "gte-mb-ft",
    "laya-ft",
]


def main() -> None:
    cfg = (
        json.loads((RESULTS / "report_config.json").read_text(encoding="utf-8"))
        if (RESULTS / "report_config.json").is_file()
        else {}
    )
    lines = cfg.get(
        "scaling",
        [
            ["hybrid", "Hybrid retrieval only"],
            ["gte-mb-zs", "+ gte reranker (zero-shot)"],
            ["skillrouter", "SkillRouter"],
        ],
    )
    scaling_chart([tuple(x) for x in lines], "exact_match", "scaling", "Exact match as the catalog grows")
    scaling_chart(
        [tuple(x) for x in lines],
        "false_injection_rate",
        "false-injection",
        "False injections as the catalog grows",
        lower_better=True,
    )
    latency_chart()
    calibration_chart(cfg.get("default", "gte-mb-zs"))
    agent_chart()
    blocks = {
        "main": main_table(MAIN, "100"),
        "large": main_table(MAIN, "100000"),
        "scaling": scaling_table(MAIN),
        "handwritten": hand_table(MAIN),
        "agent": agent_table(),
        "skillret": skillret_table(),
        "latency": latency_table(),
    }
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        "# Full results\n\nGenerated by `python -m bench.report` from `bench/results/`.\n\n"
        + "\n\n".join(f"## {k}\n\n{v}" for k, v in blocks.items())
        + "\n",
        encoding="utf-8",
    )
    if README.is_file():
        text = README.read_text(encoding="utf-8")
        for k, v in blocks.items():
            text = replace_block(text, k, v)
        README.write_text(text, encoding="utf-8")
    print("charts in assets/charts, tables in docs/results.md and README.md")


if __name__ == "__main__":
    main()

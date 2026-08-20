"""Generate deterministic bilingual vector figures for the K100 IEEE paper.

The numeric arrays are frozen values already reported in the manuscript and
the Phase 11 evidence bundle.  PDF is the publication artifact; PNG is only a
high-resolution preview for visual QA.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)

COLORS = {
    "blue": "#2F6B9A",
    "light_blue": "#DCEAF5",
    "orange": "#D97A28",
    "light_orange": "#F9E4D0",
    "green": "#3B8C6E",
    "light_green": "#DCEFE7",
    "purple": "#7357A5",
    "light_purple": "#E9E2F3",
    "gray": "#687078",
    "light_gray": "#ECEFF1",
    "red": "#B4473D",
}


def set_language(lang: str) -> None:
    if lang == "cn":
        mpl.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    else:
        mpl.rcParams["font.family"] = ["DejaVu Sans"]


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def add_box(ax, xy, width, height, title, lines, face, edge, title_color=None):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.73,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=title_color or edge,
    )
    ax.text(
        x + width / 2,
        y + height * 0.36,
        lines,
        ha="center",
        va="center",
        fontsize=8.2,
        linespacing=1.25,
        color="#202428",
    )


def add_arrow(ax, start, end, color="#4A5056"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def framework_figure(lang: str) -> None:
    set_language(lang)
    if lang == "cn":
        text = {
            "pipeline": "部署流水线",
            "stages": [
                ("冻结模型与输入", "Prithvi-EO-2.0 + UPerNet\n1×6×224×224 FP32"),
                ("后端差异诊断", "$L_i$, $C_i$, $P_i$\n单幅冻结诊断影像"),
                ("固定精度配置", "M0-M5：FP16浮点岛 + INT8 QDQ\nFP32 head与段间接口"),
                ("K100执行", "25个ONNX/MXR常驻会话\nOrtValue → logits / 类别图"),
            ],
            "evidence": "五层独立证据",
            "tracks": [
                ("任务效用", "mIoU · Water / Boundary IoU"),
                ("执行放置", "MIGraphX > 0 · CPU = 0"),
                ("实际内核", "INT8 I8II · FP16 HBH"),
                ("系统性能", "同拓扑FP32 · 生产式same-CLI"),
                ("运行验收", "双节点 · 恢复 · 60分钟"),
            ],
            "decision": "证据汇合后的部署决策",
            "choice": "Full FP16：速度优先     |     M5：存储优先",
            "boundary": "边界：严格CPU-MIGraphX logits等价仍为 failed",
        }
    else:
        text = {
            "pipeline": "DEPLOYMENT PIPELINE",
            "stages": [
                ("Frozen Model & Input", "Prithvi-EO-2.0 + UPerNet\n1×6×224×224 FP32"),
                ("Backend Diagnosis", "$L_i$, $C_i$, $P_i$\none frozen diagnostic image"),
                ("Fixed Precision Maps", "M0-M5: FP16 islands + INT8 QDQ\nFP32 head and interfaces"),
                ("K100 Execution", "25 resident ONNX/MXR sessions\nOrtValue → logits / mask"),
            ],
            "evidence": "FIVE INDEPENDENT EVIDENCE LAYERS",
            "tracks": [
                ("Task utility", "mIoU · Water / Boundary IoU"),
                ("Placement", "MIGraphX > 0 · CPU = 0"),
                ("Realized kernels", "INT8 I8II · FP16 HBH"),
                ("System performance", "matched FP32 · production CLI"),
                ("Operations", "two nodes · recovery · 60 min"),
            ],
            "decision": "EVIDENCE-BASED DEPLOYMENT DECISION",
            "choice": "Full FP16: speed first     |     M5: storage first",
            "boundary": "Boundary: strict CPU-MIGraphX logit equivalence remains failed",
        }

    # Draw at the final IEEE two-column width.  Text therefore remains at its
    # intended print size instead of being shrunk from a presentation canvas.
    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#1F2933"
    muted = "#52606D"
    rule = "#9AA5B1"
    paper = "#F7F9FB"
    accents = [COLORS["blue"], COLORS["orange"], COLORS["purple"], COLORS["green"]]

    def section_label(y, label, line_start):
        ax.text(0.015, y, label, ha="left", va="center", fontsize=7.0,
                fontweight="bold", color=muted, transform=ax.transAxes)
        ax.plot([line_start, 0.985], [y, y], color=rule, linewidth=0.55,
                transform=ax.transAxes, clip_on=False)

    def stage_box(x, number, title, body, accent):
        y, w, h = 0.655, 0.225, 0.235
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=paper, edgecolor=rule, linewidth=0.75,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.plot([x, x + w], [y + h, y + h], color=accent, linewidth=2.2,
                transform=ax.transAxes, solid_capstyle="round", clip_on=False)
        ax.text(x + 0.025, y + h - 0.044, str(number), ha="center", va="center",
                fontsize=6.4, fontweight="bold", color="white", transform=ax.transAxes,
                bbox=dict(boxstyle="circle,pad=0.19", facecolor=accent, edgecolor=accent, linewidth=0))
        ax.text(x + 0.050, y + h - 0.044, title, ha="left", va="center",
                fontsize=7.2, fontweight="bold", color=ink, transform=ax.transAxes)
        ax.text(x + w / 2, y + 0.082, body, ha="center", va="center",
                fontsize=6.35, color=ink, linespacing=1.23, transform=ax.transAxes)
        return x, y, w, h

    def evidence_box(x, title, body, accent):
        y, w, h = 0.300, 0.184, 0.175
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.010",
            facecolor="white", edgecolor=rule, linewidth=0.65,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(box)
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 0.010, h, boxstyle="round,pad=0,rounding_size=0.006",
                facecolor=accent, edgecolor=accent, linewidth=0,
                transform=ax.transAxes, clip_on=False,
            )
        )
        ax.text(x + w / 2 + 0.004, y + 0.116, title, ha="center", va="center",
                fontsize=6.75, fontweight="bold", color=ink, transform=ax.transAxes)
        ax.text(x + w / 2 + 0.004, y + 0.054, body, ha="center", va="center",
                fontsize=5.75, color=muted, transform=ax.transAxes)

    section_label(0.955, text["pipeline"], 0.22)
    stage_x = [0.015, 0.265, 0.515, 0.765]
    for idx, (x, (title, body), accent) in enumerate(zip(stage_x, text["stages"], accents), start=1):
        stage_box(x, idx, title, body, accent)
    for idx in range(3):
        ax.annotate(
            "", xy=(stage_x[idx + 1] - 0.008, 0.772),
            xytext=(stage_x[idx] + 0.233, 0.772), xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", color=muted, linewidth=0.85,
                            shrinkA=0, shrinkB=0, mutation_scale=8),
        )

    section_label(0.555, text["evidence"], 0.34 if lang == "en" else 0.22)
    track_x = [0.015, 0.213, 0.411, 0.609, 0.807]
    track_accents = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["gray"]]
    for x, (title, body), accent in zip(track_x, text["tracks"], track_accents):
        evidence_box(x, title, body, accent)

    # Evidence converges into one explicit engineering decision.
    ax.annotate(
        "", xy=(0.50, 0.246), xytext=(0.50, 0.292), xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=muted, linewidth=0.85, mutation_scale=8),
    )
    decision = FancyBboxPatch(
        (0.185, 0.092), 0.630, 0.132,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor=COLORS["light_green"], edgecolor=COLORS["green"], linewidth=0.9,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(decision)
    ax.text(0.50, 0.180, text["decision"], ha="center", va="center",
            fontsize=6.6, fontweight="bold", color=COLORS["green"], transform=ax.transAxes)
    ax.text(0.50, 0.126, text["choice"], ha="center", va="center",
            fontsize=7.35, fontweight="bold", color=ink, transform=ax.transAxes)
    ax.text(0.50, 0.026, text["boundary"], ha="center", va="center",
            fontsize=6.0, color=COLORS["red"], transform=ax.transAxes)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save(fig, f"fig1_k100_framework_{lang}")


LOCAL_MAE = np.array(
    [0.4239, 0.0642, 0.0749, 0.0709, 0.0581, 0.0656, 0.0689, 0.0702, 0.1005, 0.0703,
     0.0963, 0.0838, 0.0554, 0.0797, 0.1284, 0.1076, 0.0958, 0.1327, 0.1258, 0.1081,
     0.1107, 0.1074, 0.0386, 0.0446, 0.1957]
)
CUM_MAE = np.array(
    [0.4239, 0.4944, 0.5913, 0.6450, 0.6628, 0.6985, 0.7098, 0.7023, 0.6770, 0.6697,
     0.6673, 0.6414, 0.6366, 0.6658, 0.6615, 0.6586, 0.6658, 0.6749, 0.7048, 0.7366,
     0.7653, 0.7919, 0.8203, 0.8176, 0.1871]
)


def sensitivity_figure(lang: str) -> None:
    set_language(lang)
    labels = [str(i) for i in range(24)] + (["头"] if lang == "cn" else ["Head"])
    mappings = np.zeros((6, 25), dtype=int)  # 0 INT8, 1 FP16, 2 FP32
    fp16_sets = [set(), {0}, {0, 14}, {0, 14, 17}, {0, 14, 17, 18}, {0, *range(14, 22)}]
    for row, selected in enumerate(fp16_sets):
        for idx in selected:
            mappings[row, idx] = 1
        mappings[row, 24] = 2

    fig = plt.figure(figsize=(14.8, 8.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.25, 1.05], hspace=0.30)
    ax = fig.add_subplot(gs[0])
    x = np.arange(25)
    ax.axvspan(13.5, 21.5, color=COLORS["light_orange"], alpha=0.58, zorder=0)
    ax.plot(x, LOCAL_MAE, color=COLORS["orange"], marker="o", markersize=4.2, linewidth=1.8,
            label="局部后端 MAE $L_i$" if lang == "cn" else "Local backend MAE $L_i$")
    ax.plot(x, CUM_MAE, color=COLORS["blue"], marker="s", markersize=3.7, linewidth=1.8,
            label="累计链 MAE $C_i$" if lang == "cn" else "Cumulative-chain MAE $C_i$")
    peaks = [0, 14, 17, 18]
    peak_offsets = {0: (0, 15), 14: (0, 11), 17: (-20, 13), 18: (22, 10)}
    ax.scatter(peaks, LOCAL_MAE[peaks], s=80, facecolors="none", edgecolors=COLORS["red"], linewidths=1.6, zorder=5)
    for idx in peaks:
        ax.annotate(
            f"{idx}: {LOCAL_MAE[idx]:.4f}",
            (idx, LOCAL_MAE[idx]),
            xytext=peak_offsets[idx],
            textcoords="offset points",
            ha="center",
            color=COLORS["red"],
            fontsize=7.5,
        )
    ax.text(
        17.5,
        0.86,
        "诊断样本的后期高差异区域 14–21" if lang == "cn" else "late high-discrepancy region in the diagnostic sample: 14–21",
        ha="center",
        va="center",
        color="#8A5B11",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.set_xlim(-0.6, 24.6)
    ax.set_ylim(0, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("分段（24个backbone block + head）" if lang == "cn" else "Segment (24 backbone blocks + head)")
    ax.set_ylabel("平均绝对误差（MAE）" if lang == "cn" else "Mean absolute error (MAE)")
    ax.grid(axis="y", color="#D9DDE0", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax2 = fig.add_subplot(gs[1])
    cmap = ListedColormap(["#4C83B6", "#E99A4D", "#8A9096"])
    ax2.imshow(mappings, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    ax2.set_yticks(np.arange(6))
    ax2.set_yticklabels(["M0", "M1", "M2", "M3", "M4", "M5"])
    ax2.set_xticks(np.arange(25))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("分段精度映射" if lang == "cn" else "Per-segment precision mapping")
    ax2.set_ylabel("候选" if lang == "cn" else "Candidate")
    ax2.set_xticks(np.arange(-0.5, 25, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=0.7)
    ax2.tick_params(which="minor", bottom=False, left=False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    legend_labels = ["INT8 QDQ", "FP16", "固定FP32 head" if lang == "cn" else "fixed FP32 head"]
    ax2.legend(
        handles=[Patch(facecolor=cmap(i), edgecolor="none", label=lab) for i, lab in enumerate(legend_labels)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
    )
    save(fig, f"fig2_sensitivity_mapping_{lang}")


def tradeoff_figure(lang: str) -> None:
    set_language(lang)
    names = ["FP32", "Full FP16", "M0", "M1", "M2", "M3", "M4", "M5"]
    capacity = np.array([2566.424, 1284.756, 763.951, 789.378, 813.353, 837.522, 861.689, 981.759])
    miou = np.array([0.860587, 0.860456, 0.859288, 0.860324, 0.860044, 0.860228, 0.860371, 0.860272])
    segmented_names = ["FP32", "M0", "M1", "M2", "M3", "M4", "M5"]
    segmented_latency = np.array([22.209, 24.823, 24.274, 24.179, 23.925, 23.540, 22.080])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), gridspec_kw={"width_ratios": [1.08, 1.0]})
    markers = {"FP32": "s", "Full FP16": "D", "M0": "X"}
    colors = {"FP32": COLORS["gray"], "Full FP16": COLORS["purple"], "M0": COLORS["red"]}
    offsets = {
        "FP32": (-42, 8), "Full FP16": (-24, 10), "M0": (-8, -18), "M1": (-14, 10),
        "M2": (-16, -18), "M3": (6, -17), "M4": (7, 9), "M5": (7, -17),
    }
    for i, name in enumerate(names):
        marker = markers.get(name, "o")
        color = colors.get(name, COLORS["blue"])
        ax.scatter(capacity[i], miou[i], marker=marker, s=92, color=color,
                   edgecolors="#30363B", linewidths=0.9, zorder=3)
        ax.annotate(name, (capacity[i], miou[i]), xytext=offsets[name], textcoords="offset points",
                    fontsize=7.8, fontweight="bold" if name in {"Full FP16", "M4", "M5"} else "normal")
    ax.axhline(miou[0], color=COLORS["gray"], linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_xlim(680, 2700)
    ax.set_ylim(0.85915, 0.86072)
    ax.set_xlabel("ONNX+MXR逻辑容量（MiB）" if lang == "cn" else "Logical ONNX+MXR capacity (MiB)")
    ax.set_ylabel("mIoU")
    ax.set_title("(a) 任务指标—逻辑容量" if lang == "cn" else "(a) Task metric versus logical capacity")
    ax.grid(color="#D9DDE0", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    bar_colors = [COLORS["gray"], COLORS["red"], COLORS["blue"], COLORS["blue"],
                  COLORS["blue"], COLORS["blue"], COLORS["green"]]
    bars = ax2.bar(segmented_names, segmented_latency, color=bar_colors, edgecolor="#30363B", linewidth=0.7)
    bars[1].set_hatch("//")
    ax2.axhline(segmented_latency[0], color=COLORS["gray"], linestyle="--", linewidth=1.0)
    for bar, value in zip(bars, segmented_latency):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.10, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=7.4)
    ax2.annotate(
        "-0.58% vs. FP32\nlatency parity" if lang == "en" else "相对FP32低0.58%\n时延近似持平",
        xy=(6, segmented_latency[-1] + 0.03), xytext=(5.28, 23.05),
        ha="center", va="bottom", fontsize=7.8, color="#30363B", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#30363B", linewidth=0.85,
                        shrinkA=2, shrinkB=2, mutation_scale=8),
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#8B949E",
                  linewidth=0.6, alpha=0.94),
    )
    callout = (
        "单体Full FP16（same-CLI部署范围）\n15.151 ms；不属于25段同拓扑比较"
        if lang == "cn"
        else "Monolithic Full FP16 (same-CLI deployment scope)\n15.151 ms; not part of the 25-segment comparison"
    )
    ax2.text(0.98, 0.98, callout, transform=ax2.transAxes, ha="right", va="top", fontsize=7.8,
             color=COLORS["purple"], bbox=dict(boxstyle="round,pad=0.35", facecolor=COLORS["light_purple"],
                                                edgecolor=COLORS["purple"], linewidth=0.9))
    ax2.set_ylim(20.8, 25.6)
    ax2.set_ylabel("端到端logits median（ms）" if lang == "cn" else "End-to-end-logits median (ms)")
    ax2.set_title("(b) 25段同拓扑系统时延" if lang == "cn" else "(b) Topology-matched 25-segment latency")
    ax2.grid(axis="y", color="#D9DDE0", linewidth=0.7)
    ax2.spines[["top", "right"]].set_visible(False)
    note = (
        "M0速度实验的轮间CV超过5%，仅作描述；M5与分段FP32为时延近似持平，不构成正式INT8加速结论。"
        if lang == "cn"
        else "M0 is descriptive because its between-trial CV exceeds 5%. M5 is at latency parity with segmented FP32; no formal INT8 speedup is claimed."
    )
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.5, color="#4D5358")
    fig.subplots_adjust(bottom=0.17, wspace=0.27)
    save(fig, f"fig3_tradeoff_{lang}")


def main() -> None:
    for lang in ("en", "cn"):
        framework_figure(lang)
        sensitivity_figure(lang)
        tradeoff_figure(lang)
    for path in sorted(OUT.glob("fig[123]_*.pdf")):
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()

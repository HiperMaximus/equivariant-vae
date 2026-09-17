"""Build the compact, English-language figures used by the INCISCOS paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GRAY = "#6b7280"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def downstream() -> None:
    normal = load_json(
        ROOT
        / "runs/local/ubc_ocean_mil_test_scored_v1/normal_vae/scored_predictions_and_metrics.json"
    )["metrics"]
    so2 = load_json(
        ROOT
        / "runs/local/ubc_ocean_mil_test_scored_v1/so2_vae/scored_predictions_and_metrics.json"
    )["metrics"]
    tissue = load_json(ROOT / "runs/local/tissue_test_scored_v1/label_efficiency_table.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)

    names = ["Macro-F1", "Balanced\naccuracy", "Accuracy"]
    normal_values = [normal["macro_f1"], normal["balanced_accuracy"], normal["accuracy"]]
    so2_values = [so2["macro_f1"], so2["balanced_accuracy"], so2["accuracy"]]
    x = np.arange(len(names))
    width = 0.34
    axes[0].bar(x - width / 2, normal_values, width, color=BLUE, label="Normal VAE")
    axes[0].bar(x + width / 2, so2_values, width, color=ORANGE, label=r"$\mathrm{SO}(2)$ VAE")
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 0.72)
    axes[0].set_ylabel("Sealed-test score")
    axes[0].set_title("(a) WSI diagnosis, 23 slides")
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.6)
    axes[0].legend(frameon=False, loc="upper left")

    rows = tissue["rows"]
    budgets = np.array([row["labels_per_class"] for row in rows])
    normal_f1 = np.array([row["normal_vae"]["metrics"]["macro_f1"] for row in rows])
    so2_f1 = np.array([row["so2_vae"]["metrics"]["macro_f1"] for row in rows])
    axes[1].plot(budgets, normal_f1, "o-", color=BLUE, linewidth=1.8, label="Normal VAE")
    axes[1].plot(budgets, so2_f1, "o-", color=ORANGE, linewidth=1.8, label=r"$\mathrm{SO}(2)$ VAE")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(FixedLocator(budgets))
    axes[1].xaxis.set_major_formatter(FixedFormatter(["250", "500", "1k", "2.5k", "5.7k"]))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylim(0.47, 0.80)
    axes[1].set_xlabel("Training labels per class (log scale)")
    axes[1].set_ylabel("Sealed-test macro-F1")
    axes[1].set_title("(b) Tissue label efficiency, 31,572 patches")
    axes[1].grid(color="#e5e7eb", linewidth=0.6)
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].annotate(
        "simultaneous CI\nexcludes 0",
        xy=(500, so2_f1[1]),
        xytext=(760, 0.545),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8},
        color=GRAY,
        fontsize=7.5,
    )

    fig.savefig(OUT / "downstream_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def decoder_geometry() -> None:
    result = load_json(
        ROOT
        / "runs/local/decoded_latent_transform_v1/decoded_latent_transform_v1"
        / "decoded_latent_transform_summary.json"
    )
    comparisons = result["comparisons"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)

    metrics = [
        ("Decoded action", comparisons["action_ratio"]),
        ("Canonicalization", comparisons["canonical_ratio"]),
    ]
    positions = [0.82, 1.18, 1.82, 2.18]
    datasets = []
    colors = []
    for _, metric in metrics:
        datasets.extend([metric["normal"], metric["so2"]])
        colors.extend([BLUE, ORANGE])
    box = axes[0].boxplot(
        datasets,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    for index, values in enumerate(datasets):
        jitter = np.linspace(-0.055, 0.055, len(values))
        axes[0].scatter(
            positions[index] + jitter,
            values,
            s=7,
            color=colors[index],
            alpha=0.45,
            linewidths=0,
        )
    axes[0].axhline(0.5, color=GRAY, linestyle="--", linewidth=0.9, label="absolute target")
    axes[0].set_xticks([1, 2], [name for name, _ in metrics])
    axes[0].set_ylim(0.42, 0.86)
    axes[0].set_ylabel("Normalized RMS ratio (lower is better)")
    axes[0].set_title("(a) Dense 5-degree sweep, 25 paired patches")
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.6)
    axes[0].plot([], [], color=BLUE, linewidth=6, label="Normal VAE")
    axes[0].plot([], [], color=ORANGE, linewidth=6, label=r"$\mathrm{SO}(2)$ VAE")
    axes[0].legend(frameon=False, loc="upper right")

    exact = comparisons["exact_d4"]
    transforms = ["rot90", "rot180", "rot270"]
    labels = [r"$90^\circ$", r"$180^\circ$", r"$270^\circ$"]
    normal_exact = [
        exact[name]["action_ratio_per_patch"]["all_25_descriptive"]["normal_median"]
        for name in transforms
    ]
    so2_exact = [
        exact[name]["action_ratio_per_patch"]["all_25_descriptive"]["so2_median"]
        for name in transforms
    ]
    x = np.arange(3)
    width = 0.34
    axes[1].bar(x - width / 2, normal_exact, width, color=BLUE, label="Normal VAE")
    axes[1].bar(x + width / 2, so2_exact, width, color=ORANGE, label=r"$\mathrm{SO}(2)$ VAE")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-7, 1.2)
    axes[1].set_ylabel("Decoded-action RMS ratio (log scale)")
    axes[1].set_title(r"(b) Exact $C_4$ decoder action")
    axes[1].grid(axis="y", color="#e5e7eb", linewidth=0.6)
    axes[1].text(
        1.0,
        8e-6,
        r"$\mathrm{SO}(2)$ medians $<3\times10^{-6}$",
        ha="center",
        va="bottom",
        color=ORANGE,
        fontsize=7.5,
    )

    fig.savefig(OUT / "decoder_geometry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    downstream()
    decoder_geometry()

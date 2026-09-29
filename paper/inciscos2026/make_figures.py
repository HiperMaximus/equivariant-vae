"""Build the compact, English-language figures used by the INCISCOS paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
BLUE = "#216b80"
ORANGE = "#b45143"
GRAY = "#64717b"
GRID = "#dfe5e8"


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
            "axes.edgecolor": "#a9b5bb",
            "axes.labelcolor": "#26343c",
            "text.color": "#26343c",
            "xtick.color": "#4c5c65",
            "ytick.color": "#4c5c65",
            "figure.dpi": 160,
        }
    )


def reconstruction_mosaic() -> None:
    originals = torch.load(
        ROOT / "docs/data/fixed25/originals.pt",
        map_location="cpu",
        weights_only=True,
    )["images_uint8"][:9]
    normal = torch.load(
        ROOT
        / "runs/kaggle/selected_runtime_full_v4_session3/artifacts/fixed25"
        / "boundary_060000/reconstruction_progress.pt",
        map_location="cpu",
        weights_only=True,
    )["reconstruction"][:9]
    so2 = torch.load(
        ROOT
        / "runs/kaggle/so2_selected_runtime_full_session7_fresh_v1_retry1"
        / "artifacts/fixed25/boundary_060000/reconstruction_progress.pt",
        map_location="cpu",
        weights_only=True,
    )["reconstruction"][:9]

    fig, axes = plt.subplots(3, 9, figsize=(7.15, 2.8))
    groups = [originals, normal, so2]
    for group_index, group in enumerate(groups):
        for patch_index in range(9):
            row, column = divmod(patch_index, 3)
            ax = axes[row, 3 * group_index + column]
            image = group[patch_index].permute(1, 2, 0).numpy()
            if group_index:
                image = (np.clip(image, -1, 1) + 1) / 2
            else:
                image = image / 255
            ax.imshow(image)
            ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.87, bottom=0.005,
                        wspace=0.025, hspace=0.025)
    for x, title in zip((1 / 6, 1 / 2, 5 / 6),
                        ("Original", "Conventional VAE", r"$\mathrm{SO}(2)$ VAE")):
        fig.text(x, 0.94, title, ha="center", va="center", fontsize=10)
    fig.savefig(OUT / "reconstructions_fixed9.png", dpi=300)
    plt.close(fig)


def reconstruction_boxplots() -> None:
    path = ROOT / "runs/local/vae_test_reconstruction_scored_v1/metrics/per_wsi_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.8), constrained_layout=True)
    metrics = [
        ("mae_norm", "MAE", "Normalized RGB"),
        ("mse_norm", "MSE", "Normalized RGB"),
        ("psnr_img", "PSNR", "dB"),
        ("ssim_img", "SSIM", "Image domain"),
    ]
    for ax, (key, title, unit) in zip(axes.flat, metrics):
        datasets = [
            np.array([float(row[f"{branch}_{key}"]) for row in rows])
            for branch in ("normal", "so2")
        ]
        box = ax.boxplot(
            datasets,
            positions=[1, 2],
            widths=0.36,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#26343c", "linewidth": 1.4},
            whiskerprops={"color": GRAY, "linewidth": 0.9},
            capprops={"color": GRAY, "linewidth": 0.9},
        )
        for patch, color in zip(box["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.24)
        jitter = np.linspace(-0.12, 0.12, len(rows))
        for position, values, color in zip((1, 2), datasets, (BLUE, ORANGE)):
            ax.scatter(position + jitter, values, s=10, color=color,
                       alpha=0.72, linewidths=0, zorder=3)
        ax.set_xticks([1, 2], ["Conventional", r"$\mathrm{SO}(2)$"])
        ax.set_xlim(0.62, 2.38)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", length=0)
    fig.savefig(OUT / "reconstruction_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    fig, axes = plt.subplots(2, 1, figsize=(7.15, 4.7), constrained_layout=True)

    rows = tissue["rows"]
    budgets = np.array([row["labels_per_class"] for row in rows])
    normal_f1 = np.array([row["normal_vae"]["metrics"]["macro_f1"] for row in rows])
    so2_f1 = np.array([row["so2_vae"]["metrics"]["macro_f1"] for row in rows])
    axes[0].plot(budgets, normal_f1, "o-", color=BLUE, linewidth=2.0,
                 markersize=5, label="Conventional VAE", zorder=3)
    axes[0].plot(budgets, so2_f1, "s--", color=ORANGE, linewidth=2.0,
                 markersize=5, label=r"$\mathrm{SO}(2)$ VAE", zorder=3)
    axes[0].set_xscale("log")
    axes[0].xaxis.set_major_locator(FixedLocator(budgets))
    axes[0].xaxis.set_major_formatter(
        FixedFormatter(["250", "500", "1,000", "2,500", "5,671"])
    )
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylim(0.46, 0.80)
    axes[0].set_xlabel("Training labels per tissue class")
    axes[0].set_ylabel("Test macro-F1")
    axes[0].set_title("(a) Tissue recognition, 31,572 test patches", loc="left")
    axes[0].grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    axes[0].text(500, so2_f1[1] + 0.010, "*", ha="center", va="bottom",
                 color=ORANGE, fontsize=12, fontweight="bold")
    axes[0].spines["left"].set_visible(False)
    axes[0].tick_params(axis="both", length=0)

    names = ["Macro-F1", "Balanced accuracy", "Accuracy"]
    normal_values = [normal["macro_f1"], normal["balanced_accuracy"], normal["accuracy"]]
    so2_values = [so2["macro_f1"], so2["balanced_accuracy"], so2["accuracy"]]
    y = np.arange(len(names))[::-1]
    offset = 0.17
    axes[1].barh(y + offset, normal_values, height=0.28, color=BLUE,
                 label="Conventional VAE", zorder=3)
    axes[1].barh(y - offset, so2_values, height=0.28, color=ORANGE,
                 label=r"$\mathrm{SO}(2)$ VAE", zorder=3)
    for values, positions in ((normal_values, y + offset), (so2_values, y - offset)):
        for value, position in zip(values, positions, strict=True):
            axes[1].text(value + 0.012, position, f"{value:.3f}",
                         va="center", fontsize=8, fontweight="bold")
    axes[1].set_yticks(y, names)
    axes[1].set_xlim(0, 0.70)
    axes[1].set_xticks(np.arange(0, 0.71, 0.1))
    axes[1].set_ylim(-0.55, 2.55)
    axes[1].set_xlabel("Test score (higher is better)")
    axes[1].set_title("(b) Five-class WSI diagnosis, 23 test slides", loc="left")
    axes[1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2)
    axes[1].grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(axis="both", length=0)

    fig.savefig(OUT / "downstream_results.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def training_curves() -> None:
    # Each directory is a successive segment of the accepted 60,000-step run.
    branches = [
        (
            "Conventional VAE",
            BLUE,
            [
                "selected_runtime_full_v2",
                "selected_runtime_full_v3_session2",
                "selected_runtime_full_v4_session3",
            ],
        ),
        (
            r"$\mathrm{SO}(2)$ VAE",
            ORANGE,
            [
                "so2_selected_runtime_full_v1_session1",
                "so2_selected_runtime_full_v2_session2",
                "so2_selected_runtime_full_v3_session3",
                "so2_selected_runtime_full_v4_session4",
                "so2_selected_runtime_full_v5_session5",
                "so2_selected_runtime_full_session6_fresh_v1",
                "so2_selected_runtime_full_session7_fresh_v1_retry1",
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharey=True,
                             constrained_layout=True)
    for ax, (title, color, segments) in zip(axes, branches, strict=True):
        train_bins: dict[int, list[float]] = {}
        validation: dict[int, list[tuple[float, float, int]]] = {}
        for segment in segments:
            metrics = ROOT / "runs/kaggle" / segment / "metrics"
            with (metrics / "train_steps.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    step = int(row["optimizer_step"])
                    boundary = (step - 1) // 3000 * 3000 + 3000
                    train_bins.setdefault(boundary, []).append(float(row["recon_loss"]))
            with (metrics / "validation_metrics.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["view"] == "deterministic_denoising":
                        step = int(row["optimizer_step"])
                        validation.setdefault(step, []).append((
                            float(row["recon_loss"]),
                            float(row["recon_loss_std"]),
                            int(row["sample_count"]),
                        ))
        steps = np.array(sorted(validation))
        train = np.array([np.mean(train_bins[int(step)]) for step in steps])
        means = []
        spreads = []
        for step in steps:
            records = validation[int(step)]
            weights = np.array([count for _, _, count in records])
            mu = np.average([mean for mean, _, _ in records], weights=weights)
            second = np.average([std**2 + mean**2 for mean, std, _ in records],
                                weights=weights)
            means.append(mu)
            spreads.append(np.sqrt(max(0.0, second - mu**2)))
        means = np.array(means)
        spreads = np.array(spreads)
        ax.fill_between(steps / 1000, means - spreads, means + spreads,
                        color=color, alpha=0.14, linewidth=0,
                        label="Validation ±1 recorded SD")
        ax.plot(steps / 1000, train, ":", color=GRAY, linewidth=1.7,
                label="Train, 3k-step mean")
        ax.plot(steps / 1000, means, "o-", color=color, linewidth=1.8,
                markersize=3.3, label="Validation mean")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Optimizer updates (thousands)")
        ax.set_xlim(0, 61)
        ax.set_xticks([0, 15, 30, 45, 60])
        ax.set_ylim(0.07, 0.15)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel("Reconstruction loss")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(OUT / "vae_training_curves.png", dpi=300, bbox_inches="tight")
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
    reconstruction_mosaic()
    reconstruction_boxplots()
    downstream()
    training_curves()
    decoder_geometry()

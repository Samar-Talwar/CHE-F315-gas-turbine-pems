"""Stage 02: train-only PCA structure and K-Means operating regimes."""

from __future__ import annotations

from importlib import import_module

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pems_common import (
    FIGURES_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SENSORS,
    ensure_directories,
    random_split_indices,
)

load_data = import_module("00_load_data").load_data

TABLE03_COLUMNS = ["cluster", "n_members", "inertia"] + SENSORS


def main() -> None:
    ensure_directories()
    df = load_data()
    train_idx, _ = random_split_indices(len(df))
    train = df.iloc[train_idx].copy()

    scaler = StandardScaler().fit(train[SENSORS])
    scaled = scaler.transform(train[SENSORS])

    pca = PCA(random_state=RANDOM_STATE).fit(scaled)
    scores = pca.transform(scaled)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    raw_labels = kmeans.fit_predict(scaled)
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    tey_index = SENSORS.index("TEY")
    order = np.argsort(centroids_original[:, tey_index])
    relabel = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([relabel[int(label)] for label in raw_labels], dtype=int)
    centroids_ordered = centroids_original[order]
    centroids_ordered_scaled = centroids_scaled[order]
    members = np.bincount(labels, minlength=3)

    rows = []
    for cluster in range(3):
        row = {
            "cluster": cluster,
            "n_members": int(members[cluster]),
            "inertia": float(kmeans.inertia_),
        }
        row.update(
            {
                sensor: float(centroids_ordered[cluster, index])
                for index, sensor in enumerate(SENSORS)
            }
        )
        rows.append(row)
    table03 = pd.DataFrame(rows, columns=TABLE03_COLUMNS)
    table03.to_csv(RESULTS_DIR / "table03_kmeans_centroids.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    components = np.arange(1, len(ratios) + 1)
    axes[0].bar(components, ratios, color="#4c72b0", alpha=0.8, label="individual")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_xticks(components)
    twin = axes[0].twinx()
    twin.plot(
        components,
        cumulative,
        color="#c44e52",
        marker="o",
        lw=2,
        label="cumulative",
    )
    twin.set_ylabel("cumulative explained variance")
    twin.set_ylim(0, 1.02)
    axes[0].set_title("PCA scree and cumulative explained variance")
    lines_1, labels_1 = axes[0].get_legend_handles_labels()
    lines_2, labels_2 = twin.get_legend_handles_labels()
    axes[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    loadings = pca.components_.T
    arrow_scale = 3.2
    for index, sensor in enumerate(SENSORS):
        axes[1].arrow(
            0,
            0,
            arrow_scale * loadings[index, 0],
            arrow_scale * loadings[index, 1],
            color="#333333",
            width=0.008,
            head_width=0.09,
            length_includes_head=True,
        )
        axes[1].text(
            arrow_scale * loadings[index, 0] * 1.12,
            arrow_scale * loadings[index, 1] * 1.12,
            sensor,
            fontsize=9,
            ha="center",
            va="center",
            color="#b22222",
        )
    axes[1].axhline(0, color="grey", lw=0.7)
    axes[1].axvline(0, color="grey", lw=0.7)
    axes[1].set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    axes[1].set_title("PC1–PC2 biplot with sensor loadings")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_pca_scree_biplot.png", dpi=150)
    plt.close(fig)

    palette = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for cluster in range(3):
        mask = labels == cluster
        axes[0].scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=8,
            alpha=0.35,
            color=palette[cluster],
            label=f"regime {cluster} (n={int(mask.sum())})",
        )
    centroid_scores = pca.transform(centroids_ordered_scaled)
    axes[0].scatter(
        centroid_scores[:, 0],
        centroid_scores[:, 1],
        marker="X",
        s=260,
        c=["black"],
        edgecolors="white",
        linewidths=1.5,
        label="centroids",
        zorder=5,
    )
    axes[0].set_xlabel(f"PC1 ({ratios[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({ratios[1] * 100:.1f}%)")
    axes[0].set_title("K-Means regimes in PCA space")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].axis("off")
    cell_text = [
        [f"{value:.2f}" for value in centroids_ordered[cluster]]
        for cluster in range(3)
    ]
    table = axes[1].table(
        cellText=cell_text,
        rowLabels=[f"regime {cluster}" for cluster in range(3)],
        colLabels=SENSORS,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    axes[1].set_title("Original-unit centroid summaries", pad=24)
    fig.suptitle("K-Means (k=3) operating regimes — random 80% training partition")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_kmeans_regimes.png", dpi=150)
    plt.close(fig)

    print(f"Explained variance ratios: {np.round(ratios, 4)}")
    print(f"Cumulative variance: {np.round(cumulative, 4)}")
    print(table03.to_string(index=False))


if __name__ == "__main__":
    main()

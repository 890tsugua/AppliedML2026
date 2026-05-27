from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_religion_clusters(
    geo_metadata_path="country_cnn/data/country_metadata.csv",
    religion_cluster_path="country_cnn/data/country_religion_clusters.csv",
    output_dir="country_cnn/outputs/plots",
):
    geo = pd.read_csv(geo_metadata_path)
    religion = pd.read_csv(religion_cluster_path)

    df = geo.merge(religion, on="country")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(df["religion_group"].unique())
    cmap = plt.get_cmap("tab10")

    color_map = {
        group: cmap(i % 10)
        for i, group in enumerate(groups)
    }

    plt.figure(figsize=(15, 10))

    for group in groups:
        sub = df[df["religion_group"] == group]

        plt.scatter(
            sub["longitude"],
            sub["latitude"],
            label=group,
            s=100,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.6,
            color=color_map[group],
        )

    for _, row in df.iterrows():
        plt.text(
            row["longitude"] + 0.4,
            row["latitude"] + 0.4,
            row["country"],
            fontsize=7,
            bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1.2),
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Countries grouped by approximate religious-majority category")
    plt.grid(True, alpha=0.3)
    plt.legend(
        title="Religion group",
        fontsize=8,
        title_fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout()

    output_path = output_dir / "country_religion_clusters_geography.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"Saved religion cluster plot to: {output_path}")


if __name__ == "__main__":
    plot_religion_clusters()

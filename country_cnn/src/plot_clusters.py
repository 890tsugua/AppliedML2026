from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_clusters(
    metadata_path="country_cnn/data/country_metadata.csv",
    cluster_path="country_cnn/data/country_clusters.csv",
    output_dir="country_cnn/outputs/plots",
):
    metadata = pd.read_csv(metadata_path)
    clusters = pd.read_csv(cluster_path)

    df = metadata.merge(clusters, on="country")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: January vs July temperature
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        df["temp_jan"],
        df["temp_jul"],
        c=df["cluster"],
        s=80,
        alpha=0.8,
    )

    for _, row in df.iterrows():
        plt.text(
            row["temp_jan"] + 0.2,
            row["temp_jul"] + 0.2,
            row["country"],
            fontsize=7,
        )

    plt.xlabel("Average temperature in January (°C)")
    plt.ylabel("Average temperature in July (°C)")
    plt.title("Country clusters in climate space")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()

    temp_plot_path = output_dir / "country_clusters_temperature.png"
    plt.savefig(temp_plot_path, dpi=200)
    plt.close()

    print(f"Saved temperature plot to: {temp_plot_path}")

    # Plot 2: Longitude vs latitude
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["cluster"],
        s=80,
        alpha=0.8,
    )

    for _, row in df.iterrows():
        plt.text(
            row["longitude"] + 0.5,
            row["latitude"] + 0.5,
            row["country"],
            fontsize=7,
        )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Country clusters in geographic space")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.tight_layout()

    geo_plot_path = output_dir / "country_clusters_geography.png"
    plt.savefig(geo_plot_path, dpi=200)
    plt.close()

    print(f"Saved geography plot to: {geo_plot_path}")


if __name__ == "__main__":
    plot_clusters()

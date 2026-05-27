import pandas as pd
from pathlib import Path


def make_culture_clusters(
    culture_metadata_path="country_cnn/data/country_culture_metadata.csv",
    output_path="country_cnn/data/country_culture_clusters.csv",
):
    """
    Convert manually defined culture groups into cluster labels.

    Output columns:
        country, cluster, culture_group
    """

    culture_metadata_path = Path(culture_metadata_path)
    output_path = Path(output_path)

    df = pd.read_csv(culture_metadata_path)

    required_columns = ["country", "culture_group"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert culture group names to integer cluster labels
    df["cluster"] = pd.Categorical(df["culture_group"]).codes

    df_out = df[["country", "cluster", "culture_group"]].copy()
    df_out = df_out.sort_values(["cluster", "country"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Saved culture clusters to: {output_path}")

    print("\nCulture clusters:")
    for cluster_id in sorted(df_out["cluster"].unique()):
        group_name = df_out[df_out["cluster"] == cluster_id]["culture_group"].iloc[0]
        countries = df_out[df_out["cluster"] == cluster_id]["country"].tolist()
        print(f"Cluster {cluster_id} ({group_name}): {countries}")


if __name__ == "__main__":
    make_culture_clusters()

from pathlib import Path
import pandas as pd


def make_religion_clusters(
    religion_metadata_path="country_cnn/data/country_religion_metadata.csv",
    output_path="country_cnn/data/country_religion_clusters.csv",
):
    df = pd.read_csv(religion_metadata_path)

    required_columns = ["country", "religion_group"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["cluster"] = pd.Categorical(df["religion_group"]).codes

    df_out = df[["country", "cluster", "religion_group"]].copy()
    df_out = df_out.sort_values(["cluster", "country"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    print(f"Saved religion clusters to: {output_path}")

    print("\nReligion clusters:")
    for cluster_id in sorted(df_out["cluster"].unique()):
        group = df_out[df_out["cluster"] == cluster_id]["religion_group"].iloc[0]
        countries = df_out[df_out["cluster"] == cluster_id]["country"].tolist()
        print(f"Cluster {cluster_id} ({group}): {countries}")


if __name__ == "__main__":
    make_religion_clusters()

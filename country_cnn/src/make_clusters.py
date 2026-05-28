from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def make_country_clusters(
    metadata_path="country_cnn/data/country_metadata.csv",
    output_path="country_cnn/data/country_clusters.csv",
    n_clusters=5,
):
    df = pd.read_csv(metadata_path)

    df["abs_latitude"] = df["latitude"].abs()

    features = [
        "abs_latitude",
        "longitude",
        "temp_jan",
        "temp_jul",
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(n_clusters, len(df))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_df = df[["country", "cluster"]].copy()
    cluster_df = cluster_df.sort_values(["cluster", "country"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_path, index=False)

    print(f"Saved clusters to: {output_path}")

    print("\nClusters:")
    for cluster_id in sorted(cluster_df["cluster"].unique()):
        countries = cluster_df[cluster_df["cluster"] == cluster_id]["country"].tolist()
        print(f"Cluster {cluster_id}: {countries}")


if __name__ == "__main__":
    make_country_clusters()

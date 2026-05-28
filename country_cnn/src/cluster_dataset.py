import pandas as pd
from torch.utils.data import Dataset


class CountryToClusterDataset(Dataset):
    """
    Wraps an ImageFolder dataset and replaces country labels with cluster labels.
    """

    def __init__(self, imagefolder_dataset, cluster_csv_path):
        self.dataset = imagefolder_dataset

        cluster_df = pd.read_csv(cluster_csv_path)
        cluster_df["cluster"] = cluster_df["cluster"].astype(int)

        self.country_to_cluster = dict(
            zip(cluster_df["country"], cluster_df["cluster"])
        )

        self.idx_to_country = {
            idx: country
            for country, idx in imagefolder_dataset.class_to_idx.items()
        }

        dataset_countries = set(imagefolder_dataset.classes)
        cluster_countries = set(self.country_to_cluster.keys())

        missing = sorted(dataset_countries - cluster_countries)

        if missing:
            raise ValueError(
                "These image folder countries are missing from the cluster CSV:\n"
                f"{missing}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, country_label = self.dataset[idx]

        country_name = self.idx_to_country[country_label]
        cluster_label = self.country_to_cluster[country_name]

        return image, cluster_label

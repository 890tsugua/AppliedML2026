import pandas as pd
from pathlib import Path


def create_metadata_template(
    country_to_idx_path="country_cnn/country_to_idx.csv",
    output_path="country_cnn/data/country_metadata.csv",
):
    country_to_idx_path = Path(country_to_idx_path)
    output_path = Path(output_path)

    countries = []

    with open(country_to_idx_path, "r") as f:
        for line in f:
            country, idx = line.strip().split(",")
            countries.append(country)

    df = pd.DataFrame({
        "country": countries,
        "latitude": "",
        "longitude": "",
        "temp_jan": "",
        "temp_jul": "",
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved metadata template to: {output_path}")
    print(df)


if __name__ == "__main__":
    create_metadata_template()

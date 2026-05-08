from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_params() -> dict:
    with Path("params.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    test_size = float(params["data"]["test_size"])
    random_state = int(params["data"]["random_state"])

    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=4,
        n_classes=2,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Prepared data written to data/processed/")


if __name__ == "__main__":
    main()

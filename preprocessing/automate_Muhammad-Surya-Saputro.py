from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = ("mean", "se", "worst")
BASE_FEATURES = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
)


def column_names() -> list[str]:
    features = [f"{name}_{group}" for group in FEATURE_GROUPS for name in BASE_FEATURES]
    return ["id", "diagnosis", *features]


def load_raw_data(raw_path: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_path)
    df = pd.read_csv(raw_path, header=None, names=column_names())
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df = clean_df.drop(columns=["id"])
    clean_df = clean_df.drop_duplicates()
    clean_df["target"] = clean_df["diagnosis"].map({"B": 0, "M": 1}).astype(int)
    clean_df = clean_df.drop(columns=["diagnosis"])

    feature_columns = [col for col in clean_df.columns if col != "target"]
    scaler = StandardScaler()
    clean_df[feature_columns] = scaler.fit_transform(clean_df[feature_columns])
    return clean_df


def split_and_save(processed_df: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        processed_df,
        test_size=0.2,
        random_state=42,
        stratify=processed_df["target"],
    )

    paths = {
        "full": output_dir / "breast_cancer_processed.csv",
        "train": output_dir / "train.csv",
        "test": output_dir / "test.csv",
    }
    processed_df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


def run_preprocessing(
    raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = Path(raw_path) if raw_path else base_dir / "breast_cancer_raw" / "wdbc.data"
    output_dir = Path(output_dir) if output_dir else base_dir / "preprocessing" / "breast_cancer_preprocessing"

    raw_df = load_raw_data(raw_path)
    processed_df = preprocess_data(raw_df)
    split_and_save(processed_df, output_dir)
    return processed_df


if __name__ == "__main__":
    result = run_preprocessing()
    print(f"Preprocessing selesai. Shape dataset: {result.shape}")


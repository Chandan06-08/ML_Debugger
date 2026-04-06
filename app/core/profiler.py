import pandas as pd

def profile_data(path):
    df = pd.read_csv(path)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "class_distribution": df["target"].value_counts().to_dict() if "target" in df.columns else None
    }
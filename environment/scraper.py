import pandas as pd

def load_historical_fgi(filepath="fear-greed-2011-2023.csv"):
    df = pd.read_csv(filepath)

    # Let pandas infer the format and normalize
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()

    # Drop any rows where date couldn't be parsed
    df = df.dropna(subset=["Date"])

    # Rename for consistency
    df = df.rename(columns={"Fear Greed": "FearGreedIndex"})

    return df[["Date", "FearGreedIndex"]]

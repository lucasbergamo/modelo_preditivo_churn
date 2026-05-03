from src.data.load import load_bronze
from src.data.preprocess import bronze_to_silver, save_silver, silver_to_features
from src.data.split import save_gold, split_and_scale
from src.utils.reproducibility import set_global_seed


def run() -> None:
    set_global_seed()
    df_bronze = load_bronze()
    df_silver = bronze_to_silver(df_bronze)
    save_silver(df_silver)
    df_features = silver_to_features(df_silver)
    splits = split_and_scale(df_features)
    save_gold(splits)


if __name__ == "__main__":
    run()

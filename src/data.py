"""Đọc và chia dữ liệu."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Đọc CSV gốc (phân tách ';', có BOM) và chuẩn hóa tên cột."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]  # cột 'Daytime/evening attendance\t'
    return df


def split_data(df: pd.DataFrame):
    """Chia train/test 80/20 có phân tầng theo nhãn. Tập test chỉ dùng một lần ở cuối."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

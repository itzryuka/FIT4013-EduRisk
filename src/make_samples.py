"""Tạo các tệp CSV mẫu để thử chức năng "Dự đoán theo danh sách".

Mọi dòng đều lấy từ TẬP TEST (885 sinh viên mô hình chưa từng thấy khi huấn luyện),
nên kết quả dự đoán trên các tệp này là trung thực.

Chạy:  python -m src.make_samples   ->  ghi vào data/samples/
"""
import numpy as np
import pandas as pd

from .config import ROOT, SCENARIOS, TARGET
from .data import load_data, split_data

OUT = ROOT / "data" / "samples"


def _save(df, name, sep=";"):
    path = OUT / name
    df.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")
    print(f"  {name:<34} {len(df):>4} dòng × {df.shape[1]} cột (phân tách '{sep}')")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split_data(load_data())
    test = X_test.copy()
    test[TARGET] = y_test
    rng = np.random.RandomState(7)

    # 1) 30 SV có nhãn thật, mỗi lớp 10 người -> đối chiếu đúng/sai ngay trên web
    idx = [i for c in sorted(test[TARGET].unique())
           for i in test.index[test[TARGET] == c].to_series().sample(10, random_state=rng)]
    s1 = test.loc[idx]
    _save(s1.sample(frac=1, random_state=rng), "01_30_sv_co_nhan_that.csv")

    # 2) Khóa mới nhập học: chỉ 24 cột có lúc nhập học, không có nhãn -> chọn kịch bản "Lúc nhập học"
    s2 = test.sample(25, random_state=rng)[SCENARIOS["admission"]["columns"]]
    _save(s2, "02_khoa_moi_nhap_hoc.csv")

    # 3) Sau học kỳ 1: 30 cột, không có nhãn, phân tách bằng dấu phẩy (kiểm tra tự nhận dạng)
    s3 = test.sample(40, random_state=rng)[SCENARIOS["sem1"]["columns"]]
    _save(s3, "03_sau_hoc_ky_1.csv", sep=",")

    # 4) Toàn bộ tập test có nhãn -> kết quả trên web phải khớp với trang Kết quả mô hình
    _save(test, "04_toan_bo_tap_test.csv")

    # 5) Tệp có lỗi cố ý -> kiểm tra hệ thống báo dòng lỗi thay vì dừng hẳn
    s5 = test.sample(10, random_state=rng).astype(object)
    s5.iloc[2, s5.columns.get_loc("Age at enrollment")] = "hai mươi"
    s5.iloc[5, s5.columns.get_loc("Curricular units 1st sem (grade)")] = ""
    s5.iloc[8, s5.columns.get_loc("Admission grade")] = "N/A"
    _save(s5, "05_co_dong_loi.csv")


if __name__ == "__main__":
    print("Tạo tệp mẫu trong", OUT)
    main()

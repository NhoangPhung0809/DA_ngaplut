"""
Calibration Study - kiểm tra xác suất model dự đoán ra có ĐÁNG TIN hay không: khi model nói "70% khả
năng Ngập nặng", trong THỰC TẾ những lần model nói 70% đó có đúng khoảng 70% xảy ra ngập thật không?

Dùng CÙNG cấu hình "Full" như `ablation_study.py` (đủ 8 đặc trưng lag/rolling mưa + đủ luật nhãn gồm
rain_3day + SMOTE cân bằng lớp, RandomForestClassifier tham số cố định) để 2 nghiên cứu nhất quán với
nhau và có thể trích dẫn chung 1 bộ kết quả trong luận văn.

3 công cụ chuẩn:
- Reliability diagram (đường tin cậy): so trục X = xác suất model dự đoán, trục Y = tần suất THẬT xảy
  ra trong nhóm đó - đường chéo 45 độ là "hoàn hảo".
- Brier Score: sai số bình phương trung bình giữa xác suất dự đoán và nhãn thật (0/1) - càng THẤP càng
  tốt (0 = hoàn hảo).
- ECE (Expected Calibration Error): trung bình có trọng số của |độ chính xác thật - độ tin cậy dự đoán|
  qua các bin xác suất - càng THẤP càng tốt.

Đánh giá riêng từng lớp theo kiểu One-vs-Rest (vd "là Ngập nặng" vs "không phải Ngập nặng"), vì
Calibration Curve định nghĩa cho bài toán nhị phân.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from analyze_and_train import (
    FEATURE_COLS,
    TARGET_COL,
    build_daily_feature_dataset,
    chronological_train_test_split,
    create_multiclass_flood_label,
    load_and_concatenate_csvs,
    preprocess_features,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "data" / "calibration_study_results.json"

RF_FIXED_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
)

CLASS_NAMES = {0: "An toàn", 1: "Ngập nhẹ", 2: "Ngập nặng"}
N_BINS = 10


def compute_expected_calibration_error(y_true_binary: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    ECE = tổng có trọng số (theo số mẫu mỗi bin) của |tần suất thật - xác suất trung bình dự đoán|
    trong bin đó, qua tất cả các bin chia đều [0, 1].
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)

    total_samples = len(y_prob)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_indices == bin_id
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue
        bin_confidence = float(y_prob[mask].mean())
        bin_accuracy = float(y_true_binary[mask].mean())
        ece += (bin_count / total_samples) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def main() -> None:
    print("Đang nạp và gộp dữ liệu thô...")
    raw_df = load_and_concatenate_csvs()
    daily_feature_df = build_daily_feature_dataset(raw_df)
    labeled_df = create_multiclass_flood_label(daily_feature_df)
    modeling_df = preprocess_features(labeled_df)

    print("Chia tập train/test theo thời gian...")
    X_train, X_test, y_train, y_test = chronological_train_test_split(modeling_df, train_ratio=0.8)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

    print("Cân bằng dữ liệu train bằng SMOTE...")
    min_class_count = int(y_train.value_counts().min())
    smote_k_neighbors = min(5, max(1, min_class_count - 1))
    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    X_train_balanced_arr, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    X_train_balanced = pd.DataFrame(X_train_balanced_arr, columns=FEATURE_COLS)

    print("Huấn luyện RandomForestClassifier (tham số cố định, giống ablation_study.py)...")
    model = RandomForestClassifier(**RF_FIXED_PARAMS)
    model.fit(X_train_balanced, y_train_balanced)

    y_proba = model.predict_proba(X_test_scaled)
    class_order = list(model.classes_)

    per_class_results = []
    for class_label in [0, 1, 2]:
        if class_label not in class_order:
            continue
        class_idx = class_order.index(class_label)
        y_true_binary = (y_test.to_numpy() == class_label).astype(int)
        y_prob_class = y_proba[:, class_idx]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob_class, n_bins=N_BINS, strategy="uniform"
        )
        brier = float(brier_score_loss(y_true_binary, y_prob_class))
        ece = compute_expected_calibration_error(y_true_binary, y_prob_class, n_bins=N_BINS)

        print(
            f"\nLớp {class_label} - {CLASS_NAMES[class_label]}: "
            f"Brier Score = {brier:.4f} | ECE = {ece:.4f}"
        )

        per_class_results.append(
            {
                "class_label": class_label,
                "class_name": CLASS_NAMES[class_label],
                "brier_score": brier,
                "ece": ece,
                "n_positive": int(y_true_binary.sum()),
                "n_total": int(len(y_true_binary)),
                "reliability_curve": {
                    "mean_predicted_value": [float(v) for v in mean_predicted_value],
                    "fraction_of_positives": [float(v) for v in fraction_of_positives],
                },
            }
        )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_used": "RandomForestClassifier (tham số cố định, cấu hình Full giống ablation_study.py)",
        "fixed_params": RF_FIXED_PARAMS,
        "n_bins": N_BINS,
        "test_rows": int(len(X_test)),
        "results": per_class_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}\nTÓM TẮT CALIBRATION STUDY\n{'=' * 70}")
    for r in per_class_results:
        print(f"Lớp {r['class_label']} - {r['class_name']:<10} Brier={r['brier_score']:.4f}  ECE={r['ece']:.4f}")
    print(f"\nĐã lưu kết quả đầy đủ vào: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

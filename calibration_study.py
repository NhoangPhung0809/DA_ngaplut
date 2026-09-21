"""
Calibration Study - kiểm tra xác suất model dự đoán ra có ĐÁNG TIN hay không: khi model nói "70% khả
năng Ngập nặng", trong THỰC TẾ những lần model nói 70% đó có đúng khoảng 70% xảy ra ngập thật không?

Chạy 3 ARM trên CÙNG 1 tập test (giống cách `ablation_study.py` so sánh nhiều cấu hình) để trả lời câu
hỏi thực tế: "cân bằng dữ liệu (SMOTE/CTGAN) có làm lệch xác suất dự đoán so với KHÔNG cân bằng gì
không?":
  1. "SMOTE (cân bằng)"     : đủ 8 đặc trưng + đủ luật nhãn (gồm rain_3day) + SMOTE cân bằng lớp train
  2. "CTGAN (cân bằng)"     : Y HỆT arm 1, chỉ khác đúng 1 chỗ - dùng CTGAN thay SMOTE để cân bằng
  3. "None (không cân bằng)": Y HỆT arm 1, chỉ khác đúng 1 chỗ - KHÔNG cân bằng, giữ nguyên phân phối gốc

LÝ DO cần arm "None": tập train sau SMOTE/CTGAN có tỷ lệ lớp ~1:1:1 (cả 2 đều nhắm mục tiêu cân bằng
MỌI lớp bằng lớp đa số - xem `apply_gan_data_augmentation()`/`apply_smote_to_training_data()` trong
`analyze_and_train.py`), trong khi tập test vẫn giữ tỷ lệ THẬT (~98.5%/0.07%/1.3%) - lệch prior này
(huấn luyện trên 1 tỷ lệ, suy luận trên tỷ lệ khác) là nguyên nhân THẬT khiến model "quá tự tin" ở lớp
thiểu số khi soi qua reliability diagram (đã phát hiện qua code review). Arm "None" cho bằng chứng
THẬT về việc bỏ cân bằng có thực sự làm giảm hiện tượng này hay không, thay vì chỉ suy luận lý thuyết
suông. Arm "CTGAN" cho biết lệch prior này có phụ thuộc vào PHƯƠNG PHÁP sinh mẫu cụ thể (SMOTE nội suy
vs CTGAN sinh bằng GAN) hay không - dự đoán: KHÔNG, vì nguyên nhân là TỶ LỆ lớp sau cân bằng, không
phải chất lượng/cách sinh mẫu (xem giải thích đã trao đổi).

KHÔNG dùng trực tiếp `apply_gan_data_augmentation()`/`balance_training_data()` thật của
`analyze_and_train.py` - hàm đó có side-effect gọi `export_ctgan_comparison_artifacts()`, ghi ĐÈ lên
đúng file (`data_before_ctgan.csv`/`data_after_ctgan.csv`/`ctgan_class_distribution.json`) mà Tab 2
của app.py đang dùng để hiển thị "Cân bằng dữ liệu (CTGAN Before/After)" của LẦN TRAIN THẬT gần nhất -
chạy script nghiên cứu này sẽ âm thầm ghi đè, khiến Tab 2 hiển thị nhầm dữ liệu của lần chạy nghiên cứu
thay vì lần train thật. Script này tự gọi thẳng thư viện `ctgan.CTGAN`/`imblearn.SMOTE`, giống cách
`ablation_study.py` đã làm với SMOTE.

3 công cụ chuẩn (tính riêng cho MỖI arm):
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
    CTGAN,
    CTGAN_BATCH_SIZE,
    CTGAN_EPOCHS,
    CTGAN_MAX_TARGET_ROWS_PER_CLASS,
    CTGAN_MAX_TRAIN_ROWS_PER_CLASS,
    FEATURE_COLS,
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


def apply_ctgan_isolated(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Bản CTGAN augmentation TỰ CHỦ (không gọi `apply_gan_data_augmentation()` thật) - xem giải thích ở
    docstring đầu file (tránh side-effect ghi đè artifact CTGAN của Tab 2). Sinh mẫu cho lớp 1 và 2 lên
    tới `target_count` (bằng lớp đa số, giới hạn `CTGAN_MAX_TARGET_ROWS_PER_CLASS`) - CÙNG công thức
    `target_count`/`pac=1` như hàm thật để 2 kết quả có thể so sánh được, chỉ bỏ phần export/fallback
    phức tạp không cần thiết cho nghiên cứu này. Fallback về SMOTE nếu thiếu thư viện `ctgan` hoặc CTGAN
    lỗi khi chạy trên 1 lớp cụ thể - in cảnh báo rõ ràng thay vì crash cả script.
    """
    if CTGAN is None:
        print("Friendly warning: Không tìm thấy thư viện `ctgan`. Arm CTGAN fallback về SMOTE.")
        smote = SMOTE(random_state=42, k_neighbors=min(5, max(1, int(y_train.value_counts().min()) - 1)))
        X_arr, y_res = smote.fit_resample(X_train, y_train)
        return pd.DataFrame(X_arr, columns=FEATURE_COLS), y_res

    class_counts = y_train.value_counts().sort_index()
    majority_count = int(class_counts.max())
    target_count = min(majority_count, CTGAN_MAX_TARGET_ROWS_PER_CLASS)
    augmented_feature_frames = [X_train.reset_index(drop=True).copy()]
    augmented_target_series = [y_train.reset_index(drop=True).copy()]

    for class_label in [1, 2]:
        current_count = int(class_counts.get(class_label, 0))
        deficit = target_count - current_count
        if deficit <= 0:
            continue

        class_features = X_train.loc[y_train == class_label].copy()
        if len(class_features) > CTGAN_MAX_TRAIN_ROWS_PER_CLASS:
            class_features = class_features.sample(n=CTGAN_MAX_TRAIN_ROWS_PER_CLASS, random_state=42).reset_index(drop=True)

        try:
            gan_model = CTGAN(epochs=CTGAN_EPOCHS, batch_size=CTGAN_BATCH_SIZE, pac=1, verbose=False)
            gan_model.fit(class_features)
            synthetic_features = gan_model.sample(deficit)
            synthetic_features = synthetic_features[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
            synthetic_features = synthetic_features.fillna(class_features.median())
            synthetic_targets = pd.Series([class_label] * len(synthetic_features), name=y_train.name)
            augmented_feature_frames.append(synthetic_features.reset_index(drop=True))
            augmented_target_series.append(synthetic_targets.reset_index(drop=True))
            print(f"CTGAN generated {len(synthetic_features)} synthetic samples for class {class_label}.")
        except Exception as exc:
            print(f"Friendly warning: CTGAN failed for class {class_label} ({exc}). Bỏ qua, giữ nguyên lớp này.")

    X_train_balanced = pd.concat(augmented_feature_frames, ignore_index=True)
    y_train_balanced = pd.concat(augmented_target_series, ignore_index=True)
    shuffled_index = np.random.RandomState(42).permutation(len(X_train_balanced))
    return X_train_balanced.iloc[shuffled_index].reset_index(drop=True), y_train_balanced.iloc[shuffled_index].reset_index(drop=True)


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


def run_calibration_arm(
    arm_name: str,
    balancing_method: str,
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Huấn luyện 1 arm (SMOTE/CTGAN/None) rồi tính reliability diagram/Brier/ECE trên CÙNG 1 tập test -
    tách hàm này để `run_calibration_study()` gọi nhiều lần với `balancing_method` khác nhau, đảm bảo
    mọi bước khác (đặc trưng, luật nhãn, chia train/test, scaler) giữ NGUYÊN GIỐNG NHAU giữa các arm -
    chỉ cô lập đúng 1 biến đang so sánh (cân bằng bằng gì, hoặc không cân bằng), không lẫn hiệu ứng nào
    khác."""
    print(f"\n{'=' * 70}\nARM: {arm_name}\n{'=' * 70}")

    if balancing_method == "smote":
        min_class_count = int(y_train.value_counts().min())
        smote_k_neighbors = min(5, max(1, min_class_count - 1))
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train_balanced_arr, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        X_train_balanced = pd.DataFrame(X_train_balanced_arr, columns=FEATURE_COLS)
    elif balancing_method == "gan":
        X_train_balanced, y_train_balanced = apply_ctgan_isolated(X_train_scaled, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    print(f"Huấn luyện RandomForestClassifier (tham số cố định) - train_rows={len(X_train_balanced)}...")
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
            f"[{arm_name}] Lớp {class_label} - {CLASS_NAMES[class_label]}: "
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

    return {
        "arm_name": arm_name,
        "balancing_method": balancing_method,
        "train_rows": int(len(X_train_balanced)),
        "results": per_class_results,
    }


def run_calibration_study() -> None:
    """Đặt tên riêng (khác `main()` cũ) để `training_worker.py` import và gọi lại được sau khi huấn
    luyện chính hoàn tất - giống `run_ablation_study()` trong `ablation_study.py`, cùng lý do: tự động
    chạy kèm nút "Bắt đầu Huấn luyện Nền", không cần mở terminal chạy tay riêng nữa."""
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

    arms = [
        run_calibration_arm("SMOTE (cân bằng)", "smote", X_train_scaled, y_train, X_test_scaled, y_test),
        run_calibration_arm("CTGAN (cân bằng)", "gan", X_train_scaled, y_train, X_test_scaled, y_test),
        run_calibration_arm("None (không cân bằng)", "none", X_train_scaled, y_train, X_test_scaled, y_test),
    ]

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_used": "RandomForestClassifier (tham số cố định, cấu hình Full giống ablation_study.py)",
        "fixed_params": RF_FIXED_PARAMS,
        "n_bins": N_BINS,
        "test_rows": int(len(X_test)),
        "arms": arms,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}\nTÓM TẮT CALIBRATION STUDY (SMOTE vs None)\n{'=' * 70}")
    for arm in arms:
        print(f"\n--- {arm['arm_name']} ---")
        for r in arm["results"]:
            print(f"Lớp {r['class_label']} - {r['class_name']:<10} Brier={r['brier_score']:.4f}  ECE={r['ece']:.4f}")
    print(f"\nĐã lưu kết quả đầy đủ vào: {RESULTS_PATH}")


if __name__ == "__main__":
    run_calibration_study()

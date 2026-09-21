"""
Ablation Study - đo tác động THẬT của từng thành phần trong pipeline bằng cách bỏ CHÍNH XÁC 1 thành
phần ra khỏi cấu hình "Full" mỗi lần, rồi so sánh F1-Macro trên CÙNG 1 tập test - trả lời câu hỏi hội
đồng hay hỏi: "mỗi thành phần đóng góp bao nhiêu, có thật sự cần không?".

DÙNG RandomForestClassifier VỚI THAM SỐ CỐ ĐỊNH (giống hệt `build_model_registry()` trong
`analyze_and_train.py`) CHO CẢ 4 CẤU HÌNH - không tune lại hyperparameter riêng cho từng cấu hình, vì
mục tiêu là cô lập đúng 1 biến đang test, không lẫn với hiệu ứng tối ưu hyperparameter khác nhau.

DÙNG SMOTE (không phải CTGAN) làm phương pháp cân bằng cho các arm CÓ cân bằng - lý do: hàm thật
`apply_gan_data_augmentation()`/`apply_smote_to_training_data()` trong `analyze_and_train.py` hardcode
đúng 8 tên cột `FEATURE_COLS` khi đóng gói kết quả, sẽ lỗi ở arm "(-) Lag/Rolling" (chỉ còn 5 cột) - nên
script này tự gọi `imblearn.SMOTE` trực tiếp, kiểm soát đúng tên cột đang dùng ở từng arm, đồng thời
nhanh và xác định hơn CTGAN (không ngẫu nhiên qua epoch train GAN) - phù hợp cho việc so sánh có kiểm
soát giữa 4 arm.

4 cấu hình (leave-one-out so với Full):
  1. Full              : đủ 8 đặc trưng (gồm lag/rolling mưa) + luật nhãn đủ (gồm rain_3day) + SMOTE
  2. (-) Lag/Rolling    : bỏ 3 cột lag/rolling mưa khỏi đặc trưng đầu vào - giữ nguyên luật nhãn + SMOTE
  3. (-) rain_3day rule : bỏ điều kiện mưa tích luỹ 3 ngày khỏi LUẬT NHÃN - giữ nguyên đặc trưng + SMOTE
  4. (-) Cân bằng lớp   : không cân bằng gì cả (giữ nguyên phân phối gốc) - giữ nguyên đặc trưng + luật nhãn

KHÔNG lặp lại ablation cho "T -> T+1 shift" (chống rò rỉ) ở đây - đã có bằng chứng thật từ trước (F1
LSTM giảm 0.9263 -> 0.4876 sau khi sửa rò rỉ), không cần chạy lại.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from analyze_and_train import (
    FEATURE_COLS,
    build_daily_feature_dataset,
    chronological_train_test_split,
    create_multiclass_flood_label,
    load_and_concatenate_csvs,
    preprocess_features,
)
from shared_constants import RAIN_LAG1_COL, RAIN_LAG2_COL, RAIN_ROLLING_3D_COL

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "data" / "ablation_study_results.json"

RF_FIXED_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
)

LAG_COLS = [RAIN_LAG1_COL, RAIN_LAG2_COL, RAIN_ROLLING_3D_COL]


def run_arm(name: str, modeling_df: pd.DataFrame, drop_lag_features: bool, apply_balancing: bool) -> dict:
    print(f"\n{'=' * 70}\nARM: {name}\n{'=' * 70}")

    X_train, X_test, y_train, y_test = chronological_train_test_split(modeling_df, train_ratio=0.8)

    active_cols = [c for c in FEATURE_COLS if not (drop_lag_features and c in LAG_COLS)]
    X_train = X_train[active_cols]
    X_test = X_test[active_cols]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=active_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=active_cols, index=X_test.index)

    if apply_balancing:
        min_class_count = int(y_train.value_counts().min())
        smote_k_neighbors = min(5, max(1, min_class_count - 1))
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train_balanced_arr, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        X_train_balanced = pd.DataFrame(X_train_balanced_arr, columns=active_cols)
        balancing_method = "smote"
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        balancing_method = "none"

    model = RandomForestClassifier(**RF_FIXED_PARAMS)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test_scaled)

    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"F1-Macro: {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "arm_name": name,
        "n_features": len(active_cols),
        "features_used": active_cols,
        "balancing_method": balancing_method,
        "train_rows": int(len(X_train_balanced)),
        "test_rows": int(len(X_test)),
        "f1_macro": f1_macro,
        "per_class_report": report,
    }


def run_ablation_study() -> None:
    """Đặt tên riêng (khác `main()`) để `training_worker.py` import và gọi lại được sau khi huấn luyện
    chính hoàn tất - GÓP Ý CỦA GVHD/người dùng: tự động chạy kèm nút "Bắt đầu Huấn luyện Nền", giống
    Time Series CV đã làm, thay vì phải mở terminal chạy tay `python3 ablation_study.py` riêng."""
    print("Đang nạp và gộp dữ liệu thô...")
    raw_df = load_and_concatenate_csvs()
    daily_feature_df = build_daily_feature_dataset(raw_df)

    print("\nXây dựng 2 phiên bản nhãn (đủ luật / bỏ rain_3day)...")
    labeled_full = create_multiclass_flood_label(daily_feature_df, include_rain_3day=True)
    labeled_no_rain3day = create_multiclass_flood_label(daily_feature_df, include_rain_3day=False)

    modeling_df_full = preprocess_features(labeled_full)
    modeling_df_no_rain3day = preprocess_features(labeled_no_rain3day)

    results = []
    results.append(
        run_arm("1. Full (đủ đặc trưng + đủ luật nhãn + SMOTE)", modeling_df_full, drop_lag_features=False, apply_balancing=True)
    )
    results.append(
        run_arm("2. (-) Lag/Rolling mưa", modeling_df_full, drop_lag_features=True, apply_balancing=True)
    )
    results.append(
        run_arm("3. (-) Điều kiện rain_3day trong luật nhãn", modeling_df_no_rain3day, drop_lag_features=False, apply_balancing=True)
    )
    results.append(
        run_arm("4. (-) Cân bằng dữ liệu (giữ nguyên phân phối gốc)", modeling_df_full, drop_lag_features=False, apply_balancing=False)
    )

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_used": "RandomForestClassifier (tham số cố định, không tune riêng từng arm)",
        "fixed_params": RF_FIXED_PARAMS,
        "note": "T->T+1 shift KHÔNG ablate ở đây - đã có bằng chứng thật từ trước (LSTM F1 0.9263 -> 0.4876 khi sửa rò rỉ).",
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 70}\nTÓM TẮT ABLATION STUDY\n{'=' * 70}")
    baseline_f1 = results[0]["f1_macro"]
    for r in results:
        delta = r["f1_macro"] - baseline_f1
        marker = "" if r is results[0] else f" (Δ so với Full: {delta:+.4f})"
        print(f"{r['arm_name']:<55} F1-Macro = {r['f1_macro']:.4f}{marker}")
    print(f"\nĐã lưu kết quả đầy đủ vào: {RESULTS_PATH}")


if __name__ == "__main__":
    run_ablation_study()

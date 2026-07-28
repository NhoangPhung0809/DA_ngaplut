import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "historical"
PLOTS_DIR = BASE_DIR / "plots"
LATEST_MODELS_DIR = BASE_DIR / "models" / "latest"

TIME_COL = "Thời_gian"
TARGET_COL = "Nguy_cơ_ngập"
FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
]
EDA_COLS = [*FEATURE_COLS, TARGET_COL]
CLASS_NAME_MAP = {
    0: "An toàn",
    1: "Ngập nhẹ",
    2: "Ngập nặng",
}


def ensure_output_directory() -> None:
    """Tạo thư mục lưu biểu đồ và báo cáo phân tích."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_combine_historical_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Đọc toàn bộ dữ liệu lịch sử từ các file CSV và ghép thành một bảng duy nhất."""
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy file CSV nào trong thư mục: {data_dir}")

    dataframes = []
    print(f"Tìm thấy {len(csv_files)} file CSV trong {data_dir}")

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df["Nguồn_dữ_liệu"] = Path(file_path).stem
        dataframes.append(df)
        print(f"Đã đọc {Path(file_path).name}: {len(df)} dòng")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df[TIME_COL] = pd.to_datetime(combined_df[TIME_COL], errors="coerce")
    combined_df = combined_df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    print(f"Tổng số dòng sau khi ghép: {len(combined_df)}")
    return combined_df


def create_multiclass_flood_label(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo lại nhãn 3 lớp theo đúng luật đang dùng trong pipeline huấn luyện."""
    labeled_df = df.copy()

    for column in FEATURE_COLS:
        if column not in labeled_df.columns:
            labeled_df[column] = 0.0
        labeled_df[column] = pd.to_numeric(labeled_df[column], errors="coerce")
        labeled_df[column] = labeled_df[column].fillna(labeled_df[column].median())

    rain = labeled_df["Lượng_mưa_mm"].fillna(0)
    soil = labeled_df["Độ_ẩm_đất"].fillna(0)
    tide = labeled_df["Chiều_cao_triều_m"].fillna(0)

    heavy_flood_mask = (
        (rain > 50)
        | ((rain > 30) & (soil > 0.45))
        | ((rain > 20) & (soil > 0.40) & (tide > 1.50))
        | (tide > 2.50)
    )
    light_flood_mask = (
        (rain > 25)
        | ((rain > 15) & (soil > 0.30))
        | ((rain > 10) & (tide > 1.20))
    )

    labeled_df[TARGET_COL] = 0
    labeled_df.loc[light_flood_mask, TARGET_COL] = 1
    labeled_df.loc[heavy_flood_mask, TARGET_COL] = 2
    return labeled_df


def preprocess_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu và giữ lại các cột cần cho EDA."""
    processed_df = df.copy()

    for column in FEATURE_COLS:
        processed_df[column] = pd.to_numeric(processed_df[column], errors="coerce")
        processed_df[column] = processed_df[column].fillna(processed_df[column].median())

    processed_df[TARGET_COL] = (
        pd.to_numeric(processed_df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    )
    processed_df["Mức_độ_ngập"] = processed_df[TARGET_COL].map(CLASS_NAME_MAP)

    return processed_df[[TIME_COL, "Nguồn_dữ_liệu", *EDA_COLS, "Mức_độ_ngập"]]


def save_correlation_heatmap(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    """Vẽ và lưu correlation heatmap Pearson."""
    correlation_df = df[EDA_COLS].corr(method="pearson")
    output_path = PLOTS_DIR / "correlation_heatmap.png"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Ma trận tương quan Pearson của các biến khí tượng - thủy văn")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path, correlation_df


def save_class_distribution_plot(df: pd.DataFrame) -> tuple[Path, pd.Series]:
    """Vẽ và lưu biểu đồ phân bố lớp mục tiêu."""
    class_counts = df[TARGET_COL].value_counts().sort_index()
    plot_df = df.copy()
    output_path = PLOTS_DIR / "class_distribution.png"
    class_palette = {
        0: "#4C78A8",
        1: "#F58518",
        2: "#E45756",
    }

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(
        data=plot_df,
        x=TARGET_COL,
        hue=TARGET_COL,
        order=sorted(CLASS_NAME_MAP.keys()),
        palette=class_palette,
        dodge=False,
        legend=False,
        saturation=1,
    )
    for patch in ax.patches:
        patch.set_edgecolor("#1F1F1F")
        patch.set_linewidth(1.2)
        patch.set_alpha(0.95)

    ax.set_title("Phân bố lớp mục tiêu Nguy cơ ngập")
    ax.set_xlabel("Lớp nguy cơ ngập")
    ax.set_ylabel("Số lượng quan sát")
    tick_positions = list(range(len(CLASS_NAME_MAP)))
    tick_labels = [f"{cls} - {CLASS_NAME_MAP[cls]}" for cls in sorted(CLASS_NAME_MAP)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{int(height):,}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path, class_counts


def save_grouped_distribution_plot(
    df: pd.DataFrame,
    feature_col: str,
    output_name: str,
    chart_title: str,
    x_label: str,
) -> tuple[Path, pd.DataFrame]:
    """Vẽ histogram + KDE cho một biến, phân tách theo lớp mục tiêu."""
    output_path = PLOTS_DIR / output_name
    summary_stats = (
        df.groupby(TARGET_COL)[feature_col]
        .agg(["mean", "median", "std", "min", "max"])
        .reindex(sorted(CLASS_NAME_MAP.keys()))
        .round(2)
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=feature_col,
        hue="Mức_độ_ngập",
        bins=40,
        kde=True,
        stat="density",
        common_norm=False,
        element="step",
        palette="viridis",
        alpha=0.35,
    )
    plt.title(chart_title)
    plt.xlabel(x_label)
    plt.ylabel("Mật độ phân bố")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path, summary_stats


def format_summary_table(summary_df: pd.DataFrame) -> str:
    """Định dạng bảng thống kê mô tả ngắn gọn để chèn vào báo cáo chữ."""
    renamed_df = summary_df.copy()
    renamed_df.index = [f"Lớp {idx} - {CLASS_NAME_MAP[idx]}" for idx in renamed_df.index]
    return renamed_df.to_string()


def sync_latest_model_artifacts_to_plots() -> list[Path]:
    """Đồng bộ artifact mô hình mới nhất sang thư mục plots để tránh lệch file cũ."""
    synced_paths = []
    artifact_names = ["feature_importance.png", "confusion_matrix.png"]

    for artifact_name in artifact_names:
        source_path = LATEST_MODELS_DIR / artifact_name
        destination_path = PLOTS_DIR / artifact_name
        if source_path.exists():
            destination_path.write_bytes(source_path.read_bytes())
            synced_paths.append(destination_path)

    return synced_paths


def save_eda_metadata(df: pd.DataFrame) -> Path:
    """Lưu metadata để biết EDA đang dùng nhãn nào và dữ liệu nào."""
    metadata = {
        "label_source": "rule_based_multiclass_regenerated",
        "target_definition": {
            "0": "An toàn",
            "1": "Ngập nhẹ",
            "2": "Ngập nặng",
        },
        "total_rows": int(len(df)),
        "class_distribution": {
            str(key): int(value) for key, value in df[TARGET_COL].value_counts().sort_index().items()
        },
        "note": (
            "EDA khong dung truc tiep nhan cu trong CSV. "
            "Script tao lai nhan 3 lop theo cung logic rule-based voi pipeline huan luyen hien tai."
        ),
    }
    output_path = PLOTS_DIR / "eda_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_heatmap_analysis(correlation_df: pd.DataFrame) -> str:
    """Sinh đoạn phân tích luận văn cho heatmap tương quan."""
    target_corr = correlation_df[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    strongest_positive_feature = target_corr.index[0]
    strongest_positive_value = target_corr.iloc[0]
    strongest_negative_feature = target_corr.index[-1]
    strongest_negative_value = target_corr.iloc[-1]

    soil_rain_corr = correlation_df.loc["Lượng_mưa_mm", "Độ_ẩm_đất"]
    rain_tide_corr = correlation_df.loc["Lượng_mưa_mm", "Chiều_cao_triều_m"]

    return f"""
THESIS ANALYSIS 1. CORRELATION HEATMAP

Biểu đồ heatmap Pearson cho thấy biến `{strongest_positive_feature}` có mức tương quan dương mạnh nhất với biến mục tiêu `Nguy_cơ_ngập` với hệ số xấp xỉ {strongest_positive_value:.2f}. Điều này hàm ý rằng khi `{strongest_positive_feature}` gia tăng, xác suất xuất hiện trạng thái ngập nhẹ hoặc ngập nặng cũng tăng theo. Kết quả này phù hợp với bản chất vật lý của bài toán, vì các tín hiệu khí tượng và thủy văn cực đoan thường là tác nhân kích hoạt trực tiếp rủi ro ngập.

Ngoài ra, mối tương quan giữa `Lượng_mưa_mm` và `Độ_ẩm_đất` đạt khoảng {soil_rain_corr:.2f}, cho thấy lượng mưa lớn có xu hướng làm đất bão hòa ẩm. Đây là một quan sát quan trọng trong bối cảnh thủy văn khu vực Huế, nơi mưa kéo dài kết hợp với nền đất giữ nước có thể làm giảm khả năng thấm, khiến nước mặt tích tụ nhanh hơn và làm gia tăng nguy cơ ngập cục bộ.

Tương quan giữa `Lượng_mưa_mm` và `Chiều_cao_triều_m` ở mức {rain_tide_corr:.2f}, phản ánh rằng hai biến này không hoàn toàn đồng biến nhưng có thể cộng hưởng trong các đợt rủi ro cao. Trên thực tế, ngập lụt ở vùng ven biển và đầm phá không chỉ chịu tác động của mưa mà còn chịu ảnh hưởng của triều cường, khiến quá trình tiêu thoát nước bị chậm lại. Vì vậy, việc đồng thời đưa dữ liệu mưa, độ ẩm đất và triều vào mô hình là hợp lý về cả mặt thống kê lẫn mặt chuyên ngành.

Ở chiều ngược lại, biến `{strongest_negative_feature}` có hệ số tương quan thấp nhất với `Nguy_cơ_ngập`, xấp xỉ {strongest_negative_value:.2f}. Điều này cho thấy biến này ít mang thông tin trực tiếp hơn cho việc nhận diện ngập so với nhóm biến mưa, đất và triều. Tuy nhiên, biến này vẫn có giá trị hỗ trợ vì nó có thể phản ánh trạng thái nền khí quyển chung, giúp mô hình học được bối cảnh thời tiết trước khi xuất hiện hiện tượng ngập.
""".strip()


def build_class_distribution_analysis(class_counts: pd.Series, total_rows: int) -> str:
    """Sinh đoạn phân tích luận văn cho biểu đồ phân bố lớp."""
    class_ratios = (class_counts / total_rows * 100).round(2)
    majority_class = int(class_counts.idxmax())
    minority_class = int(class_counts.idxmin())
    class_0_count = int(class_counts.get(0, 0))
    class_1_count = int(class_counts.get(1, 0))
    class_2_count = int(class_counts.get(2, 0))

    return f"""
THESIS ANALYSIS 2. CLASS DISTRIBUTION

Biểu đồ phân bố lớp cho thấy bộ dữ liệu có sự mất cân bằng rất mạnh giữa ba mức nguy cơ ngập sau khi tái tạo nhãn theo luật chuyên gia 3 lớp. Cụ thể, lớp 0 ({CLASS_NAME_MAP[0]}) có {class_0_count:,} quan sát, lớp 1 ({CLASS_NAME_MAP[1]}) có {class_1_count:,} quan sát và lớp 2 ({CLASS_NAME_MAP[2]}) có {class_2_count:,} quan sát. Trong đó, lớp {majority_class} ({CLASS_NAME_MAP[majority_class]}) chiếm khoảng {class_ratios.loc[majority_class]:.2f}% tổng số quan sát, còn lớp {minority_class} ({CLASS_NAME_MAP[minority_class]}) chỉ chiếm khoảng {class_ratios.loc[minority_class]:.2f}%.

Mất cân bằng lớp có ý nghĩa thực tiễn rất lớn. Nếu không xử lý phù hợp, mô hình có thể thiên về dự đoán lớp chiếm đa số và bỏ sót các trường hợp ngập nhẹ hoặc ngập nặng, vốn là những lớp có giá trị cảnh báo cao nhất. Kết quả EDA này do đó củng cố tính cần thiết của bước cân bằng dữ liệu bằng SMOTE trong pipeline huấn luyện.

Về mặt học thuật, cần lưu ý rằng phân bố này được tạo ra từ nhãn rule-based đang dùng trong pipeline huấn luyện hiện tại, chứ không lấy nguyên trạng nhãn cũ có sẵn trong một số file CSV lịch sử. Điều này giúp toàn bộ EDA, huấn luyện mô hình và dashboard sử dụng cùng một định nghĩa mục tiêu, tránh tình trạng biểu đồ và phần diễn giải dựa trên các phiên bản nhãn khác nhau.
""".strip()


def build_rainfall_distribution_analysis(summary_stats: pd.DataFrame) -> str:
    """Sinh đoạn phân tích luận văn cho phân bố lượng mưa theo lớp."""
    safe_mean = float(summary_stats.loc[0, "mean"])
    light_mean = float(summary_stats.loc[1, "mean"])
    heavy_mean = float(summary_stats.loc[2, "mean"])
    heavy_median = float(summary_stats.loc[2, "median"])
    safe_median = float(summary_stats.loc[0, "median"])
    if heavy_mean >= light_mean:
        trend_paragraph = (
            f"Giá trị trung bình của lượng mưa tăng từ khoảng {safe_mean:.2f} mm ở lớp an toàn "
            f"lên {light_mean:.2f} mm ở lớp ngập nhẹ và đạt {heavy_mean:.2f} mm ở lớp ngập nặng. "
            f"Trung vị của lớp ngập nặng cũng cao hơn đáng kể so với lớp an toàn "
            f"({heavy_median:.2f} mm so với {safe_median:.2f} mm), xác nhận rằng mưa là biến "
            "phân tách mạnh giữa các mức độ ngập."
        )
    else:
        trend_paragraph = (
            f"Giá trị trung bình của lượng mưa tăng mạnh từ khoảng {safe_mean:.2f} mm ở lớp an toàn "
            f"lên {light_mean:.2f} mm ở lớp ngập nhẹ, trong khi lớp ngập nặng có giá trị trung bình "
            f"khoảng {heavy_mean:.2f} mm. Mẫu hình này cho thấy lớp ngập nặng trong bộ dữ liệu không "
            "chỉ do mưa cực lớn quyết định mà còn chịu tác động đáng kể của các điều kiện kết hợp, "
            "đặc biệt là triều cường và độ bão hòa ẩm của đất. Tuy vậy, trung vị của lớp ngập nặng "
            f"vẫn cao hơn lớp an toàn ({heavy_median:.2f} mm so với {safe_median:.2f} mm), chứng tỏ "
            "mưa vẫn là một tín hiệu quan trọng trong nhận diện rủi ro."
        )

    return f"""
THESIS ANALYSIS 3. RAINFALL DISTRIBUTION BY TARGET CLASS

Biểu đồ phân bố `Lượng_mưa_mm` theo lớp mục tiêu cho thấy lượng mưa vẫn là biến phân tách quan trọng giữa các trạng thái ngập. {trend_paragraph}

Phân bố này cho thấy mưa là điều kiện kích hoạt quan trọng của rủi ro ngập, đặc biệt trong quá trình chuyển từ lớp an toàn sang lớp ngập nhẹ. Tuy nhiên, với lớp ngập nặng, số liệu cho thấy mức độ nghiêm trọng không nhất thiết đi kèm lượng mưa trung bình lớn nhất, mà còn phụ thuộc mạnh vào tổ hợp điều kiện thủy văn khác. Đây là dấu hiệu cho thấy trong bộ dữ liệu hiện tại, triều cường đóng vai trò chi phối rõ rệt đối với lớp ngập nặng.

Mặt khác, phần đuôi phải kéo dài của phân bố mưa cho thấy tồn tại các thời điểm mưa rất lớn, đại diện cho những sự kiện cực trị. Những quan sát này đặc biệt quan trọng đối với mô hình dự báo, vì chúng chứa tín hiệu mạnh cho các lớp cảnh báo cao. Do đó, biến lượng mưa cần được xem là một trong những biến đầu vào cốt lõi trong toàn bộ hệ thống dự báo ngập.

Bảng thống kê mô tả:
{format_summary_table(summary_stats)}
""".strip()


def build_tide_distribution_analysis(summary_stats: pd.DataFrame) -> str:
    """Sinh đoạn phân tích luận văn cho phân bố chiều cao triều theo lớp."""
    safe_mean = float(summary_stats.loc[0, "mean"])
    light_mean = float(summary_stats.loc[1, "mean"])
    heavy_mean = float(summary_stats.loc[2, "mean"])
    heavy_max = float(summary_stats.loc[2, "max"])

    return f"""
THESIS ANALYSIS 4. TIDE HEIGHT DISTRIBUTION BY TARGET CLASS

Biểu đồ phân bố `Chiều_cao_triều_m` cho thấy các lớp có mức nguy cơ ngập cao thường đi kèm nền triều cao hơn. Giá trị trung bình chiều cao triều tăng từ khoảng {safe_mean:.2f} m ở lớp an toàn lên {light_mean:.2f} m ở lớp ngập nhẹ và đạt {heavy_mean:.2f} m ở lớp ngập nặng. Đặc biệt, lớp ngập nặng có thể xuất hiện ở các ngưỡng triều cực đại đến khoảng {heavy_max:.2f} m, cho thấy vai trò khuếch đại của triều cường trong quá trình hình thành ngập.

Phát hiện này có ý nghĩa thực tiễn rõ rệt đối với khu vực Huế, nơi chịu tác động đồng thời của mưa lớn, hệ thống sông ngòi dày đặc và ảnh hưởng từ biển/đầm phá. Khi mực triều tăng cao, nước trong hệ thống thoát ra cửa biển chậm hơn, làm gia tăng hiện tượng dồn ứ nước mặt. Vì vậy, cùng một lượng mưa nhưng trong điều kiện triều cao, rủi ro ngập có thể nghiêm trọng hơn đáng kể.

Kết quả EDA cũng cho thấy chiều cao triều không phải là biến duy nhất quyết định ngập, nhưng là biến bổ trợ quan trọng khi kết hợp với lượng mưa và độ ẩm đất. Điều này phù hợp với cách xây dựng nhãn rule-based cũng như định hướng mô hình hóa đa nguồn dữ liệu trong đề tài.

Bảng thống kê mô tả:
{format_summary_table(summary_stats)}
""".strip()


def build_word_ready_report(
    heatmap_analysis: str,
    class_analysis: str,
    rain_analysis: str,
    tide_analysis: str,
) -> str:
    """Ghép toàn bộ nội dung phân tích thành báo cáo hoàn chỉnh."""
    return "\n\n".join(
        [
            "PHÂN TÍCH THĂM DÒ DỮ LIỆU (EDA) CHO BÀI TOÁN DỰ BÁO NGUY CƠ NGẬP",
            (
                "Lưu ý phương pháp: Toàn bộ các biểu đồ và nhận xét dưới đây sử dụng nhãn "
                "`Nguy_cơ_ngập` được tái tạo lại theo rule-based 3 lớp giống pipeline huấn luyện hiện tại, "
                "không dùng trực tiếp nhãn cũ có thể còn lưu trong một số file CSV lịch sử."
            ),
            heatmap_analysis,
            class_analysis,
            rain_analysis,
            tide_analysis,
        ]
    )


def save_text_report(report_text: str) -> Path:
    """Lưu phần phân tích chữ để có thể copy trực tiếp vào Word."""
    output_path = PLOTS_DIR / "thesis_analysis_report.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def main() -> None:
    """Chạy toàn bộ pipeline EDA và sinh commentary cho luận văn."""
    print("=== BẮT ĐẦU EDA CHO BỘ DỮ LIỆU DỰ BÁO NGẬP ===")
    ensure_output_directory()

    df = load_and_combine_historical_data()
    df = create_multiclass_flood_label(df)
    df = preprocess_for_eda(df)
    metadata_path = save_eda_metadata(df)
    synced_artifacts = sync_latest_model_artifacts_to_plots()

    heatmap_path, correlation_df = save_correlation_heatmap(df)
    class_plot_path, class_counts = save_class_distribution_plot(df)
    rain_plot_path, rain_stats = save_grouped_distribution_plot(
        df=df,
        feature_col="Lượng_mưa_mm",
        output_name="rain_distribution_by_class.png",
        chart_title="Phân bố lượng mưa theo lớp nguy cơ ngập",
        x_label="Lượng mưa (mm)",
    )
    tide_plot_path, tide_stats = save_grouped_distribution_plot(
        df=df,
        feature_col="Chiều_cao_triều_m",
        output_name="tide_distribution_by_class.png",
        chart_title="Phân bố chiều cao triều theo lớp nguy cơ ngập",
        x_label="Chiều cao triều (m)",
    )

    heatmap_analysis = build_heatmap_analysis(correlation_df)
    class_analysis = build_class_distribution_analysis(class_counts, total_rows=len(df))
    rain_analysis = build_rainfall_distribution_analysis(rain_stats)
    tide_analysis = build_tide_distribution_analysis(tide_stats)

    report_text = build_word_ready_report(
        heatmap_analysis=heatmap_analysis,
        class_analysis=class_analysis,
        rain_analysis=rain_analysis,
        tide_analysis=tide_analysis,
    )
    report_path = save_text_report(report_text)

    print("\n=== ĐÃ LƯU CÁC BIỂU ĐỒ ===")
    print(f"- Correlation heatmap       : {heatmap_path}")
    print(f"- Class distribution        : {class_plot_path}")
    print(f"- Rain distribution         : {rain_plot_path}")
    print(f"- Tide distribution         : {tide_plot_path}")
    print(f"- Thesis analysis report    : {report_path}")
    print(f"- EDA metadata              : {metadata_path}")
    if synced_artifacts:
        print("- Synced latest model plots :")
        for synced_path in synced_artifacts:
            print(f"  * {synced_path}")

    print("\n=== NỘI DUNG PHÂN TÍCH SẴN DÙNG CHO LUẬN VĂN ===\n")
    print(report_text)


if __name__ == "__main__":
    main()

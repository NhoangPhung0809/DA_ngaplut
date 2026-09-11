"""
Script tạo tài liệu kỹ thuật (.docx) mô tả hệ thống Dự báo Ngập lụt Thừa Thiên Huế.
Chạy: python generate_documentation.py
Kết quả: TAI_LIEU_KY_THUAT_HE_THONG.docx
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_cell_background(cell, color_hex: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return h


def add_para(doc, text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    return p


def add_bullets(doc, items):
    for item in items:
        if isinstance(item, tuple):
            text, sub_items = item
            p = doc.add_paragraph(text, style="List Bullet")
            for sub in sub_items:
                doc.add_paragraph(sub, style="List Bullet 2")
        else:
            doc.add_paragraph(item, style="List Bullet")


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for p in hdr_cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_background(hdr_cells[i], "1F4E79")
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()
    return table


doc = Document()

# ---- Style mặc định ----
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

# =====================================================================================
# TRANG BÌA
# =====================================================================================
title = doc.add_heading("TÀI LIỆU KỸ THUẬT HỆ THỐNG", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = sub.add_run("Hệ thống Dự báo & Cảnh báo Ngập lụt Thừa Thiên Huế")
sub_run.font.size = Pt(16)
sub_run.bold = True

sub2 = doc.add_paragraph()
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub2_run = sub2.add_run(
    "Giải thích chức năng từng tab, công nghệ sử dụng, và cơ sở lựa chọn độ đo/biểu đồ đánh giá"
)
sub2_run.font.size = Pt(12)
sub2_run.italic = True

doc.add_page_break()

# =====================================================================================
# MỤC LỤC (thủ công - liệt kê để người đọc dễ tra cứu, không dùng field TOC động)
# =====================================================================================
add_heading(doc, "MỤC LỤC", level=1)
toc_items = [
    "1. Kiến trúc tổng quan hệ thống",
    "2. Chi tiết từng Tab trong ứng dụng",
    "   2.1. Tab 1 - Dự báo 4 ngày tới",
    "   2.2. Tab 2 - Khám phá Dữ liệu (EDA)",
    "   2.3. Tab 3 - Tiền xử lý & Huấn luyện",
    "   2.4. Tab 4 - Đánh giá Mô hình",
    "   2.5. Tab 5 - Bản đồ Tránh ngập",
    "3. Công nghệ sử dụng & Lý do lựa chọn",
    "4. Giải thích các độ đo đánh giá mô hình (Metrics)",
    "5. Giải thích các biểu đồ/sơ đồ trong hệ thống",
    "6. Trả lời các câu hỏi thường gặp về đề tài",
    "7. Chi tiết từng thuật toán/kỹ thuật sử dụng trong đồ án",
    "   7.1. Nhóm Thống kê (Statistical)",
    "   7.2. Nhóm Machine Learning",
    "   7.3. Nhóm Chuỗi thời gian (Time Series)",
    "   7.4. Nhóm Deep Learning",
    "   7.5. Nhóm Hybrid",
    "   7.6. Kỹ thuật cân bằng dữ liệu (Data Balancing)",
    "   7.7. Kỹ thuật chia dữ liệu & tinh chỉnh siêu tham số",
]
for item in toc_items:
    add_para(doc, item)
doc.add_page_break()

# =====================================================================================
# 1. KIẾN TRÚC TỔNG QUAN
# =====================================================================================
add_heading(doc, "1. Kiến trúc tổng quan hệ thống", level=1)
add_para(
    doc,
    "Hệ thống gồm 3 khối chính, tách biệt rõ trách nhiệm (separation of concerns), giúp dễ bảo trì "
    "và đúng chuẩn quy trình một dự án Machine Learning thực tế:",
)
add_table(
    doc,
    headers=["File / Module", "Vai trò"],
    rows=[
        ["fetch_data.py", "Thu thập dữ liệu khí tượng - thủy văn lịch sử (10 năm) từ Open-Meteo Archive API cho 5 địa phương, gán nhãn rule-based, lưu vào data/historical/*.csv."],
        ["analyze_and_train.py", "Pipeline huấn luyện chính: tiền xử lý, chia train/test theo thời gian, cân bằng lớp (CTGAN/SMOTE), huấn luyện nhiều nhóm mô hình (Statistical/ML/Deep Learning/Hybrid), chọn best model theo F1-Macro TUYỆT ĐỐI (không phân biệt loại mô hình), xuất deployment_config.json."],
        ["eda_analysis.py", "Sinh các biểu đồ thăm dò dữ liệu (EDA) tĩnh (heatmap, phân phối lớp, xu hướng theo tháng...) lưu vào plots/, dùng lại trong Tab EDA của app."],
        ["hyperparameter_tuning.py", "Module chuẩn (Optuna/GridSearchCV) minh họa chiến lược tinh chỉnh siêu tham số cho từng nhóm mô hình."],
        ["app.py", "Ứng dụng Streamlit - giao diện người dùng cuối, gồm 5 tab, đọc kết quả từ 2 module trên để hiển thị."],
        ["shared_constants.py", "Hằng số dùng chung (tên cột đặc trưng FEATURE_COLS...) giữa các file, tránh khai báo trùng lặp/lệch nhau."],
    ],
    col_widths=[5, 11],
)
add_para(
    doc,
    "Luồng dữ liệu tổng thể: fetch_data.py → data/historical/*.csv → analyze_and_train.py (huấn luyện, "
    "xuất models/latest/) → app.py (đọc models/latest/ để hiển thị và suy luận). Đây là mô hình MLOps "
    "cơ bản: tách rời quá trình HUẤN LUYỆN (batch, chạy nền) khỏi quá trình PHỤC VỤ (serving, real-time "
    "trên giao diện web), giúp huấn luyện lại mô hình không làm gián đoạn người dùng đang xem dashboard.",
)

doc.add_page_break()

# =====================================================================================
# 2. CHI TIẾT TỪNG TAB
# =====================================================================================
add_heading(doc, "2. Chi tiết từng Tab trong ứng dụng", level=1)
add_para(
    doc,
    "Ứng dụng có 5 tab, được sắp xếp theo thứ tự: TAB DỰ BÁO (kết quả thực tế, đặt đầu tiên vì đây là "
    "giá trị cốt lõi người dùng cuối cần thấy ngay) → 4 tab còn lại bám sát vòng đời Data Science "
    "(Data Science Lifecycle): Khám phá dữ liệu → Tiền xử lý & Huấn luyện → Đánh giá → Ứng dụng thực tế "
    "(bản đồ).",
)

# ---- Tab 1 ----
add_heading(doc, "2.1. Tab 1 - 🔮 Dự báo 4 ngày tới", level=2)
add_para(doc, "Nhiệm vụ:", bold=True)
add_bullets(
    doc,
    [
        "Hiển thị bảng dự báo nguy cơ ngập cho 4 ngày kế tiếp (T = hôm nay, T+1, T+2, T+3), cho toàn bộ "
        "5 địa phương giám sát, dùng CHÍNH model đã huấn luyện (không phải số liệu giả lập).",
        "Đây là trang đầu tiên người dùng thấy khi mở app, vì đây là SẢN PHẨM ĐẦU RA thực tế của toàn bộ "
        "hệ thống - trả lời trực tiếp câu hỏi 'mô hình dự báo được gì', trước khi đi vào các tab kỹ thuật "
        "giải thích PHƯƠNG PHÁP phía sau.",
    ],
)
add_para(doc, "Công nghệ & cách hoạt động:", bold=True)
add_bullets(
    doc,
    [
        "Open-Meteo Forecast API (khác Archive API dùng để lấy dữ liệu lịch sử): lấy dữ liệu thời tiết "
        "DỰ BÁO theo ngày (nhiệt độ, độ ẩm, lượng mưa, độ ẩm đất) cho 4 ngày tới, tự động theo múi giờ "
        "địa phương (timezone=auto) để đảm bảo ngày T luôn đúng là 'hôm nay' theo giờ Việt Nam.",
        "Chiều cao triều: ưu tiên Open-Meteo Marine API (dữ liệu thật, chỉ có ở vùng ven biển); nếu "
        "không có (tọa độ nội địa), dùng công thức triều tổng hợp (bán nhật triều 12.42 giờ + chu kỳ mặt "
        "trăng 29.53 ngày) - ĐÚNG công thức đã dùng khi tạo dữ liệu huấn luyện, đảm bảo tính nhất quán.",
        "Dispatch theo loại model (model_type đọc từ deployment_config.json): "
        "model dạng bảng (sklearn_tabular) đưa từng ngày vào như 1 dòng độc lập; model dạng chuỗi "
        "(keras_sequence: GRU/LSTM/CNN) hoặc Hybrid cần một CỬA SỔ nhiều ngày liên tiếp - hệ thống tự "
        "gọi Open-Meteo với cả past_days (quá khứ thật) và forecast_days=4 (tương lai) để dựng đủ cửa sổ "
        "đúng quy ước lúc huấn luyện.",
        "Cache theo ngày (st.cache_data persist=disk): chỉ gọi lại API/model khi cache đã qua ngày mới "
        "hoặc người dùng bấm 'Dự báo lại' - tránh gọi API lãng phí mỗi lần có người mở trang.",
    ],
)

# ---- Tab 2 ----
add_heading(doc, "2.2. Tab 2 - 📊 Khám phá Dữ liệu (EDA)", level=2)
add_para(doc, "Nhiệm vụ:", bold=True)
add_bullets(
    doc,
    [
        "Hiểu bản chất dữ liệu TRƯỚC khi làm sạch/huấn luyện - bước bắt buộc trong mọi quy trình Data "
        "Science, giúp phát hiện vấn đề (giá trị thiếu, ngoại lai, mất cân bằng lớp) sớm.",
    ],
)
add_para(doc, "Bố cục & nội dung:", bold=True)
add_bullets(
    doc,
    [
        "Dữ liệu thô (Raw Data): 20 dòng đầu của dữ liệu đã gộp từ 5 file CSV lịch sử.",
        "Thống kê mô tả (Descriptive Statistics): bảng describe() - khoảng giá trị, trung bình, độ lệch "
        "chuẩn từng biến, để phát hiện đơn vị đo bất thường.",
        "6 biểu đồ phân phối/tương quan (chi tiết ở mục 5).",
        "Xử lý giá trị thiếu & ngoại lai: bảng đếm số lượng/tỷ lệ thiếu theo cột, và bảng ngoại lai bằng "
        "CẢ IQR lẫn Z-score, tính RIÊNG cho từng lớp nguy cơ ngập (giải thích chi tiết ở mục 4).",
    ],
)

# ---- Tab 3 ----
add_heading(doc, "2.3. Tab 3 - ⚙️ Tiền xử lý & Huấn luyện", level=2)
add_para(doc, "Nhiệm vụ:", bold=True)
add_bullets(
    doc,
    [
        "Làm sạch dữ liệu, chia tập train/test đúng đặc thù chuỗi thời gian, cân bằng lớp thiểu số, và "
        "cung cấp cụm điều khiển để khởi chạy huấn luyện thật ngay trên giao diện.",
    ],
)
add_heading(doc, "2.3.0. Xử lý dữ liệu khuyết (Missing Value) - làm TRƯỚC mọi bước khác", level=3)
add_para(
    doc,
    "Dữ liệu thô có 2 dạng thiếu khác nhau, và mỗi dạng được xử lý theo 1 cách RIÊNG (không dùng "
    "chung 1 cách cho mọi cột) - xem hàm compute_train_only_medians(), create_multiclass_flood_label() "
    "và preprocess_features() trong analyze_and_train.py.",
)
add_table(
    doc,
    ["Loại cột bị thiếu", "Cách xử lý", "Vì sao"],
    [
        (
            "Cột số liệu đầu vào (Lượng_mưa_mm, Độ_ẩm_đất, Chiều_cao_triều_m, ...)",
            "ĐIỀN bằng giá trị TRUNG VỊ (median) của chính cột đó.",
            "Median ổn định hơn trung bình (mean) khi dữ liệu mưa/triều có nhiều giá trị cực đoan "
            "(outlier) - vài trận mưa rất lớn không kéo lệch giá trị điền vào như mean sẽ bị.",
        ),
        (
            "Cột thời gian (Thời_gian) hoặc nhãn ngập (Nguy_cơ_ngập) sau khi dịch chuỗi",
            "XOÁ luôn dòng đó (KHÔNG điền).",
            "Không có cách điền hợp lý cho ngày tháng hay nhãn ngập của tương lai chưa xảy ra - điền "
            "bừa sẽ tạo ra nhãn/thời điểm SAI, còn nguy hiểm hơn là thiếu dữ liệu.",
        ),
    ],
    col_widths=[5.5, 5.5, 6],
)
add_para(
    doc,
    "Điểm kỹ thuật quan trọng - median được tính THẾ NÀO để không rò rỉ dữ liệu (data leakage):",
    bold=True,
)
add_bullets(
    doc,
    [
        "Median KHÔNG tính trên toàn bộ dữ liệu, mà chỉ tính trên PHẦN SẼ THUỘC TẬP TRAIN (80% dòng đầu "
        "theo thời gian của MỖI địa phương - đúng ranh giới mà bước chia Train/Test bên dưới sẽ dùng).",
        "Lý do: nếu tính median trên CẢ tập test rồi mới điền, thì tập train sẽ vô tình 'biết trước' "
        "thống kê của tập test - đúng định nghĩa rò rỉ dữ liệu, khiến kết quả đánh giá model bị ảo (cao "
        "hơn thực tế) khi so với lúc chạy dự báo thật (chỉ có dữ liệu quá khứ).",
        "Áp dụng nhất quán ở 2 nơi: lúc tạo nhãn rule-based (create_multiclass_flood_label) và lúc tạo "
        "ma trận feature cuối cùng (preprocess_features) - cả 2 đều gọi chung 1 hàm "
        "compute_train_only_medians() để tránh 2 nơi tính ra 2 con số khác nhau.",
    ],
)

add_para(doc, "Các thành phần chính (sau khi đã xử lý xong dữ liệu khuyết ở trên):", bold=True)
add_bullets(
    doc,
    [
        "Dữ liệu đã làm sạch: gọi lại ĐÚNG hàm preprocess_features() trong analyze_and_train.py (không "
        "viết lại logic riêng ở app.py) để tránh 2 nơi xử lý cho ra kết quả khác nhau.",
        (
            "Chia Train/Test theo TimeSeriesSplit - KHÔNG dùng train_test_split(shuffle=True) hay K-Fold "
            "thường:",
            [
                "Lý do: dữ liệu là CHUỖI THỜI GIAN. Nếu xáo trộn ngẫu nhiên, dữ liệu TƯƠNG LAI có thể lọt "
                "vào tập huấn luyện (data leakage), khiến độ chính xác đánh giá bị 'ảo' - cao hơn nhiều so "
                "với khi mô hình chạy thật trong thực tế (lúc đó chỉ có dữ liệu quá khứ).",
                "TimeSeriesSplit là cross-validation kiểu walk-forward: mỗi fold sau luôn dùng NHIỀU dữ "
                "liệu quá khứ hơn để dự báo một đoạn tương lai kế tiếp, mô phỏng đúng bối cảnh vận hành "
                "thực tế.",
            ],
        ),
        (
            "Cân bằng dữ liệu bằng CTGAN (fallback SMOTE):",
            [
                "Lý do cần cân bằng: dữ liệu ngập lụt LUÔN mất cân bằng nặng - phần lớn thời gian 'An "
                "toàn', lớp 'Ngập nặng' rất hiếm dù là lớp quan trọng nhất cần dự báo đúng.",
                "CTGAN (Conditional Tabular GAN): sinh dữ liệu tổng hợp CÓ ĐIỀU KIỆN theo lớp thiểu số, "
                "học được phân phối thống kê phức tạp của dữ liệu bảng thật, thường cho chất lượng tốt "
                "hơn SMOTE (vốn chỉ nội suy tuyến tính giữa các điểm lân cận) với dữ liệu nhiều chiều.",
                "SMOTE dùng làm phương án dự phòng khi CTGAN lỗi/không cài được (CTGAN cần nhiều tài "
                "nguyên và thời gian huấn luyện hơn).",
            ],
        ),
        (
            "Tinh chỉnh siêu tham số - GridSearchCV vs Optuna:",
            [
                "GridSearchCV: dùng cho Random Forest - không gian tham số nhỏ, rời rạc (3 tham số, mỗi "
                "tham số 3-4 giá trị = 48 tổ hợp) nên duyệt vét cạn được, kết quả tái lập 100%.",
                "Optuna (Bayesian Optimization - TPE): dùng cho XGBoost/LSTM - không gian tham số lớn, "
                "liên tục, và có yếu tố KIẾN TRÚC (số lớp/số unit của mạng neural) mà GridSearchCV không "
                "biểu diễn hiệu quả được (combinatorial explosion nếu duyệt vét cạn).",
            ],
        ),
        "Cụm điều khiển MLOps: chọn danh sách mô hình cần huấn luyện, chọn phương pháp cân bằng, khởi "
        "chạy huấn luyện NỀN (background process, không chặn giao diện), xem log trực tiếp trên web.",
    ],
)

# ---- Tab 4 ----
add_heading(doc, "2.4. Tab 4 - 📈 Đánh giá Mô hình", level=2)
add_para(doc, "Nhiệm vụ:", bold=True)
add_bullets(
    doc,
    [
        "So sánh hiệu năng TẤT CẢ các mô hình đã huấn luyện (Thống kê/Machine Learning/Deep "
        "Learning/Hybrid) trên CÙNG một tập test, và rút ra khuyến nghị quản trị (nối kết quả kỹ thuật "
        "với hành động thực tế).",
    ],
)
add_para(doc, "Nội dung hiển thị:", bold=True)
add_bullets(
    doc,
    [
        "Bảng chỉ số đầy đủ: Accuracy, Precision (Macro), Recall (Macro), F1 (Macro) cho mọi model, sắp "
        "xếp theo F1 giảm dần.",
        "Biểu đồ cột ngang so sánh F1-Score toàn bộ mô hình.",
        "Biểu đồ nhóm Top 5 mô hình theo Accuracy/Precision/Recall.",
        "Confusion Matrix và Feature Importance (dạng ảnh, cho model dạng bảng).",
        "Đường cong ROC-AUC theo chiến lược One-vs-Rest (OvR) cho bài toán 3 lớp.",
        "(Xem giải thích Ý NGHĨA và LÝ DO PHÙ HỢP của từng độ đo ở Mục 4.)",
    ],
)

# ---- Tab 5 ----
add_heading(doc, "2.5. Tab 5 - 🗺️ Bản đồ Tránh ngập", level=2)
add_para(doc, "Nhiệm vụ:", bold=True)
add_bullets(
    doc,
    [
        "Ứng dụng THỰC TẾ của mô hình dự báo: giám sát trạng thái ngập của 5 địa phương, và tính tuyến "
        "đường di chuyển TỰ ĐỘNG né vùng đang ngập.",
    ],
)
add_para(doc, "Công nghệ & thiết kế:", bold=True)
add_bullets(
    doc,
    [
        "TomTom Routing API: tính tuyến đường THẬT bám theo mạng lưới đường (snap-to-road), tham số "
        "travelMode=motorcycle - vì phương tiện phổ biến nhất khi ngập cục bộ tại đô thị Việt Nam là xe "
        "máy (len lỏi được đường nhỏ/hẻm mà ô tô không đi được), routeType=fastest, traffic=true.",
        "Cơ chế avoidAreas: TomTom CHỈ hỗ trợ khai báo vùng cấm dạng HÌNH CHỮ NHẬT (rectangles), không "
        "hỗ trợ đa giác tự do - hệ thống tự động quy đổi vùng ngập (polygon) thành hình chữ nhật bao "
        "ngoài nhỏ nhất trước khi gửi API, trong khi vẫn vẽ đúng polygon gốc lên bản đồ cho trực quan.",
        "Chọn điểm đi/đến BẤT KỲ bằng cách click trực tiếp lên bản đồ (không giới hạn trong 5 điểm giám "
        "sát) - dùng streamlit-folium để đọc lại toạ độ click.",
        "TỰ ĐỘNG tính lại tuyến khi có địa phương chuyển sang trạng thái 'Ngập' - không bắt buộc người "
        "dùng phải bấm nút, dùng cơ chế 'chữ ký' (điểm đi, điểm đến, danh sách nơi đang ngập) để chỉ gọi "
        "lại API khi thực sự có gì thay đổi (tiết kiệm lượt gọi API).",
        "Folium: thư viện vẽ bản đồ tương tác (dựa trên Leaflet.js) - vẽ marker giám sát (xanh=an toàn/"
        "đỏ=ngập), vùng ngập tô đỏ, và tuyến đường vẽ xanh dương.",
    ],
)

doc.add_page_break()

# =====================================================================================
# 3. CÔNG NGHỆ SỬ DỤNG
# =====================================================================================
add_heading(doc, "3. Công nghệ sử dụng & Lý do lựa chọn", level=1)
add_table(
    doc,
    headers=["Công nghệ", "Vai trò trong hệ thống", "Lý do lựa chọn"],
    rows=[
        ["Streamlit", "Framework xây dựng giao diện web", "Cho phép viết giao diện tương tác hoàn toàn bằng Python, không cần biết HTML/CSS/JS, phù hợp để nhanh chóng đưa mô hình ML ra giao diện thực tế trong phạm vi đồ án/luận văn."],
        ["scikit-learn", "Random Forest, KNN, SVC, AdaBoost, StandardScaler, các độ đo đánh giá", "Thư viện ML chuẩn công nghiệp, ổn định, API nhất quán, dễ tích hợp với các thư viện khác (XGBoost/LightGBM/CatBoost đều tương thích API sklearn)."],
        ["XGBoost / LightGBM / CatBoost", "Các mô hình Gradient Boosting cho bài toán phân loại đa lớp", "3 thư viện boosting mạnh nhất hiện nay cho dữ liệu dạng bảng (tabular) - thường cho độ chính xác cao hơn các mô hình đơn lẻ, có cơ chế xử lý mất cân bằng lớp/regularization tốt."],
        ["TensorFlow / Keras", "Mô hình Deep Learning (LSTM/GRU/1D-CNN/CNN-LSTM) và Hybrid", "Chuẩn công nghiệp cho Deep Learning, hỗ trợ tốt dữ liệu chuỗi thời gian (sequence) - phù hợp để mô hình học được các mẫu hình biến thiên khí tượng - thủy văn qua nhiều ngày liên tiếp."],
        ["CTGAN", "Sinh dữ liệu tổng hợp cân bằng lớp thiểu số", "Mô hình GAN chuyên biệt cho dữ liệu dạng bảng có điều kiện (conditional), học được phân phối thống kê thật của dữ liệu, khắc phục hạn chế của các phương pháp oversampling truyền thống (SMOTE)."],
        ["Optuna", "Tinh chỉnh siêu tham số cho XGBoost/LSTM", "Thuật toán Bayesian Optimization (TPE) hội tụ nhanh hơn Grid/Random Search trên không gian tham số lớn, hỗ trợ pruning (cắt tỉa sớm trial kém) để tiết kiệm thời gian."],
        ["Open-Meteo API", "Nguồn dữ liệu khí tượng lịch sử (Archive) và dự báo (Forecast)", "API thời tiết miễn phí, độ phân giải giờ, có dữ liệu lịch sử dài (10 năm) và dự báo theo ngày, đủ đáp ứng nhu cầu huấn luyện lẫn suy luận thời gian thực mà không tốn chi phí."],
        ["TomTom Routing API", "Tính tuyến đường thực tế né vùng ngập", "Một trong số ít API định tuyến hỗ trợ tham số avoidAreas (né vùng cấm) và travelMode=motorcycle - phù hợp trực tiếp với bài toán chỉ đường né ngập cho xe máy."],
        ["Folium / streamlit-folium", "Vẽ bản đồ tương tác trong Streamlit", "Tích hợp trực tiếp bản đồ Leaflet.js vào Streamlit, hỗ trợ đọc sự kiện click từ người dùng - cần thiết cho tính năng chọn điểm đi/đến tuỳ ý."],
        ["Plotly", "Vẽ biểu đồ tương tác (bar chart, ROC curve)", "Biểu đồ có thể phóng to/thu nhỏ, hover xem số liệu chi tiết - phù hợp trình bày trên dashboard hơn biểu đồ tĩnh."],
    ],
    col_widths=[3.5, 6, 7.5],
)

doc.add_page_break()

# =====================================================================================
# 4. GIẢI THÍCH CÁC ĐỘ ĐO ĐÁNH GIÁ (METRICS)
# =====================================================================================
add_heading(doc, "4. Giải thích các độ đo đánh giá mô hình (Metrics)", level=1)
add_para(
    doc,
    "Bài toán của đồ án là PHÂN LOẠI ĐA LỚP (3 lớp: 0-An toàn, 1-Ngập nhẹ, 2-Ngập nặng) trên dữ liệu "
    "MẤT CÂN BẰNG NẶNG (lớp 'An toàn' chiếm đa số áp đảo). Việc lựa chọn độ đo đánh giá PHẢI tính đến "
    "đặc điểm này - đây là lý do vì sao đồ án KHÔNG chỉ dùng Accuracy đơn thuần. Mục 4.0 dưới đây trình "
    "bày ĐÚNG SỐ LIỆU THẬT của tập dữ liệu đồ án đang dùng, làm căn cứ định lượng cho toàn bộ lập luận "
    "lựa chọn độ đo ở các mục sau.",
)

add_heading(doc, "4.0. Bối cảnh: mức độ mất cân bằng THẬT của dữ liệu đồ án", level=2)
add_para(
    doc,
    "Theo kết quả EDA thực tế trên toàn bộ dữ liệu lịch sử 5 địa phương (file plots/eda_metadata.json, "
    "sinh bởi eda_analysis.py), phân bố 3 lớp nhãn Nguy_cơ_ngập như sau:",
)
add_table(
    doc,
    headers=["Lớp", "Ý nghĩa", "Số quan sát", "Tỷ lệ trong tổng 438.120 dòng"],
    rows=[
        ["0", "An toàn", "431.776", "98,55%"],
        ["1", "Ngập nhẹ", "325", "0,07%"],
        ["2", "Ngập nặng", "6.019", "1,37%"],
    ],
    col_widths=[2, 4, 4, 7],
)
add_para(
    doc,
    "Tỷ lệ mất cân bằng giữa lớp đa số (0) và lớp thiểu số hiếm nhất (1) lên tới khoảng 1.328 : 1 "
    "(431.776 / 325). Đây là mức mất cân bằng CỰC ĐOAN, không phải mất cân bằng nhẹ thông thường - hệ "
    "quả trực tiếp:",
)
add_bullets(
    doc,
    [
        "Một mô hình 'ngu' luôn dự đoán lớp 0 cho MỌI trường hợp vẫn đạt Accuracy = 98,55% - một con số "
        "trông rất ấn tượng nhưng HOÀN TOÀN VÔ DỤNG, vì mô hình đó sẽ bỏ sót 100% các đợt ngập thật. "
        "ĐÂY LÀ LÝ DO CĂN BẢN NHẤT khiến đồ án không dùng Accuracy làm tiêu chí quyết định.",
        "Với tỷ lệ 1.328:1, các độ đo dạng 'Weighted' (trung bình có trọng số theo số mẫu) vẫn sẽ bị chi "
        "phối gần như hoàn toàn bởi lớp 0 - do đó đồ án bắt buộc phải dùng dạng 'Macro' (trung bình "
        "KHÔNG trọng số, xem mục 4.2) để buộc mô hình phải học tốt CẢ lớp hiếm.",
        "Mức mất cân bằng cực đoan này cũng chính là lý do kỹ thuật cho việc BẮT BUỘC phải cân bằng lại "
        "dữ liệu huấn luyện bằng CTGAN/SMOTE (xem mục 2.3) trước khi đưa vào mô hình, chứ không thể chỉ "
        "dựa vào việc chọn độ đo đánh giá phù hợp là đủ.",
    ],
)

add_heading(doc, "4.1. Accuracy (Độ chính xác tổng thể)", level=2)
add_para(doc, "Công thức: Accuracy = (Số dự đoán đúng) / (Tổng số dự đoán).")
add_para(
    doc,
    "TẠI SAO KHÔNG DÙNG DUY NHẤT ACCURACY: với dữ liệu mất cân bằng (ví dụ 90% là lớp 'An toàn'), một "
    "mô hình 'ngu' luôn dự đoán 'An toàn' cho MỌI trường hợp vẫn đạt Accuracy = 90% - trông có vẻ tốt "
    "nhưng hoàn toàn VÔ DỤNG vì không bao giờ phát hiện được ngập thật. Đây là lý do đồ án ưu tiên các "
    "độ đo dạng Macro bên dưới thay vì chỉ báo cáo Accuracy.",
)

add_heading(doc, "4.2. Precision, Recall, F1-Score (dạng Macro)", level=2)
add_para(
    doc,
    "Với mỗi lớp, Precision = TP / (TP + FP) (trong số các lần dự đoán là lớp này, bao nhiêu % đúng "
    "thật), Recall = TP / (TP + FN) (trong số các trường hợp THẬT SỰ thuộc lớp này, mô hình phát hiện "
    "đúng được bao nhiêu %), F1 = trung bình điều hoà (harmonic mean) của Precision và Recall.",
)
add_para(
    doc,
    "'Macro' nghĩa là TÍNH RIÊNG cho từng lớp (0/1/2) rồi lấy trung bình KHÔNG trọng số theo số lượng "
    "mẫu - khác với 'Weighted' (lấy trung bình có trọng số theo số mẫu, sẽ bị lớp đa số 'An toàn' chi "
    "phối). Macro-average buộc mô hình phải dự đoán tốt CẢ 3 lớp, kể cả lớp 'Ngập nặng' hiếm gặp nhưng "
    "quan trọng nhất - ĐÂY LÀ LÝ DO CỐT LÕI khiến F1-Macro được chọn làm tiêu chí chính (leaderboard) "
    "để xếp hạng và lựa chọn best model, thay vì Accuracy hay F1-Weighted.",
)
add_para(
    doc,
    "TẠI SAO RECALL QUAN TRỌNG HƠN PRECISION TRONG BÀI TOÁN NÀY: bỏ sót một đợt ngập thật (Recall "
    "thấp, False Negative) gây hậu quả nghiêm trọng hơn nhiều so với một lần cảnh báo dư thừa (Precision "
    "thấp, False Positive) - người dân/chính quyền chỉ tốn công chuẩn bị không cần thiết nếu cảnh báo "
    "sai, nhưng có thể thiệt hại về người/tài sản nếu bỏ sót cảnh báo ngập thật. Vì vậy khi 2 mô hình có "
    "F1 gần bằng nhau, nên ưu tiên mô hình có Recall (đặc biệt Recall của lớp 'Ngập nặng') cao hơn.",
)

add_heading(doc, "4.3. ROC-AUC (One-vs-Rest)", level=2)
add_para(
    doc,
    "ROC-AUC đo khả năng PHÂN BIỆT của mô hình giữa 1 lớp và phần còn lại (diện tích dưới đường cong "
    "True Positive Rate - False Positive Rate). Với bài toán đa lớp, đồ án dùng chiến lược One-vs-Rest "
    "(OvR): vẽ 1 đường ROC riêng cho từng lớp (lớp đó là 'Positive', 2 lớp còn lại gộp thành "
    "'Negative'), giúp thấy rõ mô hình phân biệt TỐT/KÉM lớp nào nhất. AUC = 0.5 tương đương đoán ngẫu "
    "nhiên (đường chéo baseline), AUC = 1.0 là phân biệt hoàn hảo. Nếu lớp 'Ngập nặng' có AUC thấp, đó "
    "là tín hiệu cần bổ sung thêm dữ liệu ngập nặng (CTGAN/SMOTE) hoặc tinh chỉnh ngưỡng cảnh báo.",
)

add_heading(doc, "4.4. Confusion Matrix (Ma trận nhầm lẫn)", level=2)
add_para(
    doc,
    "Bảng đối chiếu nhãn THẬT (hàng) với nhãn DỰ ĐOÁN (cột) cho từng lớp - cho biết CHÍNH XÁC mô hình "
    "đang nhầm lớp nào với lớp nào. Trong bài toán này, ô quan trọng nhất cần theo dõi là trường hợp lớp "
    "'Ngập nặng' (thật) bị dự đoán nhầm thành 'An toàn' hoặc 'Ngập nhẹ' - đây là loại sai số NGUY HIỂM "
    "NHẤT vì trực tiếp dẫn đến bỏ sót cảnh báo.",
)

add_heading(doc, "4.5. Feature Importance (Độ quan trọng đặc trưng)", level=2)
add_para(
    doc,
    "Cho biết biến đầu vào nào (nhiệt độ, độ ẩm, lượng mưa, độ ẩm đất, chiều cao triều) đóng góp nhiều "
    "nhất vào quyết định của mô hình. Kết quả này giúp: (1) kiểm chứng mô hình học đúng theo logic vật "
    "lý thực tế (lượng mưa/triều cường thường có importance cao nhất - khớp với hiểu biết chuyên ngành "
    "thuỷ văn), (2) định hướng ưu tiên thu thập/giám sát chính xác các biến quan trọng nhất trong vận "
    "hành thực tế. Chỉ áp dụng được cho model dạng bảng (sklearn/XGBoost...) - không có ý nghĩa trực "
    "tiếp cho model dạng chuỗi vì mỗi đặc trưng lặp lại qua nhiều bước thời gian trong cửa sổ.",
)

add_heading(doc, "4.6. Tóm tắt: vì sao đồ án chọn bộ độ đo này", level=2)
add_para(
    doc,
    "Trong code (hàm evaluate_prediction_arrays trong analyze_and_train.py), MỌI mô hình đều được tính "
    "đồng thời 5 độ đo: accuracy, precision_macro, recall_macro, f1_macro, roc_auc_ovr_macro (tham số "
    "average='macro' được truyền tường minh cho precision_score/recall_score/f1_score của scikit-learn) "
    "- không dùng riêng lẻ 1 độ đo duy nhất, mà dùng CẢ BỘ để nhìn vấn đề từ nhiều góc độ, đồng thời chọn "
    "MỘT độ đo làm tiêu chí QUYẾT ĐỊNH cuối cùng cho việc xếp hạng/chọn best model.",
)
add_table(
    doc,
    headers=["Mục đích sử dụng", "Độ đo dùng", "Vì sao phù hợp"],
    rows=[
        [
            "Tiêu chí QUYẾT ĐỊNH xếp hạng & chọn best model (hàm select_best_model_overall)",
            "F1-Score (Macro)",
            "Cân bằng cả Precision lẫn Recall, tính riêng cho từng lớp rồi lấy trung bình không trọng số - không bị lớp 'An toàn' (98,55% dữ liệu) chi phối như Accuracy hay F1-Weighted.",
        ],
        [
            "Đánh giá mức độ 'bỏ sót cảnh báo thật'",
            "Recall (Macro), đặc biệt Recall của lớp 'Ngập nặng'",
            "Recall thấp = bỏ sót ngập thật (False Negative) - hậu quả nghiêm trọng hơn nhiều so với cảnh báo dư (False Positive) trong bài toán cảnh báo thiên tai.",
        ],
        [
            "Đánh giá mức độ 'báo động giả'",
            "Precision (Macro)",
            "Precision thấp = quá nhiều cảnh báo sai, gây tốn nguồn lực ứng phó không cần thiết và giảm lòng tin vào hệ thống nếu lặp lại nhiều lần.",
        ],
        [
            "Đánh giá khả năng phân biệt tổng thể, không phụ thuộc ngưỡng quyết định (threshold)",
            "ROC-AUC (One-vs-Rest, Macro)",
            "Đo chất lượng XÁC SUẤT dự đoán của mô hình trên toàn dải ngưỡng, hữu ích khi cần tinh chỉnh ngưỡng cảnh báo sau này mà không cần huấn luyện lại.",
        ],
        [
            "Chỉ số tham khảo nhanh, KHÔNG dùng để quyết định",
            "Accuracy",
            "Vẫn báo cáo để đối chiếu trực quan, nhưng KHÔNG dùng làm căn cứ chọn mô hình vì dễ gây hiểu lầm với dữ liệu mất cân bằng cực đoan (xem số liệu thật ở mục 4.0).",
        ],
    ],
    col_widths=[5, 3.5, 7.5],
)

doc.add_page_break()

# =====================================================================================
# 5. GIẢI THÍCH CÁC BIỂU ĐỒ / SƠ ĐỒ
# =====================================================================================
add_heading(doc, "5. Giải thích các biểu đồ/sơ đồ trong hệ thống", level=1)
add_para(
    doc,
    "Phần này giải thích MỤC ĐÍCH và CÁCH ĐỌC của từng biểu đồ xuất hiện trong ứng dụng, dùng cho phần "
    "thuyết minh luận văn.",
)

chart_table_rows = [
    [
        "Ma trận tương quan (Correlation Heatmap)",
        "Tab EDA",
        "Thể hiện hệ số tương quan Pearson giữa các biến số. Giá trị càng gần ±1 (màu càng đậm) thì 2 "
        "biến càng liên hệ tuyến tính chặt. Dùng để: kiểm tra biến nào liên hệ mạnh nhất với nhãn Nguy "
        "cơ ngập (thường là Lượng mưa/Triều cường), và phát hiện đa cộng tuyến (multicollinearity) giữa "
        "các biến đầu vào.",
    ],
    [
        "Phân bố lớp mục tiêu (Class Distribution)",
        "Tab EDA",
        "Biểu đồ cột đếm số lượng quan sát theo từng lớp (0/1/2). Trực quan hoá mức độ MẤT CÂN BẰNG của "
        "dữ liệu - là căn cứ trực tiếp để quyết định cần áp dụng CTGAN/SMOTE ở bước huấn luyện.",
    ],
    [
        "Xu hướng mưa & tỷ lệ ngập theo tháng (Monthly Trend)",
        "Tab EDA",
        "Biểu đồ đường (line chart) 2 trục: lượng mưa trung bình và tỷ lệ ngập theo từng tháng trong năm "
        "(gộp nhiều năm). Cho biết THÁNG NÀO trong năm là cao điểm ngập lụt - căn cứ để đề xuất khuyến "
        "nghị quản trị về thời điểm cần tăng cường giám sát/nguồn lực.",
    ],
    [
        "Tỷ lệ ngập theo địa phương (Flood Share by Location)",
        "Tab EDA",
        "Biểu đồ tròn (pie chart) thể hiện tỷ trọng số lần ghi nhận ngập của từng địa phương trong tổng "
        "số 5 địa phương giám sát. Cho biết ĐỊA PHƯƠNG NÀO cần ưu tiên đầu tư trạm quan trắc/lực lượng "
        "ứng trực.",
    ],
    [
        "Phân bố mưa/triều theo lớp (Rain/Tide Distribution by Class)",
        "Tab EDA",
        "Biểu đồ histogram + KDE (density) so sánh phân phối giá trị lượng mưa (hoặc triều cường) giữa "
        "3 lớp nguy cơ ngập. Cho thấy ngưỡng giá trị nào của biến đó thường gắn với ngập nặng - kiểm "
        "chứng trực quan cho luật rule-based dùng để gán nhãn.",
    ],
    [
        "So sánh F1-Score toàn bộ mô hình",
        "Tab Đánh giá",
        "Biểu đồ cột ngang xếp hạng TẤT CẢ mô hình đã huấn luyện theo F1-Macro giảm dần. Dùng để chọn "
        "trực quan mô hình tốt nhất và thấy khoảng cách hiệu năng giữa các nhóm phương pháp (Thống kê/"
        "ML/Deep Learning/Hybrid).",
    ],
    [
        "Top 5 mô hình: Accuracy/Precision/Recall",
        "Tab Đánh giá",
        "Biểu đồ cột nhóm (grouped bar) so sánh 3 độ đo cùng lúc cho 5 mô hình dẫn đầu - giúp thấy rõ "
        "TRADE-OFF giữa Precision và Recall của từng mô hình, hỗ trợ quyết định cuối cùng khi F1 các mô "
        "hình gần bằng nhau.",
    ],
    [
        "Đường cong ROC-AUC (One-vs-Rest)",
        "Tab Đánh giá",
        "3 đường cong ROC (1 đường/lớp) trên cùng 1 biểu đồ, kèm đường baseline chéo (AUC=0.5). Đường "
        "càng lồi về góc trên-trái thì mô hình phân biệt lớp đó càng tốt.",
    ],
    [
        "Confusion Matrix (dạng ảnh nhiệt - heatmap)",
        "Tab Đánh giá",
        "Ma trận 3x3 tô màu theo số lượng, đường chéo chính càng đậm thì mô hình dự đoán đúng càng nhiều.",
    ],
]
add_table(
    doc,
    headers=["Tên biểu đồ", "Xuất hiện ở", "Ý nghĩa / Cách đọc"],
    rows=chart_table_rows,
    col_widths=[4.5, 2.5, 9.5],
)

doc.add_page_break()

# =====================================================================================
# 6. TRẢ LỜI CÁC CÂU HỎI THƯỜNG GẶP VỀ ĐỀ TÀI
# =====================================================================================
add_heading(doc, "6. Trả lời các câu hỏi thường gặp về đề tài", level=1)
add_para(
    doc,
    "Mục này đối chiếu trực tiếp với hiện trạng THẬT của code tại thời điểm biên soạn - không suy diễn "
    "thêm ngoài những gì đã cài đặt và chạy được.",
)

add_heading(doc, "6.1. Mục tiêu đề tài - có bổ sung gì so với ban đầu", level=2)
add_para(doc, "Mục tiêu ban đầu: dự báo nguy cơ ngập lụt (bài toán phân loại 3 lớp).", bold=True)
add_para(doc, "Các mục tiêu đã bổ sung và đã cài đặt hoàn chỉnh:", bold=True)
add_bullets(
    doc,
    [
        (
            "Hướng dẫn đường tránh ngập:",
            [
                "Tab Bản đồ khoanh vùng ranh giới hành chính THẬT của 5 địa phương giám sát (không chỉ "
                "chấm điểm marker như bản đầu) - tô đỏ/xanh theo nguy cơ ngập dự báo.",
                "Khi có điểm ngập trên tuyến, hệ thống tự tìm đường đi thay thế né vùng ngập qua TomTom "
                "Routing API (tham số avoidAreas).",
            ],
        ),
        "Dự báo theo từng địa phương cụ thể: mỗi trong 5 địa phương (TP Huế, Hương Thủy, Hương Trà, Phú "
        "Vang, Quảng Điền) có dự báo 4 ngày riêng - click vào vùng trên bản đồ hiện popup tóm tắt.",
        "Cảnh báo sớm qua dashboard: dự báo 4 ngày tới tự động cập nhật (cache theo ngày), hiển thị bảng "
        "+ biểu đồ ngay khi mở tab, không cần thao tác thêm.",
        (
            "Đối chiếu đa nguồn dữ liệu (cross-validation):",
            [
                "Ma trận so sánh lượng mưa giữa Open-Meteo (nguồn model dùng để huấn luyện) với các API "
                "độc lập khác (Weatherbit, Visual Crossing, Tomorrow.io, Stormglass).",
                "CHỈ dùng để hiển thị đối chiếu/cảnh báo lệch dữ liệu - KHÔNG đưa thẳng giá trị nguồn "
                "khác vào model suy luận, vì model chỉ được huấn luyện trên phân phối/cách đo của "
                "Open-Meteo, đưa dữ liệu nguồn khác vào trực tiếp có thể lệch chuẩn hoá (calibration) và "
                "làm sai lệch kết quả dự đoán.",
            ],
        ),
    ],
)

add_heading(doc, "6.2. Dữ liệu đầu vào/đầu ra - bổ sung gì, xử lý ra sao", level=2)
add_para(doc, "Đầu vào (theo giờ, gộp từ 5 địa phương, qua Open-Meteo Forecast + Marine API):", bold=True)
add_bullets(
    doc,
    ["Lượng_mưa_mm, Độ_ẩm_đất, Chiều_cao_triều_m, nhiệt độ không khí, độ ẩm không khí."],
)
add_para(doc, "Các bước xử lý (thứ tự thực tế trong analyze_and_train.py):", bold=True)
add_bullets(
    doc,
    [
        "Điền dữ liệu khuyết: cột số liệu thiếu -> điền median (chỉ tính trên phần sẽ thuộc tập train, "
        "tránh rò rỉ dữ liệu); cột thời gian/nhãn thiếu -> xoá dòng. (Chi tiết đầy đủ ở Mục 2.3.0.)",
        "Time-lag shift: dùng đặc trưng ngày T để dự báo nhãn ngày T+1 (đúng bản chất bài toán DỰ BÁO, "
        "không phải mô tả hiện trạng).",
        "Gán nhãn rule-based: áp luật ngưỡng tổ hợp mưa + độ ẩm đất + triều cường để tạo nhãn 3 lớp.",
        "Cân bằng lớp thiểu số bằng CTGAN (fallback SMOTE khi lỗi) - dữ liệu gốc mất cân bằng nặng: "
        "431.776 dòng 'An toàn' / 325 dòng 'Ngập nhẹ' / 6.019 dòng 'Ngập nặng' (tỉ lệ mất cân bằng "
        "~1.328:1 giữa lớp đa số và lớp thiểu số nhất).",
    ],
)
add_para(doc, "Đầu ra:", bold=True)
add_bullets(
    doc,
    [
        "Nhãn 3 lớp Nguy_cơ_ngập: 0 = An toàn, 1 = Ngập nhẹ, 2 = Ngập nặng.",
        "Ví dụ cụ thể: 1 dòng dữ liệu ngày X tại Hương Thủy có Lượng_mưa_mm=35, Độ_ẩm_đất=0.42, "
        "Chiều_cao_triều_m=1.1 -> không thoả điều kiện 'Ngập nặng' nhưng thoả rain>25 -> gán Ngập nhẹ (1).",
    ],
)

add_heading(doc, "6.3. Phát biểu bài toán - 2 bài toán (đối chiếu với tài liệu tham khảo)", level=2)
add_para(
    doc,
    "Tài liệu tham khảo cấu trúc 'Phát biểu bài toán' thành 2 bài toán con: (1) bài toán DỰ BÁO chỉ số "
    "UV, và (2) bài toán CẢNH BÁO sức khỏe và GỢI Ý du lịch (dùng kết quả dự báo ở bài toán 1 làm đầu "
    "vào để đưa ra khuyến nghị hành động cho người dùng cuối). Đối chiếu với hệ thống hiện tại, đồ án "
    "này CÓ ĐỦ cấu trúc 2 bài toán tương đương - chỉ khác đối tượng (ngập lụt thay vì chỉ số UV):",
)
add_bullets(
    doc,
    [
        (
            "Bài toán 1 - Dự báo nguy cơ ngập lụt (tương ứng 'dự báo chỉ số UV'):",
            [
                "Bài toán phân loại đa lớp trên dữ liệu khí tượng - thủy văn: đầu vào là lượng mưa, độ "
                "ẩm đất, chiều cao triều, nhiệt độ, độ ẩm không khí; đầu ra là nhãn nguy cơ ngập 3 mức "
                "(An toàn / Ngập nhẹ / Ngập nặng). F1-Macro là tiêu chí duy nhất dùng để chọn model tốt "
                "nhất (select_best_model_overall()).",
                "Có thêm 1 bước trung gian mang bản chất HỒI QUY: các model Threshold (Linear/Polynomial "
                "Regression) và chuỗi thời gian (ARIMA/SARIMA) dự đoán 1 GIÁ TRỊ LIÊN TỤC trước (ví dụ "
                "lượng mưa dự kiến), rồi mới làm tròn/áp ngưỡng ra lớp 0/1/2 "
                "(round_and_clip_predictions()) - cột 'Dự báo Lượng mưa (mm)' ở Tab 1 chính là kết quả "
                "bước này.",
            ],
        ),
        (
            "Bài toán 2 - Cảnh báo nguy cơ ngập và gợi ý lộ trình tránh ngập (tương ứng 'cảnh báo sức "
            "khỏe và gợi ý du lịch'):",
            [
                "Cảnh báo: khoanh vùng ranh giới hành chính của 5 địa phương giám sát trên bản đồ, tô "
                "màu đỏ/xanh theo nguy cơ ngập dự báo (lấy trực tiếp kết quả từ Bài toán 1 làm đầu vào); "
                "popup tóm tắt dự báo 4 ngày khi click vào từng vùng.",
                "Gợi ý hành động: khi phát hiện điểm ngập nằm trên lộ trình người dùng chọn, hệ thống tự "
                "động tính lại tuyến đường thay thế NÉ vùng ngập qua TomTom Routing API (tham số "
                "avoidAreas) - tương đương vai trò 'gợi ý du lịch' bên tài liệu tham khảo, nhưng gợi ý ở "
                "đây là LỘ TRÌNH AN TOÀN thay vì điểm đến du lịch.",
                "Ngoài ra còn có lớp cảnh báo phụ: ma trận đối chiếu dữ liệu mưa giữa nhiều nguồn API "
                "(Mục 6.1) tự động cảnh báo khi các nguồn lệch nhau đáng kể, hỗ trợ người dùng thận trọng "
                "hơn khi đọc kết quả dự báo.",
            ],
        ),
    ],
)

add_heading(doc, "6.4. Đối tượng nghiên cứu", level=2)
add_para(
    doc,
    "Đề tài tập trung nghiên cứu mức độ nguy cơ ngập lụt như một biến mục tiêu phân loại, đồng thời xem "
    "xét mối quan hệ giữa nguy cơ ngập với các yếu tố khí tượng - thủy văn như lượng mưa, chiều cao "
    "triều, độ ẩm đất, nhiệt độ và độ ẩm không khí. Trên phương diện phương pháp, đối tượng nghiên cứu "
    "còn bao gồm các thuật toán phân loại học máy và các kỹ thuật xử lý dữ liệu mất cân bằng (CTGAN là "
    "kỹ thuật chính, SMOTE dùng làm phương án dự phòng khi CTGAN lỗi/không cài được). Bên cạnh đó, đề "
    "tài cũng quan tâm đến khả năng tích hợp mô hình dự báo vào một hệ thống dashboard trực quan hoá và "
    "cảnh báo sớm phục vụ người dùng cuối, cụ thể gồm bản đồ tương tác chỉ đường tránh ngập theo thời "
    "gian thực và cơ chế đối chiếu dữ liệu từ nhiều nguồn API thời tiết độc lập.",
)

doc.add_page_break()

# =====================================================================================
# 7. CHI TIẾT TỪNG THUẬT TOÁN/KỸ THUẬT SỬ DỤNG TRONG ĐỒ ÁN
# =====================================================================================
add_heading(doc, "7. Chi tiết từng thuật toán/kỹ thuật sử dụng trong đồ án", level=1)
add_para(
    doc,
    "Mục này giải thích TỪNG thuật toán đang chạy thật trong build_model_registry() (analyze_and_train.py) "
    "- nguyên lý hoạt động, vì sao dùng cho bài toán này, và ưu/hạn chế thực tế quan sát được qua kết "
    "quả huấn luyện. Đọc mục này để nắm chắc BẢN CHẤT từng model, không chỉ tên gọi.",
)


def add_algorithm_block(doc, name, principle, why_here, pros, cons):
    add_heading(doc, name, level=3)
    add_para(doc, "Nguyên lý hoạt động: ", bold=True)
    doc.paragraphs[-1].add_run(principle)
    add_para(doc, "Vì sao dùng trong đồ án này: ", bold=True)
    doc.paragraphs[-1].add_run(why_here)
    add_para(doc, "Ưu điểm:", bold=True)
    add_bullets(doc, pros)
    add_para(doc, "Hạn chế:", bold=True)
    add_bullets(doc, cons)
    doc.add_paragraph()


# ---- 7.1 Nhóm Thống kê ----
add_heading(doc, "7.1. Nhóm Thống kê (Statistical)", level=2)
add_algorithm_block(
    doc,
    "Linear Regression Threshold",
    "Hồi quy tuyến tính chuẩn (y = w1*x1 + w2*x2 + ... + b) - huấn luyện để dự đoán nhãn 0/1/2 NHƯ MỘT "
    "SỐ THỰC LIÊN TỤC (không phải trực tiếp ra lớp), sau đó làm tròn + giới hạn về khoảng [0, 2] "
    "(round_and_clip_predictions()) để suy ra lớp cuối cùng.",
    "Dùng làm mô hình BASELINE đơn giản nhất - để có 1 mốc so sánh 'sàn' cho các model phức tạp hơn. "
    "Nếu 1 model ML/DL không vượt qua được baseline này thì độ phức tạp thêm vào không đáng.",
    ["Huấn luyện cực nhanh, không tốn tài nguyên.", "Dễ diễn giải (trọng số mỗi biến thể hiện mức ảnh hưởng)."],
    [
        "Giả định quan hệ TUYẾN TÍNH giữa đặc trưng và nhãn - không đúng với thực tế bài toán ngập lụt "
        "(quan hệ có tính ngưỡng/phi tuyến rõ rệt, ví dụ mưa vượt 50mm mới nhảy hẳn sang Ngập nặng).",
        "Việc ép 1 giá trị hồi quy liên tục về 3 lớp rời rạc bằng làm tròn là một xấp xỉ thô, không tối "
        "ưu cho bài toán phân loại thật sự.",
    ],
)
add_algorithm_block(
    doc,
    "Polynomial Regression Threshold",
    "Giống Linear Regression Threshold nhưng thêm bước PolynomialFeatures(degree=2) trước khi hồi quy - "
    "tự sinh thêm các đặc trưng bậc 2 (x1^2, x1*x2, ...) để mô hình học được quan hệ phi tuyến bậc 2.",
    "Kiểm tra xem việc thêm 1 chút phi tuyến (bậc 2) có cải thiện đáng kể so với Linear Regression "
    "Threshold hay không, mà vẫn giữ được tốc độ huấn luyện nhanh của họ hồi quy.",
    ["Nắm bắt được một phần quan hệ phi tuyến mà Linear Regression bỏ lỡ.", "Vẫn huấn luyện nhanh."],
    [
        "Bậc 2 vẫn còn quá đơn giản so với luật ngưỡng thực tế (rule-based) dùng để gán nhãn - vốn có "
        "nhiều điều kiện tổ hợp AND/OR giữa 3 biến khác nhau.",
        "Số lượng đặc trưng tăng theo cấp số nhân khi thêm biến/bậc - dễ overfitting nếu bậc cao hơn.",
    ],
)

# ---- 7.2 Nhóm Machine Learning ----
add_heading(doc, "7.2. Nhóm Machine Learning", level=2)
add_algorithm_block(
    doc,
    "Random Forest",
    "Tập hợp (ensemble) nhiều cây quyết định (ở đây 300 cây - n_estimators=300), mỗi cây huấn luyện "
    "trên 1 tập con dữ liệu/đặc trưng ngẫu nhiên (bagging), kết quả cuối là biểu quyết đa số (voting) "
    "giữa các cây.",
    "Model tabular cổ điển mạnh, ổn định, ít nhạy cảm với việc chuẩn hoá dữ liệu hay outlier - phù hợp "
    "làm mốc so sánh vững chắc cho nhóm Machine Learning.",
    [
        "Chống overfitting tốt hơn 1 cây quyết định đơn lẻ nhờ cơ chế bagging.",
        "Tự tính được Feature Importance - hữu ích để diễn giải biến nào ảnh hưởng nhiều nhất tới nguy "
        "cơ ngập.",
    ],
    ["Chậm hơn 1 cây đơn khi dự đoán (phải hỏi ý kiến cả 300 cây).", "Kém hiệu quả hơn Boosting trên dữ liệu mất cân bằng nếu không kết hợp kỹ thuật cân bằng lớp trước."],
)
add_algorithm_block(
    doc,
    "KNN (K-Nearest Neighbors)",
    "Không 'học' tham số nào cả (lazy learning) - khi dự đoán 1 điểm mới, tìm K điểm GẦN NHẤT trong tập "
    "train (theo khoảng cách Euclidean trên không gian đặc trưng đã chuẩn hoá) rồi biểu quyết đa số nhãn "
    "của K điểm đó.",
    "Dùng làm đại diện cho nhóm thuật toán dựa trên khoảng cách/tương đồng - kiểm chứng xem dữ liệu có "
    "cấu trúc cụm rõ ràng theo lớp hay không.",
    ["Đơn giản, không giả định gì về phân phối dữ liệu.", "Hiệu quả khi ranh giới giữa các lớp có tính cục bộ (local)."],
    [
        "Chậm khi dự đoán trên tập dữ liệu lớn (phải tính khoảng cách tới TOÀN BỘ điểm train).",
        "Nhạy cảm với thang đo (scale) của đặc trưng - bắt buộc phải chuẩn hoá dữ liệu trước (đã dùng "
        "StandardScaler trong pipeline).",
        "Hiệu năng thường kém hơn các model boosting trên bài toán nhiều chiều/mất cân bằng.",
    ],
)
add_algorithm_block(
    doc,
    "SVC (Support Vector Classifier)",
    "Tìm 1 SIÊU PHẲNG (hyperplane) phân tách các lớp sao cho khoảng cách (margin) từ siêu phẳng đó tới "
    "các điểm dữ liệu gần nhất của mỗi lớp là LỚN NHẤT có thể; dùng kernel (mặc định RBF) để xử lý dữ "
    "liệu không phân tách tuyến tính được.",
    "Đại diện cho nhóm thuật toán tối ưu biên (margin-based) - thường mạnh khi số chiều đặc trưng vừa "
    "phải và ranh giới giữa các lớp rõ ràng.",
    ["Hiệu quả với dữ liệu có ranh giới lớp rõ ràng.", "Kernel RBF cho phép học ranh giới phi tuyến phức tạp."],
    [
        "Huấn luyện chậm khi số lượng mẫu lớn (độ phức tạp gần bậc 2-3 theo số mẫu) - đặc biệt rõ sau "
        "khi CTGAN sinh thêm hàng chục nghìn dòng.",
        "Khó diễn giải hơn Random Forest/cây quyết định (không có Feature Importance trực tiếp).",
    ],
)
add_algorithm_block(
    doc,
    "AdaBoost (Adaptive Boosting)",
    "Huấn luyện TUẦN TỰ nhiều model yếu (mặc định là cây quyết định nông), mỗi model sau tập trung sửa "
    "lỗi của model trước bằng cách TĂNG TRỌNG SỐ cho các điểm dữ liệu bị dự đoán sai ở vòng trước.",
    "Đại diện cho họ Boosting 'cổ điển' (trước khi Gradient Boosting hiện đại như XGBoost/LightGBM/"
    "CatBoost ra đời) - dùng để so sánh xem Boosting hiện đại cải thiện được bao nhiêu so với bản gốc.",
    ["Thường tốt hơn 1 model yếu đơn lẻ.", "Ít tham số cần tinh chỉnh hơn Gradient Boosting hiện đại."],
    [
        "Nhạy cảm với nhiễu/outlier hơn Random Forest (điểm nhiễu dễ bị tăng trọng số quá mức qua nhiều "
        "vòng).",
        "Thường thua các thuật toán Gradient Boosting hiện đại (XGBoost/LightGBM/CatBoost) về độ chính "
        "xác trên dữ liệu tabular phức tạp - đúng như quan sát thực tế trong kết quả huấn luyện.",
    ],
)
add_algorithm_block(
    doc,
    "XGBoost (Extreme Gradient Boosting)",
    "Boosting hiện đại: xây TUẦN TỰ nhiều cây quyết định, mỗi cây học để dự đoán PHẦN DƯ (residual/"
    "gradient của hàm mất mát) mà các cây trước đó còn dự đoán sai, có thêm regularization (L1/L2) để "
    "chống overfitting. Ở đây cấu hình objective='multi:softprob' cho bài toán 3 lớp, 250 cây "
    "(n_estimators=250), độ sâu tối đa 6 (max_depth=6), learning_rate=0.05.",
    "Một trong những thuật toán mạnh nhất hiện nay cho dữ liệu dạng bảng (tabular) - gần như là lựa chọn "
    "mặc định trong các cuộc thi/dự án ML thực tế với dữ liệu có cấu trúc như bài toán này.",
    [
        "Độ chính xác cao trên dữ liệu tabular, xử lý tốt tương tác phi tuyến giữa các biến.",
        "Có regularization tích hợp sẵn, ít overfitting hơn AdaBoost/cây đơn lẻ.",
        "Hỗ trợ tính Feature Importance và ROC-AUC (predict_proba) để diễn giải.",
    ],
    [
        "Nhiều siêu tham số cần tinh chỉnh (đã dùng Optuna để tự động hoá việc này - xem Mục 7.7).",
        "Dễ overfitting nếu số cây/độ sâu quá lớn so với lượng dữ liệu thật (trước khi CTGAN cân bằng).",
    ],
)
add_algorithm_block(
    doc,
    "LightGBM",
    "Cũng là Gradient Boosting như XGBoost, nhưng dùng chiến lược mọc cây THEO LÁ (leaf-wise) thay vì "
    "theo tầng (level-wise) như XGBoost mặc định - hội tụ nhanh hơn với cùng số cây, kèm kỹ thuật "
    "histogram-based binning để tăng tốc độ huấn luyện đáng kể trên dữ liệu lớn.",
    "So sánh trực tiếp với XGBoost trên CÙNG bài toán - kiểm chứng xem chiến lược leaf-wise có mang lại "
    "lợi thế tốc độ/độ chính xác nào cho đặc thù dữ liệu khí tượng - thủy văn ở đây không.",
    ["Huấn luyện nhanh hơn XGBoost đáng kể trên tập dữ liệu lớn (sau khi CTGAN sinh thêm ~18.000 dòng).", "Tốn ít bộ nhớ hơn nhờ histogram-based binning."],
    [
        "Chiến lược leaf-wise dễ overfitting hơn level-wise nếu dữ liệu ít hoặc num_leaves quá lớn so "
        "với số mẫu.",
        "Nhạy cảm hơn XGBoost với dữ liệu có nhiều nhiễu/nhãn không chắc chắn (đặc biệt đáng lưu ý vì "
        "nhãn ở đây là rule-based, không phải nhãn thực đo).",
    ],
)
add_algorithm_block(
    doc,
    "CatBoost",
    "Gradient Boosting với 2 cải tiến chính: Ordered Boosting (tránh rò rỉ dữ liệu mục tiêu khi tính "
    "gradient - target leakage nội bộ giữa các cây) và xử lý đặc trưng dạng phân loại (categorical) "
    "không cần one-hot encoding thủ công.",
    "So sánh thêm 1 biến thể Gradient Boosting nữa với cơ chế chống overfitting khác XGBoost/LightGBM - "
    "đặc biệt hữu ích nếu sau này đồ án mở rộng thêm đặc trưng dạng phân loại (ví dụ tên địa phương).",
    ["Ordered Boosting giảm nguy cơ overfitting/target leakage tốt hơn 2 thuật toán boosting còn lại.", "Ít cần tinh chỉnh siêu tham số hơn (default đã khá tốt)."],
    ["Huấn luyện thường chậm hơn LightGBM.", "Ưu thế về categorical feature chưa được tận dụng triệt để vì bài toán hiện tại chủ yếu là đặc trưng số."],
)

# ---- 7.3 Nhóm Time Series ----
add_heading(doc, "7.3. Nhóm Chuỗi thời gian (Time Series)", level=2)
add_algorithm_block(
    doc,
    "ARIMA (AutoRegressive Integrated Moving Average)",
    "Mô hình thống kê chuỗi thời gian kinh điển: kết hợp thành phần Tự hồi quy (AR - giá trị hiện tại "
    "phụ thuộc tuyến tính vào các giá trị quá khứ), Sai phân (I - Integrated, để loại xu hướng/làm chuỗi "
    "dừng - stationary), và Trung bình trượt (MA - phụ thuộc vào sai số dự báo quá khứ).",
    "Dùng làm baseline THUẦN CHUỖI THỜI GIAN (không xét đến các biến khí tượng khác cùng lúc, chỉ nhìn "
    "vào lịch sử biến mục tiêu) - đối chứng xem việc thêm các biến khí tượng - thủy văn khác (như các "
    "model tabular ở Mục 7.1-7.2) có thực sự cải thiện dự báo hơn hay không.",
    ["Là chuẩn mực (standard baseline) lâu đời cho bài toán chuỗi thời gian, dễ so sánh với tài liệu tham khảo khác.", "Không cần đặc trưng ngoài (exogenous), chỉ cần lịch sử của chính biến mục tiêu."],
    [
        "Giả định tuyến tính và dừng (stationary) - dữ liệu ngập lụt thực tế có tính mùa vụ và đột biến "
        "mạnh (mưa cực đoan) khó thoả giả định này (log cảnh báo 'Non-stationary starting autoregressive "
        "parameters' khi huấn luyện là dấu hiệu trực tiếp của việc này).",
        "Không tận dụng được các biến khí tượng khác (mưa, triều, độ ẩm đất) cùng lúc như model tabular.",
    ],
)
add_algorithm_block(
    doc,
    "SARIMA (Seasonal ARIMA)",
    "Mở rộng của ARIMA, thêm thành phần MÙA VỤ (seasonal) - mô hình hoá thêm chu kỳ lặp lại theo mùa "
    "(ví dụ mùa mưa Huế thường rơi vào tháng 9-12 hàng năm) bên cạnh xu hướng ngắn hạn mà ARIMA đã có.",
    "Kiểm chứng xem việc thêm yếu tố mùa vụ có cải thiện đáng kể so với ARIMA thường hay không, phù hợp "
    "với đặc thù khí hậu Huế có mùa mưa/mùa khô rõ rệt.",
    ["Nắm bắt được tính mùa vụ mà ARIMA thường bỏ lỡ - phù hợp đặc thù khí hậu nhiệt đới gió mùa của Huế."],
    ["Cùng hạn chế tuyến tính/dừng như ARIMA.", "Thêm tham số mùa vụ khiến việc tinh chỉnh phức tạp hơn."],
)

# ---- 7.4 Nhóm Deep Learning ----
add_heading(doc, "7.4. Nhóm Deep Learning", level=2)
add_algorithm_block(
    doc,
    "LSTM (Long Short-Term Memory)",
    "Mạng nơ-ron hồi quy (RNN) có cơ chế 'cổng' (gates: forget/input/output) để quyết định giữ lại hay "
    "quên thông tin qua nhiều bước thời gian - giải quyết được vấn đề 'quên' (vanishing gradient) mà RNN "
    "thường gặp khi chuỗi dài. Ở đây dùng cửa sổ 7 ngày liên tiếp (SEQUENCE_WINDOW=7) làm 1 mẫu đầu vào "
    "để dự đoán nhãn ngày kế tiếp.",
    "Bài toán vốn có bản chất CHUỖI THỜI GIAN đa biến (mưa/triều/độ ẩm đất biến thiên liên tục qua các "
    "ngày, ảnh hưởng tích luỹ tới nguy cơ ngập) - LSTM phù hợp để học các MẪU HÌNH biến thiên qua nhiều "
    "ngày liên tiếp mà các model tabular (chỉ nhìn 1 ngày duy nhất) không nắm bắt được.",
    [
        "Học được phụ thuộc dài hạn (long-term dependency) giữa các ngày liên tiếp - ví dụ mưa dồn dập "
        "nhiều ngày liền dù mỗi ngày riêng lẻ chưa tới ngưỡng ngập nặng.",
        "Đạt hiệu năng CAO NHẤT trong toàn bộ các model đã thử nghiệm trên dữ liệu thực tế của đồ án "
        "(F1-Macro ~0.92, Accuracy ~0.97) - xác nhận trực tiếp giả thuyết trên.",
    ],
    [
        "Cần nhiều dữ liệu và thời gian huấn luyện hơn model tabular.",
        "'Hộp đen' hơn Random Forest/XGBoost - khó diễn giải TRỰC TIẾP tại sao model dự đoán 1 kết quả "
        "cụ thể (không có Feature Importance kiểu cây quyết định).",
        "Không chạy được nếu server thiếu GPU/CUDA đúng cấu hình - dù vẫn chạy được trên CPU (chậm hơn), "
        "như log huấn luyện thực tế đã ghi nhận cảnh báo CUDA_ERROR_NO_DEVICE.",
    ],
)
add_algorithm_block(
    doc,
    "GRU (Gated Recurrent Unit)",
    "Biến thể đơn giản hoá của LSTM - gộp cổng forget/input thành 1 cổng update duy nhất (ít tham số "
    "hơn LSTM), vẫn giữ được khả năng học phụ thuộc dài hạn qua chuỗi thời gian.",
    "So sánh trực tiếp với LSTM trên CÙNG bài toán/cùng cửa sổ 7 ngày - kiểm chứng xem kiến trúc đơn "
    "giản hơn (ít tham số, huấn luyện nhanh hơn) có đánh đổi độ chính xác đáng kể hay không so với LSTM.",
    ["Ít tham số hơn LSTM -> huấn luyện nhanh hơn, ít dữ liệu hơn vẫn có thể học tốt.", "Thường cho kết quả tương đương LSTM trên nhiều bài toán chuỗi thời gian vừa và nhỏ."],
    ["Với chuỗi RẤT dài hoặc quan hệ phức tạp, đôi khi kém hơn LSTM do khả năng biểu diễn hạn chế hơn (ít cổng điều khiển hơn)."],
)
add_algorithm_block(
    doc,
    "1D-CNN (1-Dimensional Convolutional Neural Network)",
    "Áp dụng phép tích chập (convolution) - vốn nổi tiếng với ảnh 2D - theo 1 CHIỀU DUY NHẤT (thời "
    "gian), dùng các bộ lọc (filter) trượt qua cửa sổ 7 ngày để tự động phát hiện các MẪU HÌNH CỤC BỘ "
    "(local pattern) trong chuỗi, ví dụ 1 đợt tăng mưa đột ngột kéo dài 2-3 ngày.",
    "Thử nghiệm hướng tiếp cận KHÁC LSTM/GRU cho cùng dữ liệu chuỗi - CNN 1D thường huấn luyện nhanh hơn "
    "RNN vì tính toán song song được (không phụ thuộc tuần tự từng bước thời gian như LSTM/GRU).",
    ["Huấn luyện nhanh hơn LSTM/GRU nhờ tính song song hoá được (không xử lý tuần tự từng bước thời gian).", "Tốt trong việc phát hiện mẫu hình cục bộ ngắn hạn (short-term pattern)."],
    ["Kém hơn LSTM/GRU trong việc nắm bắt phụ thuộc DÀI HẠN xuyên suốt cả cửa sổ 7 ngày (bản chất convolution chỉ nhìn cục bộ qua kích thước filter)."],
)
add_algorithm_block(
    doc,
    "CNN-LSTM",
    "Kiến trúc lai theo TẦNG (không phải Hybrid ở Mục 7.5): lớp Conv1D trích xuất mẫu hình cục bộ trước, "
    "rồi đưa qua lớp LSTM để học thêm phụ thuộc dài hạn trên đặc trưng đã được Conv1D trích xuất.",
    "Kết hợp ưu điểm của cả 2 kiến trúc trên - vừa bắt được mẫu hình cục bộ ngắn hạn (nhờ Conv1D) vừa "
    "giữ được khả năng nhớ dài hạn xuyên suốt cửa sổ (nhờ LSTM ở tầng sau).",
    ["Có tiềm năng vượt qua từng kiến trúc đơn lẻ (1D-CNN hoặc LSTM riêng) nếu dữ liệu đủ lớn để tận dụng độ phức tạp thêm vào."],
    ["Nhiều tham số hơn cả 1D-CNN và LSTM riêng lẻ -> cần nhiều dữ liệu hơn để tránh overfitting, huấn luyện chậm nhất trong nhóm Deep Learning."],
)

# ---- 7.5 Nhóm Hybrid ----
add_heading(doc, "7.5. Nhóm Hybrid", level=2)
add_algorithm_block(
    doc,
    "LSTM + XGBoost Hybrid",
    "KHÁC với CNN-LSTM (là 1 mạng neural liền mạch), đây là 2 model ĐỘC LẬP ghép nối tiếp: (1) huấn "
    "luyện 1 mạng LSTM phân loại như bình thường trên cửa sổ 7 ngày, (2) sau đó LẤY RA lớp ẩn "
    "'dense_features' của LSTM đó làm EMBEDDING (biểu diễn số học đã học được của chuỗi 7 ngày), (3) "
    "dùng embedding này làm đặc trưng đầu vào để huấn luyện 1 model XGBoost classifier RIÊNG, độc lập "
    "với LSTM ban đầu.",
    "Kết hợp thế mạnh của 2 họ mô hình khác nhau: LSTM giỏi TRÍCH XUẤT đặc trưng từ chuỗi thời gian thô, "
    "còn XGBoost giỏi PHÂN LOẠI trên không gian đặc trưng đã được trích xuất tốt (thường hiệu quả hơn "
    "lớp Dense/Softmax cuối cùng của bản thân mạng neural, đặc biệt khi dữ liệu không quá lớn).",
    [
        "Tận dụng khả năng học biểu diễn chuỗi (representation learning) của LSTM MÀ KHÔNG cần lớp phân "
        "loại cuối của mạng neural (vốn dễ overfitting hơn boosting trên tập dữ liệu vừa/nhỏ).",
        "XGBoost ở tầng sau vẫn giữ được ưu điểm dễ diễn giải hơn (Feature Importance trên các chiều "
        "embedding) so với để nguyên mạng neural end-to-end.",
    ],
    [
        "Kiến trúc phức tạp nhất trong toàn bộ hệ thống - phải huấn luyện VÀ debug 2 model riêng biệt "
        "theo đúng thứ tự (lỗi ở bước 1 sẽ lan sang bước 2).",
        "Thời gian huấn luyện dài nhất (cộng dồn thời gian của cả LSTM lẫn XGBoost).",
    ],
)

# ---- 7.6 Kỹ thuật cân bằng dữ liệu ----
add_heading(doc, "7.6. Kỹ thuật cân bằng dữ liệu (Data Balancing)", level=2)
add_algorithm_block(
    doc,
    "CTGAN (Conditional Tabular GAN)",
    "Mạng đối sinh (GAN - Generative Adversarial Network) chuyên biệt cho dữ liệu DẠNG BẢNG: 1 mạng "
    "Generator học sinh ra dữ liệu giả trông giống dữ liệu thật của lớp thiểu số, 1 mạng Discriminator "
    "học phân biệt dữ liệu giả với dữ liệu thật - 2 mạng 'thi đấu' với nhau tới khi Generator sinh được "
    "dữ liệu đủ giống thật để đánh lừa Discriminator. Ở đây dùng kỹ thuật PacGAN (pac=1 sau khi sửa lỗi "
    "AssertionError chia hết batch - xem thực tế vận hành bên dưới) để ổn định quá trình huấn luyện GAN.",
    "Dữ liệu 'Ngập nhẹ' và 'Ngập nặng' quá hiếm (325 và 6.019 dòng so với 431.776 dòng 'An toàn') - nếu "
    "huấn luyện trực tiếp, model sẽ thiên vị nặng về lớp đa số. CTGAN sinh thêm dữ liệu tổng hợp CÓ ĐIỀU "
    "KIỆN theo từng lớp thiểu số, học được PHÂN PHỐI THỐNG KÊ THẬT của dữ liệu (kể cả tương quan giữa "
    "các biến), thay vì chỉ nội suy đơn giản như SMOTE.",
    [
        "Chất lượng dữ liệu tổng hợp thường tốt hơn SMOTE trên dữ liệu nhiều chiều/có tương quan phức "
        "tạp giữa các biến (ví dụ tương quan giữa mưa và độ ẩm đất).",
        "Sinh được đúng số lượng mẫu cần thiết cho MỖI lớp riêng biệt (target_count có thể tuỳ chỉnh).",
    ],
    [
        "Huấn luyện chậm và tốn tài nguyên hơn SMOTE nhiều lần (phải huấn luyện cả Generator lẫn "
        "Discriminator qua nhiều epoch).",
        "THỰC TẾ VẬN HÀNH đã gặp lỗi AssertionError do ràng buộc chia hết batch/pac của kỹ thuật PacGAN "
        "bên trong thư viện ctgan - đã khắc phục bằng cách đặt pac=1 (xem Mục 2.3 và lịch sử sửa lỗi).",
        "Có thể sinh ra mẫu 'không thực tế' (ví dụ tổ hợp mưa/triều/độ ẩm đất không bao giờ xảy ra cùng "
        "lúc trong thực tế) nếu Generator chưa hội tụ tốt - cần theo dõi qua khối 'Cân bằng dữ liệu' ở "
        "Tab 2 để phát hiện bất thường.",
    ],
)
add_algorithm_block(
    doc,
    "SMOTE (Synthetic Minority Oversampling Technique)",
    "Với mỗi điểm dữ liệu thuộc lớp thiểu số, tìm K điểm LÂN CẬN GẦN NHẤT cùng lớp, rồi tạo điểm dữ liệu "
    "MỚI bằng cách NỘI SUY TUYẾN TÍNH ngẫu nhiên giữa điểm gốc và 1 điểm lân cận được chọn.",
    "Dùng làm PHƯƠNG ÁN DỰ PHÒNG (fallback) khi CTGAN lỗi hoặc lớp thiểu số quá ít mẫu (dưới 10 dòng) để "
    "huấn luyện GAN ổn định - đảm bảo pipeline VẪN CHẠY ĐƯỢC dù CTGAN gặp sự cố, không làm gián đoạn "
    "toàn bộ quá trình huấn luyện.",
    [
        "Đơn giản, nhanh, ổn định - hầu như không bao giờ lỗi (khác CTGAN vốn có nhiều ràng buộc kỹ "
        "thuật hơn).",
        "Không cần huấn luyện thêm 1 mạng neural nào - áp dụng gần như tức thời.",
    ],
    [
        "Chỉ nội suy TUYẾN TÍNH giữa các điểm có sẵn - không học được phân phối thống kê thật, dễ tạo "
        "điểm 'nằm giữa' 2 lớp gây nhiễu ranh giới quyết định (decision boundary) nếu 2 lớp gần nhau "
        "trong không gian đặc trưng.",
        "Không xét tới tương quan phức tạp giữa nhiều biến như CTGAN.",
    ],
)

# ---- 7.7 Kỹ thuật chia dữ liệu & tinh chỉnh siêu tham số ----
add_heading(doc, "7.7. Kỹ thuật chia dữ liệu & tinh chỉnh siêu tham số", level=2)
add_algorithm_block(
    doc,
    "TimeSeriesSplit / Chia Train-Test theo thời gian",
    "Chia 80% dòng ĐẦU (theo thời gian) của MỖI địa phương làm tập train, 20% dòng SAU làm tập test - "
    "KHÔNG xáo trộn ngẫu nhiên như train_test_split(shuffle=True) hay K-Fold thường.",
    "Dữ liệu là CHUỖI THỜI GIAN. Nếu xáo trộn ngẫu nhiên, dữ liệu TƯƠNG LAI có thể lọt vào tập huấn "
    "luyện (data leakage theo thời gian), khiến độ chính xác đánh giá bị 'ảo' - cao hơn nhiều so với khi "
    "model chạy thật trong thực tế (lúc đó chỉ có dữ liệu quá khứ để dự báo tương lai).",
    ["Mô phỏng ĐÚNG bối cảnh vận hành thực tế - đánh giá trung thực khả năng dự báo tương lai của model.", "Áp dụng nhất quán cho cả bước tính median điền khuyết (Mục 2.3.0) lẫn bước chia train/test cuối cùng."],
    ["Tập test có thể không đại diện đầy đủ nếu phân phối dữ liệu thay đổi mạnh theo thời gian (ví dụ năm gần nhất mưa cực đoan hơn hẳn các năm trước)."],
)
add_algorithm_block(
    doc,
    "GridSearchCV",
    "Duyệt VÉT CẠN (exhaustive search) TẤT CẢ tổ hợp giá trị siêu tham số được khai báo trước trong 1 "
    "lưới (grid), đánh giá từng tổ hợp bằng cross-validation, chọn tổ hợp cho kết quả tốt nhất.",
    "Dùng cho Random Forest - không gian tham số nhỏ, rời rạc (3 tham số, mỗi tham số 3-4 giá trị = 48 "
    "tổ hợp) nên duyệt vét cạn khả thi về thời gian, và kết quả TÁI LẬP 100% (deterministic).",
    ["Đảm bảo tìm được tổ hợp TỐT NHẤT trong đúng phạm vi lưới đã khai báo (không có yếu tố ngẫu nhiên).", "Dễ hiểu, dễ kiểm tra lại kết quả."],
    ["Chi phí tính toán tăng theo cấp số NHÂN khi thêm tham số/giá trị - không khả thi với không gian tham số lớn (ví dụ XGBoost/LSTM có hàng chục tham số liên tục)."],
)
add_algorithm_block(
    doc,
    "Optuna (Bayesian Optimization - TPE)",
    "Dùng thuật toán TPE (Tree-structured Parzen Estimator) - 1 dạng Bayesian Optimization - để THÔNG "
    "MINH chọn tổ hợp siêu tham số tiếp theo cần thử dựa trên kết quả của các lần thử TRƯỚC ĐÓ, thay vì "
    "duyệt vét cạn hay chọn ngẫu nhiên hoàn toàn.",
    "Dùng cho XGBoost/LSTM - không gian tham số lớn, liên tục, và có yếu tố KIẾN TRÚC (số lớp/số unit "
    "của mạng neural) mà GridSearchCV không biểu diễn hiệu quả được (sẽ bùng nổ tổ hợp - combinatorial "
    "explosion - nếu cố duyệt vét cạn).",
    [
        "Hội tụ về vùng tham số tốt nhanh hơn Grid/Random Search trên không gian lớn, nhờ học từ lịch "
        "sử các lần thử trước.",
        "Hỗ trợ pruning (cắt tỉa sớm các trial có dấu hiệu kém ngay giữa chừng) để tiết kiệm thời gian "
        "huấn luyện.",
    ],
    ["Có yếu tố ngẫu nhiên (kết quả 2 lần chạy có thể hơi khác nhau, khác GridSearchCV tái lập 100%).", "Cần đủ số lượng trial (lần thử) mới phát huy hết lợi thế so với Random Search thuần."],
)

doc.add_page_break()

# =====================================================================================
# GHI CHÚ CUỐI
# =====================================================================================
add_heading(doc, "Ghi chú", level=1)
add_para(
    doc,
    "Tài liệu này được sinh tự động dựa trên đúng code hiện có của hệ thống tại thời điểm biên soạn "
    "(app.py, analyze_and_train.py, eda_analysis.py). Khi code thay đổi, nên chạy lại "
    "generate_documentation.py để cập nhật tài liệu, tránh tài liệu và code bị lệch nhau theo thời gian.",
)

output_path = "TAI_LIEU_KY_THUAT_HE_THONG.docx"
doc.save(output_path)
print("Da tao file:", output_path)

"""
Hằng số dùng CHUNG cho toàn bộ pipeline (app.py, analyze_and_train.py, train_model.py,
eda_analysis.py, hyperparameter_tuning.py).

TẠI SAO FILE NÀY TỒN TẠI: trước đây `FEATURE_COLS` được định nghĩa ĐỘC LẬP ở 5 file khác nhau (cùng
1 giá trị copy-paste) - nếu bộ đặc trưng của model thay đổi (thêm/bớt/đổi thứ tự cột), phải nhớ sửa
đúng cả 5 nơi, dễ sửa sót 1-2 chỗ khiến scaler/model suy luận sai lệch mà không có lỗi rõ ràng nào
báo trước (dữ liệu vẫn "chạy được", chỉ là sai). Nay chỉ còn DUY NHẤT 1 định nghĩa ở đây, các file
khác `from shared_constants import FEATURE_COLS`.

File này KHÔNG import bất kỳ thư viện nặng nào (pandas, numpy, sklearn...) - chỉ chứa hằng số Python
thuần, để import được an toàn ở MỌI nơi (kể cả app.py, vốn cố tình tránh import trực tiếp
`analyze_and_train.py` ở cấp module để không kéo theo TensorFlow/XGBoost/CTGAN chỉ để đọc 1 hằng số).
"""

# Tên 3 cột LAG/TÍCH LUỸ mưa (bổ sung để model thấy được mưa các ngày TRƯỚC đó, không chỉ đúng ngày
# hiện tại) - tách thành hằng số riêng để `analyze_and_train.py` (tính lag trên dữ liệu lịch sử nhiều
# địa phương) và `app.py` (tính lag trên dữ liệu dự báo Open-Meteo lúc suy luận) LUÔN dùng đúng 1 tên
# cột giống nhau, không copy-paste chuỗi ký tự dễ gõ sai ở 2 nơi khác nhau.
RAIN_LAG1_COL = "Lượng_mưa_mm_lag1"  # Lượng mưa của ĐÚNG 1 ngày trước
RAIN_LAG2_COL = "Lượng_mưa_mm_lag2"  # Lượng mưa của ĐÚNG 2 ngày trước
RAIN_ROLLING_3D_COL = "Lượng_mưa_tích_luỹ_3_ngày"  # Tổng mưa 3 ngày gần nhất (gồm cả ngày hiện tại)

# Danh sách đặc trưng đầu vào của model, ĐÚNG THỨ TỰ dùng khi huấn luyện (StandardScaler ghi nhớ thứ
# tự cột này qua `feature_names_in_`) - đổi thứ tự ở đây tương đương đổi thứ tự cột đưa vào model.
#
# 3 CỘT LAG MƯA (thêm mới): TRƯỚC ĐÂY model chỉ thấy lượng mưa của ĐÚNG 1 ngày (ngày đang xét) để dự
# báo nguy cơ ngập ngày HÔM SAU - bỏ sót hoàn toàn hiệu ứng "đất đã bão hoà nước" do mưa dồn dập nhiều
# ngày liên tiếp (một nguyên nhân gây ngập THẬT ở Huế, dù ngày dự báo riêng lẻ mưa không quá lớn). Bổ
# sung mưa của 1-2 ngày trước + tổng tích luỹ 3 ngày để model có tín hiệu này - xem
# `analyze_and_train.py::build_daily_feature_dataset()` (huấn luyện) và `app.py::predict_4_days_
# forecast()`/`predict_days_ahead_forecast_sequence()` (suy luận) để biết nơi các cột này được TÍNH ra
# (file này chỉ khai tên, không tính toán, vì cố tình không import pandas/numpy - xem docstring đầu
# file).
FEATURE_COLS = [
    "Nhiệt_độ_C",
    "Độ_ẩm_%",
    "Lượng_mưa_mm",
    "Độ_ẩm_đất",
    "Chiều_cao_triều_m",
    RAIN_LAG1_COL,
    RAIN_LAG2_COL,
    RAIN_ROLLING_3D_COL,
]

import requests
import folium

# 1. Khai báo API Key
API_KEY = "opH6G1fIc4yptvSrCqQ6iZI5yaifz1Je"# Thay key thật của bạn vào đây nếu cần

# 2. Khai báo tọa độ 
start_point = "10.7725,106.6980"  # Điểm A
end_point = "10.7923,106.6908"    # Điểm B

def test_routing_api():
    print("🚀 Đang gửi yêu cầu tìm đường đến TomTom API...")
    
    # Đã sửa: URL không cần chứa param avoidArea nữa
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_point}:{end_point}/json"
    
    # Params gửi kèm URL (Chỉ chứa Key và lệnh bật kẹt xe)
    params = {
        "key": API_KEY,
        "traffic": "true" # Đã sửa lại đúng chuẩn của TomTom
    }

    # Đã sửa: Khai báo Body JSON chuẩn của TomTom để né ngập
    payload = {
        "avoidAreas": {
            "rectangles": [
                {
                    "southWestCorner": {
                        "latitude": 10.7810,
                        "longitude": 106.6920
                    },
                    "northEastCorner": {
                        "latitude": 10.7850,
                        "longitude": 106.6970
                    }
                }
            ]
        }
    }

    # Bắt buộc phải có header này để TomTom hiểu mình gửi JSON
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Đã sửa: Dùng requests.post thay vì requests.get
        response = requests.post(url, params=params, json=payload, headers=headers)
        data = response.json()

        if "routes" in data and len(data["routes"]) > 0:
            print("✅ Tìm thấy đường đi thành công! Đang tiến hành vẽ bản đồ...")
            
            route_points = data["routes"][0]["legs"][0]["points"]
            route_coords = [[point["latitude"], point["longitude"]] for point in route_points]

            # VẼ BẢN ĐỒ
            m = folium.Map(location=[10.7725, 106.6980], zoom_start=14)

            folium.Marker([10.7725, 106.6980], popup="Điểm A", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker([10.7923, 106.6908], popup="Điểm B", icon=folium.Icon(color="blue")).add_to(m)

            bounds = [[10.7810, 106.6920], [10.7850, 106.6970]]
            folium.Rectangle(bounds, color="red", fill=True, popup="VÙNG NGẬP LỤT").add_to(m)

            folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

            m.save("ban_do_test.html")
            print("🎉 Xong! Hãy mở file 'ban_do_test.html' để xem kết quả.")
            
        else:
            print("❌ Lỗi: Không tìm thấy đường đi.")
            print(data)

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    test_routing_api()
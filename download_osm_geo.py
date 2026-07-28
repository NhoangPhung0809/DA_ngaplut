
import os
import osmnx as ox
import geopandas as gpd
import pandas as pd

def main():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    
    # Danh sách OSM Relation ID của các huyện Thừa Thiên Huế
    districts = [
        {"name": "TP Huế", "osm_id": "R4468442"},
        {"name": "Hương Thủy", "osm_id": "R4468443"},
        {"name": "Phú Vang", "osm_id": "R4468444"},
        {"name": "Hương Trà", "osm_id": "R4468445"},
        {"name": "Quảng Điền", "osm_id": "R4468446"},
    ]
    
    print("🔄 Đang tải ranh giới hành chính từ OpenStreetMap...")
    
    gdfs = []
    for district in districts:
        try:
            print(f"📥 Đang tải: {district['name']} (OSM: {district['osm_id']})")
            # Tải ranh giới từ OSM ID
            gdf = ox.geocode_to_gdf(district["osm_id"], by_osmid=True)
            gdf["name"] = district["name"]
            gdfs.append(gdf)
        except Exception as e:
            print(f"⚠️ Lỗi với {district['name']}: {e}")
    
    if gdfs:
        merged = pd.concat(gdfs, ignore_index=True)
        os.makedirs("data/geo", exist_ok=True)
        output_path = "data/geo/thuathienhue_districts.geojson"
        merged.to_file(output_path, driver="GeoJSON")
        
        print(f"\n✅ Thành công! Đã lưu {len(merged)} huyện vào: {output_path}")
        print("🎯 Bây giờ chạy lại app để xem ranh giới thực tế!")
    else:
        print("\n❌ Không tải được dữ liệu nào!")

if __name__ == "__main__":
    main()

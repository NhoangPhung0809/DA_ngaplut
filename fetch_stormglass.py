import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "data" / "stormglass_cache.json"
STORMGLASS_URL = "https://api.stormglass.io/v2/tide/extremes/point"
LATITUDE = 16.4637
LONGITUDE = 107.5909


def fetch_stormglass_tide_data():
    """Gọi Stormglass Tide Extremes API và trả về đúng JSON response."""
    api_key = os.getenv("STORMGLASS_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Thiếu STORMGLASS_API_KEY trong môi trường hoặc file .env.")

    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=2)
    params = {
        "lat": LATITUDE,
        "lng": LONGITUDE,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
    }
    headers = {
        "Authorization": api_key,
        "accept": "application/json",
    }

    response = requests.get(STORMGLASS_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_json_cache(payload):
    """Lưu nguyên JSON Stormglass vào file cache cục bộ."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main():
    payload = fetch_stormglass_tide_data()
    save_json_cache(payload)
    print(f"Đã lưu cache Stormglass vào: {CACHE_PATH}")


if __name__ == "__main__":
    main()

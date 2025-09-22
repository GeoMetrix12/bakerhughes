import requests, pandas as pd
from typing import Tuple

GC_URL = "https://geocoding-api.open-meteo.com/v1/search"
FC_URL = "https://api.open-meteo.com/v1/forecast"

def geocode_city(q: str) -> Tuple[float,float,str]:
    r = requests.get(GC_URL, params={"name": q, "count": 1})
    r.raise_for_status()
    js = r.json()
    if not js.get("results"): raise ValueError(f"No geocoding match for {q}")
    res = js["results"][0]
    return float(res["latitude"]), float(res["longitude"]), res.get("name", q)

def forecast_hours(lat: float, lon: float, hours: int = 48, tz: str = "auto") -> pd.DataFrame:
    r = requests.get(FC_URL, params={
        "latitude":lat, "longitude":lon, "timezone":tz,
        "hourly":"temperature_2m,dewpoint_2m,cloudcover,precipitation,pressure_msl,windspeed_10m,winddirection_10m"
    })
    r.raise_for_status()
    H = r.json()["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(H["time"]),
        "temp": H["temperature_2m"],
        "dew": H["dewpoint_2m"],
        "cloud": H["cloudcover"],
        "precip": H["precipitation"],
        "msl": H["pressure_msl"],
        "wind": H["windspeed_10m"],
        "winddir": H["winddirection_10m"],
    })
    return df.iloc[:hours].reset_index(drop=True)

def outline_bullets(df: pd.DataFrame) -> list:
    hot = int((df["temp"] >= 32).sum())
    rain = int((df["precip"] >= 0.5).sum())
    windy = int((df["wind"] >= 8).sum())
    bullets = []
    if hot: bullets.append(f"{hot} hot hours (>=32°C) — consider pre-cooling.")
    if rain: bullets.append(f"{rain} rainy hours — adjust daylight compensation and humidity.")
    if windy: bullets.append(f"{windy} windy hours — watch infiltration.")
    if not bullets: bullets.append("Mild conditions forecast.")
    return bullets

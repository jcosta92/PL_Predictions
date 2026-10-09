"""Download hourly weather at every Premier League stadium for every match, 2000-2026.

Source: Open-Meteo historical archive (https://open-meteo.com/en/docs/historical-weather-api),
free, no API key, ~10 km resolution, 1940 to a few days ago. Run this on a normal
internet connection; it makes one request per stadium (46 requests).

For each match it keeps the weather at kickoff (Football-Data has kickoff times from
2019/20; earlier matches are assumed to start at 15:00) and the rain in the 6 hours
before kickoff (a wet pitch matters more than rain during the match).

Output: data/raw/weather_openmeteo_kickoff.csv with one row per match:
    Date, HomeTeam, AwayTeam, temp_c, feels_like_c, rain_mm_6h, rain_mm_hour,
    wind_kmh, gust_kmh, humidity, cloud_pct, weather_code

Usage:  python fetch_weather.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

# current stadium of each club that played in the Premier League since 2000/01.
# Clubs that moved ground (Arsenal 2006, Man City 2003, West Ham 2016, Tottenham 2019,
# Brentford 2020, Southampton 2001, Leicester 2002) moved a few km at most, which is
# below the weather model's resolution.
STADIUMS = {
    "Arsenal": (51.5549, -0.1084), "Aston Villa": (52.5092, -1.8847), "Birmingham": (52.4756, -1.8681),
    "Blackburn": (53.7286, -2.4892), "Blackpool": (53.8046, -3.0481), "Bolton": (53.5805, -2.5356),
    "Bournemouth": (50.7352, -1.8383), "Bradford": (53.8042, -1.7590), "Brentford": (51.4907, -0.2888),
    "Brighton": (50.8616, -0.0837), "Burnley": (53.7890, -2.2302), "Cardiff": (51.4728, -3.2031),
    "Charlton": (51.4865, 0.0364), "Chelsea": (51.4817, -0.1910), "Coventry": (52.4481, -1.4956),
    "Crystal Palace": (51.3983, -0.0855), "Derby": (52.9150, -1.4472), "Everton": (53.4388, -2.9663),
    "Fulham": (51.4750, -0.2217), "Huddersfield": (53.6543, -1.7684), "Hull": (53.7462, -0.3677),
    "Ipswich": (52.0551, 1.1446), "Leeds": (53.7778, -1.5721), "Leicester": (52.6204, -1.1422),
    "Liverpool": (53.4308, -2.9608), "Luton": (51.8843, -0.4316), "Man City": (53.4831, -2.2004),
    "Man United": (53.4631, -2.2913), "Middlesbrough": (54.5781, -1.2168), "Newcastle": (54.9756, -1.6217),
    "Norwich": (52.6222, 1.3092), "Nott'm Forest": (52.9399, -1.1329), "Portsmouth": (50.7964, -1.0639),
    "QPR": (51.5093, -0.2322), "Reading": (51.4222, -0.9828), "Sheffield United": (53.3703, -1.4710),
    "Southampton": (50.9058, -1.3910), "Stoke": (52.9884, -2.1755), "Sunderland": (54.9144, -1.3882),
    "Swansea": (51.6428, -3.9348), "Tottenham": (51.6043, -0.0664), "Watford": (51.6499, -0.4015),
    "West Brom": (52.5090, -1.9639), "West Ham": (51.5387, -0.0166), "Wigan": (53.5477, -2.6538),
    "Wolves": (52.5903, -2.1304),
}
HOURLY = ["temperature_2m", "apparent_temperature", "precipitation", "wind_speed_10m", "wind_gusts_10m",
          "relative_humidity_2m", "cloud_cover", "weather_code"]


def kickoff_times() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "data" / "raw" / "football_data_E0_2000_2026.csv",
                      usecols=["MatchDate", "HomeTeam", "AwayTeam", "MatchTime"])
    raw = raw.rename(columns={"MatchDate": "Date"}).replace({"HomeTeam": {"Nottm Forest": "Nott'm Forest"},
                                                            "AwayTeam": {"Nottm Forest": "Nott'm Forest"}})
    raw["Date"] = pd.to_datetime(raw.Date)
    raw["hour"] = pd.to_datetime(raw.MatchTime, format="%H:%M:%S", errors="coerce").dt.hour.fillna(15).astype(int)
    return raw[["Date", "HomeTeam", "AwayTeam", "hour"]]


def fetch_stadium(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY), "timezone": "Europe/London", "wind_speed_unit": "kmh"})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h.time)
    return h


def main() -> None:
    matches = pd.read_csv(ROOT / "data" / "processed" / "matches.csv", parse_dates=["Date"])
    matches = matches.merge(kickoff_times(), on=["Date", "HomeTeam", "AwayTeam"], how="left")
    matches["hour"] = matches.hour.fillna(15).astype(int)
    out = []
    for team, (lat, lon) in STADIUMS.items():
        home = matches[matches.HomeTeam == team]
        if home.empty:
            continue
        start, end = home.Date.min().strftime("%Y-%m-%d"), home.Date.max().strftime("%Y-%m-%d")
        for attempt in range(3):
            try:
                h = fetch_stadium(lat, lon, start, end)
                break
            except requests.RequestException as e:
                print(f"{team}: {e}; retrying")
                time.sleep(10)
        else:
            raise SystemExit(f"could not fetch {team}")
        h = h.set_index("time")
        for r in home.itertuples(index=False):
            ko = r.Date + pd.Timedelta(hours=int(r.hour))
            if ko not in h.index:
                continue
            w = h.loc[ko]
            before = h.loc[ko - pd.Timedelta(hours=6): ko, "precipitation"].sum()
            out.append({"Date": r.Date, "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
                        "temp_c": w.temperature_2m, "feels_like_c": w.apparent_temperature,
                        "rain_mm_6h": before, "rain_mm_hour": w.precipitation, "wind_kmh": w.wind_speed_10m,
                        "gust_kmh": w.wind_gusts_10m, "humidity": w.relative_humidity_2m,
                        "cloud_pct": w.cloud_cover, "weather_code": w.weather_code})
        print(f"{team}: {len(home)} home matches", flush=True)
        time.sleep(1)  # be polite to the free API
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "data" / "raw" / "weather_openmeteo_kickoff.csv", index=False)
    print(f"{len(df)} of {len(matches)} matches with kickoff weather")


if __name__ == "__main__":
    main()

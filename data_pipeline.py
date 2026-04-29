"""
Data Pipeline — Weather & Electricity Price Feeds for the Heat Pump Controller.

This module provides:
1. BrightSky API integration for German weather data (free, no API key)
2. aWATTar API integration for electricity prices
3. Synthetic data generators for offline training
4. Grid signal loading from i4b's bundled data

All data is returned as pandas DataFrames with datetime indices,
ready to be consumed by the simulation environment.
"""

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Tuple
import warnings

from config import PRICE_CONFIG, I4B_ROOT


# ============================================================================
# 1. Weather Data — BrightSky API (German DWD data)
# ============================================================================

class BrightSkyClient:
    """Client for the BrightSky API (free German weather data from DWD).

    BrightSky provides hourly weather observations and forecasts for any
    location in Germany. No API key is required.

    Reference: https://brightsky.dev/
    """

    BASE_URL = PRICE_CONFIG["brightsky_base_url"]

    def __init__(self, lat: float = None, lon: float = None):
        self.lat = lat or PRICE_CONFIG["default_lat"]
        self.lon = lon or PRICE_CONFIG["default_lon"]

    def fetch_weather(
        self,
        date: str,
        last_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch historical or forecast weather data.

        Parameters
        ----------
        date : str
            Start date in YYYY-MM-DD format.
        last_date : str, optional
            End date. If None, fetches one day.

        Returns
        -------
        pd.DataFrame
            Hourly weather data with columns:
            - T_amb: Ambient temperature [°C]
            - solar: Solar irradiance [kWh/m²]
            - wind_speed: Wind speed [km/h]
            - cloud_cover: Cloud cover [%]
            - humidity: Relative humidity [%]
        """
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "date": date,
        }
        if last_date:
            params["last_date"] = last_date

        try:
            response = requests.get(
                f"{self.BASE_URL}/weather",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("weather"):
                warnings.warn(f"No weather data returned for {date}")
                return pd.DataFrame()

            records = []
            for entry in data["weather"]:
                records.append({
                    "timestamp": pd.Timestamp(entry["timestamp"]),
                    "T_amb": entry.get("temperature"),
                    "solar": entry.get("solar", 0.0),
                    "wind_speed": entry.get("wind_speed", 0.0),
                    "cloud_cover": entry.get("cloud_cover", 0.0),
                    "humidity": entry.get("relative_humidity", 50.0),
                })

            df = pd.DataFrame(records)
            df = df.set_index("timestamp")
            df.index = df.index.tz_convert("Europe/Berlin")
            return df.astype(float)

        except requests.RequestException as e:
            warnings.warn(f"BrightSky API request failed: {e}")
            return pd.DataFrame()

    def fetch_forecast(self, date: str) -> pd.DataFrame:
        """Fetch weather forecast (up to 10 days ahead).

        Parameters
        ----------
        date : str
            Start date in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            Same format as fetch_weather().
        """
        # BrightSky returns forecasts for future dates automatically
        return self.fetch_weather(date)


# ============================================================================
# 2. Electricity Prices — aWATTar API
# ============================================================================

class AWattarClient:
    """Client for electricity prices via aWATTar API.

    Fetches electricity market data from aWATTar.
    No API key is required.
    
    If request fails, falls back to synthetic price generation.
    """

    BASE_URL = "https://api.awattar.at/v1/marketdata"

    def fetch_day_ahead_prices(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch day-ahead electricity prices.

        Parameters
        ----------
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            Hourly prices with columns:
            - price_eur_mwh: Day-ahead price [€/MWh]
            - price_eur_kwh: Price [€/kWh]
            - price_rank: Rank within day (0=cheapest, 23=most expensive)
            - is_cheap_hour: True if below daily median
        """
        try:
            start_ts = pd.Timestamp(start, tz="Europe/Berlin")
            end_ts = pd.Timestamp(end, tz="Europe/Berlin") + pd.Timedelta(days=1)
            
            # aWATTar expects timestamps in milliseconds
            params = {
                "start": int(start_ts.timestamp() * 1000),
                "end": int(end_ts.timestamp() * 1000)
            }
            
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("data"):
                warnings.warn(f"No price data returned from aWATTar. Using synthetic prices.")
                return self.generate_synthetic_prices(start, end)
                
            records = []
            for entry in data["data"]:
                records.append({
                    "timestamp": pd.Timestamp(entry["start_timestamp"], unit="ms", tz="UTC").tz_convert("Europe/Berlin"),
                    "price_eur_mwh": entry["marketprice"]
                })
                
            df = pd.DataFrame(records)
            df = df.set_index("timestamp")
            
            # Ensure index frequency is hourly
            df = df.resample('h').mean().ffill()

            df["price_eur_kwh"] = df["price_eur_mwh"] / 1000
            df["price_rank"] = df.groupby(df.index.date)["price_eur_mwh"].rank()
            df["is_cheap_hour"] = df.groupby(df.index.date)[
                "price_eur_mwh"
            ].transform(lambda x: x < x.median())
            return df

        except Exception as e:
            warnings.warn(f"aWATTar query failed: {e}. Using synthetic prices.")
            return self.generate_synthetic_prices(start, end)

    @staticmethod
    def generate_synthetic_prices(
        start: str,
        end: str,
        base_price: float = None,
    ) -> pd.DataFrame:
        """Generate realistic synthetic electricity prices.

        Creates a daily price profile that mimics German spot market behavior:
        - Low prices overnight (wind/solar oversupply)
        - Morning ramp-up (demand increase)
        - Midday dip (solar peak)
        - Evening peak (high demand + low solar)

        Parameters
        ----------
        start, end : str
            Date range in YYYY-MM-DD format.
        base_price : float
            Base price in €/kWh (default from config).

        Returns
        -------
        pd.DataFrame
            Same format as fetch_day_ahead_prices().
        """
        base = base_price or PRICE_CONFIG["synthetic_base_price"]
        peak_hours = PRICE_CONFIG["peak_hours"]

        idx = pd.date_range(start, end, freq="h", tz="Europe/Berlin")
        np.random.seed(42)

        prices = []
        for ts in idx:
            hour = ts.hour

            # Base pattern: German electricity market shape
            if hour in peak_hours:
                # Peak: morning ramp + evening peak
                multiplier = PRICE_CONFIG["synthetic_peak_multiplier"]
            elif 0 <= hour <= 5:
                # Night: low demand
                multiplier = PRICE_CONFIG["synthetic_off_peak_multiplier"]
            elif 10 <= hour <= 15:
                # Midday: solar duck curve dip
                multiplier = 0.75
            else:
                multiplier = 1.0

            # Add daily/seasonal variation
            day_noise = np.random.normal(0, 0.03)
            seasonal = 0.1 * np.sin(2 * np.pi * ts.dayofyear / 365)
            price = base * (multiplier + day_noise + seasonal)
            prices.append(max(0.01, price))  # Floor at 1 ct/kWh

        df = pd.DataFrame(
            {"price_eur_kwh": prices},
            index=idx,
        )
        df["price_eur_mwh"] = df["price_eur_kwh"] * 1000
        df["price_rank"] = df.groupby(df.index.date)["price_eur_mwh"].rank()
        df["is_cheap_hour"] = df.groupby(df.index.date)[
            "price_eur_mwh"
        ].transform(lambda x: x < x.median())
        return df


# ============================================================================
# 3. i4b Grid Signal Loader
# ============================================================================

def load_i4b_grid_signal(resolution: str = "15min") -> pd.DataFrame:
    """Load bundled grid signal data from i4b.

    Parameters
    ----------
    resolution : str
        '1h' for hourly or '15min' for 15-minute resolution.

    Returns
    -------
    pd.DataFrame
        Grid signal timeseries from i4b's data/grid/ directory.
    """
    if resolution == "15min":
        path = I4B_ROOT / "data" / "grid" / "grid_signals_15min.csv"
    else:
        path = I4B_ROOT / "data" / "grid" / "grid_signals.csv"

    if not path.exists():
        raise FileNotFoundError(f"Grid signal file not found: {path}")

    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return df


# ============================================================================
# 4. Synthetic Weather Generator (for training without API)
# ============================================================================

def generate_synthetic_weather(
    start: str,
    end: str,
    latitude: float = 49.87,
    freq: str = "h",
) -> pd.DataFrame:
    """Generate synthetic seasonal weather data for Germany.

    Creates realistic temperature, solar irradiance, and wind patterns
    based on typical Central European climate statistics.

    Parameters
    ----------
    start, end : str
        Date range in YYYY-MM-DD format.
    latitude : float
        Latitude for solar geometry (default: Darmstadt).
    freq : str
        Frequency ('h' for hourly, '15min' for quarter-hourly).

    Returns
    -------
    pd.DataFrame
        Synthetic weather with columns:
        - T_amb: Ambient temperature [°C]
        - solar: Global horizontal irradiance [W/m²]
        - wind_speed: Wind speed [m/s]
        - cloud_cover: Cloud cover [%]
    """
    idx = pd.date_range(start, end, freq=freq, tz="Europe/Berlin")
    np.random.seed(42)

    n = len(idx)
    doy = idx.dayofyear.values  # Day of year
    hour = idx.hour.values + idx.minute.values / 60.0

    # Temperature: seasonal + diurnal + noise
    # German seasonal: -2°C (Jan) to 20°C (Jul) mean
    T_seasonal = 9 + 11 * np.sin(2 * np.pi * (doy - 100) / 365)
    T_diurnal = 4 * np.sin(2 * np.pi * (hour - 6) / 24)
    T_noise = np.random.normal(0, 2, n)
    T_amb = T_seasonal + T_diurnal + T_noise

    # Solar irradiance: seasonal * time-of-day envelope
    max_solar_seasonal = 200 + 600 * np.sin(
        np.clip(2 * np.pi * (doy - 80) / 365, 0, np.pi)
    )
    # Simple daylight mask
    sunrise = 6 - 2 * np.sin(2 * np.pi * (doy - 80) / 365)
    sunset = 18 + 2 * np.sin(2 * np.pi * (doy - 80) / 365)
    daylight_mask = ((hour >= sunrise) & (hour <= sunset)).astype(float)
    solar_base = max_solar_seasonal * np.sin(
        np.clip(np.pi * (hour - sunrise) / (sunset - sunrise), 0, np.pi)
    ) * daylight_mask
    # Cloud attenuation
    cloud = np.random.beta(2, 5, n) * 100  # Skewed towards clear
    solar = solar_base * (1 - 0.7 * cloud / 100)
    solar = np.clip(solar, 0, 1000)

    # Wind: mostly 2-6 m/s with occasional gusts
    wind = np.random.weibull(2, n) * 4 + 1

    df = pd.DataFrame(
        {
            "T_amb": T_amb,
            "solar": solar,
            "wind_speed": wind,
            "cloud_cover": cloud,
        },
        index=idx,
    )
    return df


# ============================================================================
# 5. Combined Data Pipeline
# ============================================================================

class DataPipeline:
    """Unified data pipeline combining weather and price data.

    Provides a single interface to fetch or generate all data needed
    by the simulation environment.
    """

    def __init__(
        self,
        lat: float = None,
        lon: float = None,
    ):
        self.weather_client = BrightSkyClient(lat=lat, lon=lon)
        self.price_client = AWattarClient()

    def get_training_data(
        self,
        start: str,
        end: str,
        use_synthetic: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get weather and price data for training.

        Parameters
        ----------
        start, end : str
            Date range in YYYY-MM-DD format.
        use_synthetic : bool
            If True, generates synthetic data (faster, no API needed).

        Returns
        -------
        tuple of (weather_df, price_df)
        """
        if use_synthetic:
            weather = generate_synthetic_weather(start, end)
            prices = AWattarClient.generate_synthetic_prices(start, end)
        else:
            weather = self.weather_client.fetch_weather(start, end)
            prices = self.price_client.fetch_day_ahead_prices(start, end)

        return weather, prices

    def get_live_data(self, date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get live weather and price data for a specific day.

        Parameters
        ----------
        date : str
            Date in YYYY-MM-DD format.

        Returns
        -------
        tuple of (weather_df, price_df)
        """
        weather = self.weather_client.fetch_weather(date)
        prices = self.price_client.fetch_day_ahead_prices(date, date)
        return weather, prices

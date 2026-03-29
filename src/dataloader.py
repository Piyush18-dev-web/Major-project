"""
data_loader.py - Load, generate, and merge multiple traffic datasets.

Supports:
  - CSV, JSON, Excel file loading
  - Synthetic dataset generation
  - Multi-dataset merging on timestamp + location
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils import setup_logger, ensure_dir

logger = setup_logger()


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATORS
# ─────────────────────────────────────────────

def generate_traffic_volume(start: str = "2024-01-01", days: int = 90,
                             segments: list = None, output_dir: str = "data") -> pd.DataFrame:
    """
    Generate synthetic hourly traffic volume data.

    Args:
        start: Start date string (YYYY-MM-DD)
        days: Number of days to simulate
        segments: List of road segment names
        output_dir: Directory to save CSV

    Returns:
        DataFrame with columns: timestamp, segment_id, volume, actual_speed, free_flow_speed
    """
    if segments is None:
        segments = ["Seg_A", "Seg_B", "Seg_C", "Seg_D"]

    np.random.seed(42)
    records = []
    base_dt = datetime.strptime(start, "%Y-%m-%d")

    for day in range(days):
        for hour in range(24):
            dt = base_dt + timedelta(days=day, hours=hour)
            is_weekend = dt.weekday() >= 5
            is_peak = (7 <= hour <= 9) or (17 <= hour <= 20)

            for seg in segments:
                # Base volume: higher on weekdays and during peak hours
                base_vol = 1200 if is_peak and not is_weekend else 600 if not is_weekend else 400
                # Add random noise
                volume = int(np.random.normal(base_vol, base_vol * 0.15))
                volume = max(50, volume)

                free_flow_speed = np.random.choice([60, 80, 100])  # km/h
                # Speed drops when volume is high
                congestion_factor = max(0.1, 1 - (volume / 3000))
                actual_speed = round(free_flow_speed * congestion_factor + np.random.normal(0, 3), 1)
                actual_speed = max(5.0, actual_speed)

                records.append({
                    "timestamp": dt,
                    "segment_id": seg,
                    "volume": volume,
                    "actual_speed": actual_speed,
                    "free_flow_speed": free_flow_speed
                })

    df = pd.DataFrame(records)
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "traffic_volume.csv")
    df.to_csv(path, index=False)
    logger.info(f"Traffic volume dataset saved → {path} ({len(df)} rows)")
    return df


def generate_weather(start: str = "2024-01-01", days: int = 90,
                     output_dir: str = "data") -> pd.DataFrame:
    """
    Generate synthetic hourly weather data.

    Args:
        start: Start date string (YYYY-MM-DD)
        days: Number of days to simulate
        output_dir: Directory to save CSV

    Returns:
        DataFrame with columns: timestamp, temperature, rainfall_mm, visibility_km, condition
    """
    np.random.seed(7)
    records = []
    base_dt = datetime.strptime(start, "%Y-%m-%d")
    conditions = ['clear', 'cloudy', 'foggy', 'rainy', 'stormy']
    condition_weights = [0.40, 0.25, 0.10, 0.20, 0.05]

    for day in range(days):
        # Pick a daily condition that persists with variation
        daily_condition = np.random.choice(conditions, p=condition_weights)
        for hour in range(24):
            dt = base_dt + timedelta(days=day, hours=hour)
            temp = round(np.random.normal(25, 8), 1)

            # Adjust weather metrics based on condition
            if daily_condition == 'clear':
                rain, vis = 0.0, round(np.random.uniform(8, 12), 1)
                cond = 'clear'
            elif daily_condition == 'cloudy':
                rain, vis = 0.0, round(np.random.uniform(5, 9), 1)
                cond = 'cloudy'
            elif daily_condition == 'foggy':
                rain, vis = 0.0, round(np.random.uniform(0.5, 3), 1)
                cond = 'foggy'
            elif daily_condition == 'rainy':
                rain = round(np.random.uniform(2, 20), 1)
                vis = round(np.random.uniform(2, 6), 1)
                cond = 'rainy'
            else:  # stormy
                rain = round(np.random.uniform(20, 60), 1)
                vis = round(np.random.uniform(0.2, 2), 1)
                cond = 'stormy'

            records.append({
                "timestamp": dt,
                "temperature": temp,
                "rainfall_mm": rain,
                "visibility_km": vis,
                "condition": cond
            })

    df = pd.DataFrame(records)
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "weather.csv")
    df.to_csv(path, index=False)
    logger.info(f"Weather dataset saved → {path} ({len(df)} rows)")
    return df


def generate_accidents(start: str = "2024-01-01", days: int = 90,
                        segments: list = None, output_dir: str = "data") -> pd.DataFrame:
    """
    Generate synthetic accident/incident dataset.

    Args:
        start: Start date string (YYYY-MM-DD)
        days: Number of days to simulate
        segments: List of road segment names
        output_dir: Directory to save CSV

    Returns:
        DataFrame with columns: timestamp, segment_id, incident_type, duration_min, severity
    """
    if segments is None:
        segments = ["Seg_A", "Seg_B", "Seg_C", "Seg_D"]

    np.random.seed(13)
    records = []
    base_dt = datetime.strptime(start, "%Y-%m-%d")
    incident_types = ['accident', 'breakdown', 'roadwork', 'flooding']

    for day in range(days):
        # On average 2-5 incidents per day across all segments
        n_incidents = np.random.randint(1, 6)
        for _ in range(n_incidents):
            hour = np.random.randint(0, 24)
            dt = base_dt + timedelta(days=day, hours=hour)
            seg = np.random.choice(segments)
            itype = np.random.choice(incident_types, p=[0.4, 0.3, 0.2, 0.1])
            duration = int(np.random.exponential(45))  # minutes
            severity = np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.35, 0.15])
            records.append({
                "timestamp": dt,
                "segment_id": seg,
                "incident_type": itype,
                "duration_min": duration,
                "severity": severity
            })

    df = pd.DataFrame(records)
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "accidents.csv")
    df.to_csv(path, index=False)
    logger.info(f"Accidents dataset saved → {path} ({len(df)} rows)")
    return df


def generate_road_network(segments: list = None, output_dir: str = "data") -> pd.DataFrame:
    """
    Generate a static road network metadata dataset.

    Args:
        segments: List of road segment names
        output_dir: Directory to save CSV

    Returns:
        DataFrame with columns: segment_id, road_type, lanes, speed_limit_kmh, length_km
    """
    if segments is None:
        segments = ["Seg_A", "Seg_B", "Seg_C", "Seg_D"]

    np.random.seed(99)
    road_types = ['highway', 'arterial', 'local', 'expressway']
    records = []

    for seg in segments:
        rtype = np.random.choice(road_types)
        lanes = np.random.choice([2, 4, 6])
        speed_limit = {'highway': 100, 'expressway': 120, 'arterial': 60, 'local': 40}[rtype]
        length = round(np.random.uniform(1, 15), 1)
        records.append({
            "segment_id": seg,
            "road_type": rtype,
            "lanes": lanes,
            "speed_limit_kmh": speed_limit,
            "length_km": length
        })

    df = pd.DataFrame(records)
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "road_network.csv")
    df.to_csv(path, index=False)
    logger.info(f"Road network dataset saved → {path} ({len(df)} rows)")
    return df


def generate_all_datasets(output_dir: str = "data") -> dict:
    """
    Generate all five synthetic datasets and return them as a dictionary.

    Args:
        output_dir: Directory to save all CSVs

    Returns:
        Dictionary of DataFrames keyed by dataset name
    """
    segments = ["Seg_A", "Seg_B", "Seg_C", "Seg_D"]
    print("\n🔧 Generating synthetic datasets...")
    datasets = {
        "traffic": generate_traffic_volume(segments=segments, output_dir=output_dir),
        "weather": generate_weather(output_dir=output_dir),
        "accidents": generate_accidents(segments=segments, output_dir=output_dir),
        "road_network": generate_road_network(segments=segments, output_dir=output_dir),
    }
    print(f"✅ All datasets saved to '{output_dir}/' folder.\n")
    return datasets


# ─────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────

def load_file(filepath: str) -> pd.DataFrame:
    """
    Auto-detect and load a data file (CSV, JSON, or Excel).

    Args:
        filepath: Path to the file

    Returns:
        Loaded DataFrame
    """
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    logger.info(f"Loaded '{filepath}' → {df.shape}")
    return df


def load_from_directory(directory: str) -> dict:
    """
    Load all recognized data files from a directory.
    Detects dataset type by filename keywords.

    Args:
        directory: Path to folder containing dataset files

    Returns:
        Dictionary of DataFrames by type
    """
    datasets = {}
    keyword_map = {
        'traffic': 'traffic',
        'weather': 'weather',
        'accident': 'accidents',
        'road': 'road_network',
        'incident': 'accidents'
    }

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[-1].lower()
        if ext not in ['.csv', '.json', '.xls', '.xlsx']:
            continue

        # Auto-detect type from filename
        detected_key = None
        for keyword, key in keyword_map.items():
            if keyword in fname.lower():
                detected_key = key
                break
        if detected_key is None:
            detected_key = os.path.splitext(fname)[0]

        try:
            datasets[detected_key] = load_file(fpath)
            logger.info(f"Detected '{detected_key}' from file '{fname}'")
        except Exception as e:
            logger.warning(f"Could not load '{fname}': {e}")

    return datasets


# ─────────────────────────────────────────────
# DATASET MERGER
# ─────────────────────────────────────────────

def merge_datasets(datasets: dict) -> pd.DataFrame:
    """
    Merge all loaded datasets into a single DataFrame.
    Primary join: traffic (left) + weather on timestamp hour
                + accidents aggregated per (timestamp_hour, segment)
                + road_network on segment_id

    Args:
        datasets: Dictionary of DataFrames

    Returns:
        Merged DataFrame ready for preprocessing
    """
    # ── Traffic (base) ──
    df = datasets.get('traffic')
    if df is None:
        raise ValueError("Traffic dataset is required for merging.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Round timestamp to nearest hour for joining
    df['ts_hour'] = df['timestamp'].dt.floor('h')

    # ── Weather ──
    if 'weather' in datasets:
        w = datasets['weather'].copy()
        w['timestamp'] = pd.to_datetime(w['timestamp'])
        w['ts_hour'] = w['timestamp'].dt.floor('h')
        w = w.drop(columns=['timestamp'])
        df = df.merge(w, on='ts_hour', how='left')
        logger.info("Merged: weather ✓")

    # ── Accidents ──
    if 'accidents' in datasets:
        a = datasets['accidents'].copy()
        a['timestamp'] = pd.to_datetime(a['timestamp'])
        a['ts_hour'] = a['timestamp'].dt.floor('h')
        # Aggregate: count incidents and total delay per (hour, segment)
        a_agg = a.groupby(['ts_hour', 'segment_id']).agg(
            incident_count=('incident_type', 'count'),
            total_duration_min=('duration_min', 'sum')
        ).reset_index()
        df = df.merge(a_agg, on=['ts_hour', 'segment_id'], how='left')
        df['incident_count'] = df['incident_count'].fillna(0)
        df['total_duration_min'] = df['total_duration_min'].fillna(0)
        logger.info("Merged: accidents ✓")

    # ── Road Network ──
    if 'road_network' in datasets:
        rn = datasets['road_network'].copy()
        df = df.merge(rn, on='segment_id', how='left')
        logger.info("Merged: road_network ✓")

    df = df.drop(columns=['ts_hour'], errors='ignore')
    logger.info(f"Final merged shape: {df.shape}")
    return df

"""Vergelijking Humor 20 RH-calibrator en dauwpuntsmeter (RH + T)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_EXTENSIONS = {".csv", ".txt", ".dat"}

TYPICAL_SETPOINTS = np.array(
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    dtype=float,
)


def list_data_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    skip_names = {"requirements.txt", "readme.txt", "gebruik.txt"}
    files = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in DATA_EXTENSIONS
        and p.name.lower() not in skip_names
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _read_text_head(path: Path, max_lines: int = 40) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            lines: list[str] = []
            with path.open(encoding=encoding, errors="strict") as handle:
                for i, line in enumerate(handle):
                    if i >= max_lines:
                        break
                    lines.append(line)
            return "".join(lines)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Kan bestandskop niet lezen: {path.name} ({last_error})")


def _sensor_header_row(path: Path) -> int | None:
    """Zoek de headerrij van een sensor-export (Date Time / Temperature / Humidity)."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open(encoding=encoding, errors="strict") as handle:
                for i, line in enumerate(handle):
                    low = line.lower().replace('"', "")
                    if "date time" in low and ("humidity" in low or "%rh" in low):
                        return i
                    if i > 80:
                        break
            return None
        except Exception:  # noqa: BLE001
            continue
    return None


def _is_sensor_export(path: Path) -> bool:
    try:
        head = _read_text_head(path).lower()
    except ValueError:
        return False
    if "exported sensor data" in head:
        return True
    if _sensor_header_row(path) is not None and (
        "humidity (%rh)" in head or "temperature (c)" in head
    ):
        return True
    return False


def _read_raw_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", encoding=encoding, engine="python")
            if df.shape[1] == 1:
                df = pd.read_csv(path, sep=",", encoding=encoding, engine="python")
            df.columns = [str(c).strip().rstrip(";") for c in df.columns]
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Kan bestand niet lezen: {path.name} ({last_error})")


def _to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _find_column(df: pd.DataFrame, *needles: str, exclude: tuple[str, ...] = ()) -> str | None:
    lowered = [(c, c.lower()) for c in df.columns]
    for original, low in lowered:
        if any(ex.lower() in low for ex in exclude):
            continue
        if all(n.lower() in low for n in needles):
            return original
    return None


def classify_file(path: str | Path) -> str:
    path = Path(path)
    if _is_sensor_export(path):
        return "sensor"

    df = _read_raw_table(path)
    joined = " ".join(df.columns).lower()
    if "humor" in joined:
        return "humor"
    if "dew point" in joined or "dauwpunt" in joined:
        return "dpm"
    if _find_column(df, "rh") and (
        _find_column(df, "temp") or _find_column(df, "datetime")
    ):
        if _find_column(df, "mirror") or _find_column(df, "frost"):
            return "dpm"
        return "humor"
    return "onbekend"


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    """Parse datetimes with fixed formats — dateutil fallback crashes on Python 3.14."""
    text = values.astype(str).str.strip()
    lower = text.str.lower()
    invalid = (
        text.isin(["", "nan", "None", "NaT", "NaN"])
        | lower.str.startswith("date")
        | lower.str.contains(r"time;|datetime", regex=True, na=False)
    )
    text = text.mask(invalid)

    formats = (
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    )
    best: pd.Series | None = None
    best_count = -1
    for fmt in formats:
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        count = int(parsed.notna().sum())
        if count > best_count:
            best = parsed
            best_count = count
        if best_count == int((~invalid).sum()) and best_count > 0:
            break
    if best is None:
        return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    return best.astype("datetime64[ns]")


def _load_humor_raw(path: str | Path) -> pd.DataFrame:
    df = _read_raw_table(path)
    date_col = _find_column(df, "date")
    time_col = _find_column(df, "time")
    temp_col = _find_column(df, "temp") or _find_column(df, "t")
    rh_col = _find_column(df, "rh")

    if temp_col is None or rh_col is None:
        raise ValueError(f"Humor-bestand mist temperatuur- of RH-kolom: {Path(path).name}")

    if date_col and time_col and date_col != time_col:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        tijd = _parse_datetime_series(combined)
    elif _find_column(df, "datetime"):
        tijd = _parse_datetime_series(df[_find_column(df, "datetime")])
    else:
        raise ValueError(f"Humor-bestand mist datum/tijd-kolommen: {Path(path).name}")

    out = pd.DataFrame(
        {
            "tijd": tijd,
            "t": _to_float(df[temp_col]),
            "rh": _to_float(df[rh_col]),
        }
    )
    out = out.dropna(subset=["tijd", "t", "rh"]).sort_values("tijd")
    return out.drop_duplicates(subset=["tijd"]).reset_index(drop=True)


def _load_dpm_raw(path: str | Path) -> pd.DataFrame:
    df = _read_raw_table(path)
    time_col = _find_column(df, "datetime") or _find_column(df, "date")
    t_col = (
        _find_column(df, "external temp")
        or _find_column(df, "temp")
        or _find_column(df, "head temp")
    )
    rh_col = None
    for col in df.columns:
        low = col.lower().strip()
        if low.endswith(" rh") or low == "rh":
            rh_col = col
            break
    if rh_col is None:
        rh_col = _find_column(df, "rh", exclude=("rhw", "pid"))

    if time_col is None:
        raise ValueError(f"DPM-bestand mist tijdkolom: {Path(path).name}")
    if t_col is None or rh_col is None:
        raise ValueError(f"DPM-bestand mist temperatuur- of RH-kolom: {Path(path).name}")

    tijd = _parse_datetime_series(df[time_col])
    rh = _to_float(df[rh_col])
    rh_median = rh.dropna().median()
    if pd.notna(rh_median) and rh_median <= 1.5:
        rh = rh * 100.0

    out = pd.DataFrame(
        {
            "tijd": tijd,
            "t": _to_float(df[t_col]),
            "rh": rh,
        }
    )
    out = out.dropna(subset=["tijd", "t", "rh"]).sort_values("tijd")
    return out.drop_duplicates(subset=["tijd"]).reset_index(drop=True)


def _load_sensor_export_raw(path: str | Path) -> pd.DataFrame:
    """Laad CSV-export met metadata-kop (Date Time / Temperature / Humidity)."""
    path = Path(path)
    header_row = _sensor_header_row(path)
    if header_row is None:
        raise ValueError(f"Sensor-export mist header 'Date Time' / Humidity: {path.name}")

    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                skiprows=header_row,
                encoding=encoding,
                engine="python",
                on_bad_lines="skip",
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if df is None:
        raise ValueError(f"Kan sensor-export niet lezen: {path.name} ({last_error})")

    df.columns = [str(c).strip().strip('"') for c in df.columns]
    time_col = (
        _find_column(df, "date time")
        or _find_column(df, "datetime")
        or _find_column(df, "date")
    )
    temp_col = _find_column(df, "temperature") or _find_column(df, "temp")
    rh_col = _find_column(df, "humidity") or _find_column(df, "%rh") or _find_column(df, "rh")

    if time_col is None or temp_col is None or rh_col is None:
        raise ValueError(
            f"Sensor-export mist tijd-, temperatuur- of RH-kolom: {path.name}"
        )

    out = pd.DataFrame(
        {
            "tijd": _parse_datetime_series(df[time_col]),
            "t": _to_float(df[temp_col]),
            "rh": _to_float(df[rh_col]),
        }
    )
    out = out.dropna(subset=["tijd", "t", "rh"]).sort_values("tijd")
    return out.drop_duplicates(subset=["tijd"]).reset_index(drop=True)


def load_instrument_file(path: str | Path) -> tuple[str, pd.DataFrame]:
    """Laad meetbestand als (soort, dataframe met kolommen tijd / t / rh)."""
    kind = classify_file(path)
    if kind == "humor":
        return kind, _load_humor_raw(path)
    if kind == "dpm":
        return kind, _load_dpm_raw(path)
    if kind == "sensor":
        return kind, _load_sensor_export_raw(path)
    raise ValueError(f"Bestandstype niet herkend: {Path(path).name}")


def load_instrument_bytes(filename: str, data: bytes) -> tuple[str, pd.DataFrame]:
    """Laad een geüpload bestand (bytes) via een tijdelijke map."""
    import tempfile

    safe_name = Path(filename).name or "upload.csv"
    with tempfile.TemporaryDirectory(prefix="humorrh_") as tmpdir:
        path = Path(tmpdir) / safe_name
        path.write_bytes(data)
        return load_instrument_file(path)


def _ts_ns(value) -> int:
    """Timestamp as int64 nanoseconds (avoids pandas.Timedelta on Python 3.14)."""
    return int(pd.Timestamp(value).value)


def _shift_datetime_series(series: pd.Series, seconds: float) -> pd.Series:
    if not seconds:
        return series
    ns = series.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ns = ns + int(float(seconds) * 1_000_000_000)
    return pd.to_datetime(ns, unit="ns")


def shift_datetime_series(series: pd.Series, seconds: float) -> pd.Series:
    """Publieke wrapper voor tijdoffset zonder pandas.Timedelta."""
    return _shift_datetime_series(series, seconds)


def filter_time_range(df: pd.DataFrame, t_start, t_end) -> pd.DataFrame:
    """Selecteer rijen binnen [t_start, t_end] zonder pandas boolean-indexing bugs."""
    if df.empty:
        return df.copy()
    ns = df["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    a = _ts_ns(t_start)
    b = _ts_ns(t_end)
    if a > b:
        a, b = b, a
    idxs = np.flatnonzero((ns >= a) & (ns <= b)).astype(np.intp)
    if idxs.size == 0:
        return df.iloc[0:0].copy()
    return pd.DataFrame({col: df[col].to_numpy().take(idxs) for col in df.columns})


def _nearest_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str,
    max_delta_seconds: float,
) -> pd.DataFrame:
    """Nearest-time join without pandas.merge_asof (crashes on Python 3.14)."""
    if left.empty or right.empty:
        return left.iloc[0:0].copy()

    left = left.sort_values(on).reset_index(drop=True)
    right = right.sort_values(on).reset_index(drop=True)

    left_ns = left[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    right_ns = right[on].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    tol_ns = int(float(max_delta_seconds) * 1_000_000_000)

    idx = np.searchsorted(right_ns, left_ns, side="left")
    idx_lo = np.clip(idx - 1, 0, len(right_ns) - 1)
    idx_hi = np.clip(idx, 0, len(right_ns) - 1)
    delta_lo = np.abs(left_ns - right_ns[idx_lo])
    delta_hi = np.abs(left_ns - right_ns[idx_hi])
    use_hi = delta_hi < delta_lo
    best = np.where(use_hi, idx_hi, idx_lo)
    best_delta = np.where(use_hi, delta_hi, delta_lo)
    matched = best_delta <= tol_ns

    if not matched.any():
        return left.iloc[0:0].copy()

    left_idx = np.flatnonzero(matched).astype(np.intp)
    best_idx = best[matched].astype(np.intp)
    out = pd.DataFrame({col: left[col].to_numpy().take(left_idx) for col in left.columns})
    for col in right.columns:
        if col == on:
            continue
        out[col] = right[col].to_numpy().take(best_idx)
    return out


def merge_measurements(
    first: pd.DataFrame,
    second: pd.DataFrame,
    second_offset_seconds: float = 0.0,
    max_delta_seconds: float = 30.0,
) -> pd.DataFrame:
    """Koppel elk punt van bestand 1 aan dichtstbijzijnde punt van bestand 2 (RH + T)."""
    left = first[["tijd", "t", "rh"]].copy().sort_values("tijd")
    left = left.rename(columns={"t": "t_1", "rh": "rh_1"})

    right = second[["tijd", "t", "rh"]].copy().sort_values("tijd")
    right["tijd"] = _shift_datetime_series(right["tijd"], second_offset_seconds)
    right = right.rename(columns={"t": "t_2", "rh": "rh_2"})

    merged = _nearest_join(
        left,
        right,
        on="tijd",
        max_delta_seconds=max_delta_seconds,
    )
    if merged.empty:
        return merged
    merged["d_rh"] = merged["rh_1"] - merged["rh_2"]
    merged["d_t"] = merged["t_1"] - merged["t_2"]
    return merged.reset_index(drop=True)


def nearest_setpoint(rh: float, tolerance: float) -> float | None:
    if pd.isna(rh):
        return None
    idx = int(np.argmin(np.abs(TYPICAL_SETPOINTS - rh)))
    sp = float(TYPICAL_SETPOINTS[idx])
    if abs(rh - sp) <= tolerance:
        return sp
    return None


def detect_stable_blocks(
    merged: pd.DataFrame,
    rh_tolerance: float = 1.0,
    settle_minutes: float = 5.0,
    min_stable_minutes: float = 5.0,
    end_margin_minutes: float = 2.0,
) -> pd.DataFrame:
    """Vind RH-plateaus op bestand 1 en gemiddel RH/T in stabiele vensters.

    Stabiel venster:
    - start na ``settle_minutes`` (waarde is dan ingeregeld / stabiel)
    - eindigt uiterlijk ``end_margin_minutes`` vóór de start van het volgende
      RH-setpoint (klokken lopen niet synchroon)
    """
    if merged.empty or "rh_1" not in merged.columns:
        return pd.DataFrame()

    work = merged.sort_values("tijd").reset_index(drop=True)
    assigned = [nearest_setpoint(v, rh_tolerance) for v in work["rh_1"]]
    work["setpoint"] = pd.Series(assigned, dtype="object")

    # Ruwe plateaus: opeenvolgende rijen met hetzelfde setpoint
    plateaus: list[tuple[int, int]] = []
    start = 0
    n = len(work)
    for i in range(1, n + 1):
        if i == n or work.at[i, "setpoint"] != work.at[start, "setpoint"]:
            plateaus.append((start, i))
            start = i

    blocks: list[dict] = []
    settle_ns = int(float(settle_minutes) * 60 * 1_000_000_000)
    end_margin_ns = int(float(end_margin_minutes) * 60 * 1_000_000_000)

    for p_idx, (i0, i1) in enumerate(plateaus):
        sp = work.at[i0, "setpoint"]
        if sp is None or (isinstance(sp, float) and np.isnan(sp)):
            continue
        sp = float(sp)
        block = work.iloc[i0:i1].reset_index(drop=True)
        t0 = block["tijd"].iloc[0]
        t1 = block["tijd"].iloc[-1]
        t0_ns = _ts_ns(t0)
        t1_ns = _ts_ns(t1)
        duration_min = (t1_ns - t0_ns) / 1_000_000_000 / 60.0

        # Start pas als stabiel (na inregeltijd)
        stable_start_ns = t0_ns + settle_ns
        # Eindig uiterlijk 2 min vóór start van het volgende RH-setpoint-blok
        stable_end_ns = t1_ns
        next_start = None
        for q in range(p_idx + 1, len(plateaus)):
            j0, _j1 = plateaus[q]
            next_sp = work.at[j0, "setpoint"]
            if next_sp is None or (isinstance(next_sp, float) and np.isnan(next_sp)):
                continue
            next_start = work.at[j0, "tijd"]
            next_start_ns = _ts_ns(next_start)
            stable_end_ns = min(stable_end_ns, next_start_ns - end_margin_ns)
            break

        if stable_end_ns <= stable_start_ns:
            continue

        tijd_ns = block["tijd"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        idxs = np.flatnonzero(
            (tijd_ns >= stable_start_ns) & (tijd_ns <= stable_end_ns)
        ).astype(np.intp)
        if idxs.size == 0:
            continue

        stable = pd.DataFrame({col: block[col].to_numpy().take(idxs) for col in block.columns})
        s0_ns = _ts_ns(stable["tijd"].iloc[0])
        s1_ns = _ts_ns(stable["tijd"].iloc[-1])
        stable_min = (s1_ns - s0_ns) / 1_000_000_000 / 60.0
        if len(stable) < 3 or stable_min < min_stable_minutes * 0.5:
            continue

        prev_sp = blocks[-1]["setpoint"] if blocks else None
        if prev_sp is None:
            richting = "start"
        elif sp > prev_sp:
            richting = "opwaarts"
        elif sp < prev_sp:
            richting = "neerwaarts"
        else:
            richting = "herhaal"

        def mean(col: str) -> float:
            return float(stable[col].mean())

        def std(col: str) -> float:
            return float(stable[col].std(ddof=1)) if len(stable) > 1 else 0.0

        blocks.append(
            {
                "setpoint": sp,
                "richting": richting,
                "blok_start": t0,
                "blok_einde": t1,
                "volgende_blok": next_start,
                "stabiel_van": stable["tijd"].iloc[0],
                "stabiel_tot": stable["tijd"].iloc[-1],
                "blokduur_min": round(duration_min, 1),
                "stabiel_min": round(stable_min, 1),
                "n": int(len(stable)),
                "t_1": mean("t_1"),
                "rh_1": mean("rh_1"),
                "t_2": mean("t_2"),
                "rh_2": mean("rh_2"),
                "d_rh": mean("d_rh"),
                "d_t": mean("d_t"),
                "std_rh_1": std("rh_1"),
                "std_t_1": std("t_1"),
                "std_d_rh": std("d_rh"),
                "std_d_t": std("d_t"),
            }
        )

    return pd.DataFrame(blocks)


DISPLAY_COLUMNS = [
    ("tijd", "Tijd"),
    ("t_1", "T bestand 1 [°C]"),
    ("rh_1", "RH bestand 1 [%]"),
    ("t_2", "T bestand 2 [°C]"),
    ("rh_2", "RH bestand 2 [%]"),
    ("d_rh", "ΔRH (1−2) [%]"),
    ("d_t", "ΔT (1−2) [°C]"),
]

BLOCK_COLUMNS = [
    ("setpoint", "RH set [%]"),
    ("richting", "Richting"),
    ("stabiel_van", "Stabiel van"),
    ("stabiel_tot", "Stabiel tot"),
    ("volgende_blok", "Volgende setpoint"),
    ("stabiel_min", "Stabiel [min]"),
    ("n", "n"),
    ("rh_1", "RH 1 [%]"),
    ("rh_2", "RH 2 [%]"),
    ("d_rh", "ΔRH [%]"),
    ("t_1", "T 1 [°C]"),
    ("t_2", "T 2 [°C]"),
    ("d_t", "ΔT [°C]"),
    ("std_rh_1", "Std RH 1 [%]"),
    ("std_d_rh", "Std ΔRH [%]"),
    ("std_d_t", "Std ΔT [°C]"),
]

DIFF_COLUMNS = [
    ("setpoint", "RH set [%]"),
    ("richting", "Richting"),
    ("stabiel_van", "Stabiel van"),
    ("stabiel_tot", "Stabiel tot"),
    ("n", "n"),
    ("rh_1", "RH 1 [%]"),
    ("rh_2", "RH 2 [%]"),
    ("d_rh", "ΔRH (1−2) [%]"),
    ("std_d_rh", "Std ΔRH [%]"),
    ("t_1", "T 1 [°C]"),
    ("t_2", "T 2 [°C]"),
    ("d_t", "ΔT (1−2) [°C]"),
    ("std_d_t", "Std ΔT [°C]"),
]

SINGLE_DISPLAY_COLUMNS = [
    ("tijd", "Tijd"),
    ("t", "T [°C]"),
    ("rh", "RH [%]"),
]


def format_for_display(df: pd.DataFrame, columns: list[tuple[str, str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[label for _, label in columns])
    out = pd.DataFrame()
    for src, label in columns:
        if src not in df.columns:
            continue
        series = df[src]
        if src == "setpoint":
            out[label] = series.round(0).astype("Int64")
        else:
            out[label] = series
    return out

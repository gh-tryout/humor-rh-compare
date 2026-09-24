"""Streamlit-app: vergelijk RH en temperatuur van twee meetbestanden."""

from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from rh_compare import (
    BLOCK_COLUMNS,
    DIFF_COLUMNS,
    DISPLAY_COLUMNS,
    SINGLE_DISPLAY_COLUMNS,
    detect_stable_blocks,
    filter_time_range,
    format_for_display,
    list_data_files,
    load_instrument_bytes,
    load_instrument_file,
    merge_measurements,
    shift_datetime_series,
)

st.set_page_config(
    page_title="Humor 20 vs dauwpuntsmeter",
    page_icon="💧",
    layout="wide",
)

st.title("Humor 20 vs dauwpuntsmeter")
st.caption(
    "Website: upload meetbestanden, kies een tijdspanne, vergelijk RH en temperatuur. "
    "Ondersteund: Humor CSV, dauwpuntsmeter TXT, sensor-export CSV."
)


def default_data_folder() -> str:
    env = os.environ.get("HUMOR_RH_DATA_DIR")
    if env:
        return env
    if getattr(sys, "frozen", False):
        return str(Path(sys.executable).resolve().parent)
    return str(Path(__file__).resolve().parent)


def _column_config(df: pd.DataFrame) -> dict:
    config: dict = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            config[col] = st.column_config.DatetimeColumn(col, format="YYYY-MM-DD HH:mm:ss", width="medium")
        elif pd.api.types.is_integer_dtype(series):
            config[col] = st.column_config.NumberColumn(col, format="%d", width="small")
        elif pd.api.types.is_float_dtype(series):
            config[col] = st.column_config.NumberColumn(col, format="%.3f", width="small")
        else:
            config[col] = st.column_config.TextColumn(col, width="small")
    return config


def _to_py_dt(value) -> datetime:
    ts = pd.Timestamp(value)
    return ts.to_pydatetime().replace(tzinfo=None)


def _df_to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>Geen gegevens</em></p>"
    return df.to_html(index=False, border=0, classes="data", escape=True)


def build_report_html(
    *,
    path1_name: str,
    path2_name: str,
    kind1: str,
    kind2: str,
    range_start,
    range_end,
    settle_min: float,
    end_margin: float,
    fig_main: go.Figure,
    fig_diff: go.Figure,
    fig_blocks: go.Figure | None,
    merged_display: pd.DataFrame,
    diff_display: pd.DataFrame,
    block_display: pd.DataFrame,
    summary_display: pd.DataFrame,
) -> str:
    """Bouw één HTML-rapport met interactieve Plotly-grafieken en tabellen."""
    chart_main = fig_main.to_html(full_html=False, include_plotlyjs="cdn")
    chart_diff = fig_diff.to_html(full_html=False, include_plotlyjs=False)
    chart_blocks = (
        fig_blocks.to_html(full_html=False, include_plotlyjs=False) if fig_blocks is not None else ""
    )

    blocks_section = ""
    if not diff_display.empty:
        blocks_section = f"""
        <h2>Stabiele verschillen</h2>
        <p>Start na {settle_min:g} min inregeltijd; einde uiterlijk {end_margin:g} min
        vóór het volgende setpoint.</p>
        <h3>Tabel verschillen (gestabiliseerd)</h3>
        {_df_to_html_table(diff_display)}
        <h3>Details per blok</h3>
        {_df_to_html_table(block_display)}
        <h3>Samenvatting per setpoint</h3>
        {_df_to_html_table(summary_display)}
        <h3>ΔRH per setpoint</h3>
        {chart_blocks}
        """
    else:
        blocks_section = "<h2>Stabiele verschillen</h2><p><em>Geen stabiele blokken gevonden.</em></p>"

    return f"""<!DOCTYPE html>
<html lang="nl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Humor RH vergelijking</title>
  <style>
    body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
    h3 {{ margin-top: 1.4rem; }}
    .meta {{ color: #444; margin-bottom: 1.5rem; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.5rem 0 1.5rem; }}
    table.data th, table.data td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    table.data th {{ background: #f3f3f3; }}
    table.data tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Humor 20 vs dauwpuntsmeter</h1>
  <div class="meta">
    <div><strong>Bestand 1:</strong> {html.escape(path1_name)} ({html.escape(kind1)})</div>
    <div><strong>Bestand 2:</strong> {html.escape(path2_name)} ({html.escape(kind2)})</div>
    <div><strong>Tijdspanne:</strong> {html.escape(str(range_start))} → {html.escape(str(range_end))}</div>
    <div><strong>Gegenereerd:</strong> {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </div>

  <h2>RH &amp; temperatuur</h2>
  {chart_main}

  <h2>Verschillen (bestand 1 − bestand 2)</h2>
  {chart_diff}

  <h2>Gekoppelde tabel</h2>
  {_df_to_html_table(merged_display)}

  {blocks_section}
</body>
</html>
"""


@st.cache_data(show_spinner="Bestand laden…")
def cached_load(path_str: str, mtime: float):
    kind, df = load_instrument_file(path_str)
    return kind, df


@st.cache_data(show_spinner="Bestand laden…")
def cached_load_bytes(filename: str, data: bytes):
    return load_instrument_bytes(filename, data)


def _sync_selection_order(checked_names: list[str]) -> list[str]:
    order: list[str] = list(st.session_state.get("selection_order", []))
    order = [name for name in order if name in checked_names]
    for name in checked_names:
        if name not in order:
            order.append(name)
    st.session_state.selection_order = order
    return order


def _clear_upload_keys() -> None:
    for key in ("upload_file_1", "upload_file_2"):
        if key in st.session_state:
            del st.session_state[key]


with st.sidebar:
    st.header("Bestanden")
    on_spaces = bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))
    source_mode = st.radio(
        "Bron",
        options=["Uploaden"] if on_spaces else ["Uploaden", "Lokale map"],
        index=0,
        help="Op Hugging Face Spaces: bestanden uploaden. Lokaal kun je ook een map kiezen.",
    )

    files: list[Path] = []
    folder_path = Path(default_data_folder())

    if st.session_state.pop("reset_selection", False):
        st.session_state["selection_order"] = []
        _clear_upload_keys()
        for path in list_data_files(folder_path) if folder_path.is_dir() else []:
            st.session_state[f"file_{path.name}"] = False

    if st.button("Opnieuw beginnen", help="Wis uploads/selectie en start opnieuw."):
        st.session_state["reset_selection"] = True
        st.rerun()

    name1 = ""
    name2 = ""
    data1: bytes | None = None
    data2: bytes | None = None
    selected_paths: list[Path] = []

    if source_mode == "Uploaden":
        st.caption("Upload eerst bestand 1, kies een tijdspanne, daarna bestand 2.")
        up1 = st.file_uploader(
            "Bestand 1",
            type=["csv", "txt", "dat"],
            key="upload_file_1",
        )
        up2 = st.file_uploader(
            "Bestand 2",
            type=["csv", "txt", "dat"],
            key="upload_file_2",
        )
        if up1 is not None:
            name1 = up1.name
            data1 = up1.getvalue()
        if up2 is not None:
            name2 = up2.name
            data2 = up2.getvalue()
    else:
        folder = st.text_input(
            "Datamap",
            value=default_data_folder(),
            help="Pad naar de map met meetbestanden.",
        )
        folder_path = Path(folder)
        if not folder_path.is_dir():
            st.error("Deze map bestaat niet.")
        else:
            files = list_data_files(folder_path)
            if not files:
                st.warning("Geen .csv of .txt bestanden gevonden.")

        st.caption("Vink eerst één bestand aan, kies een tijdspanne, daarna een tweede.")
        checked: list[str] = []
        for path in files:
            if st.checkbox(path.name, key=f"file_{path.name}"):
                checked.append(path.name)
        selection_order = _sync_selection_order(checked)
        selected_paths = [folder_path / name for name in selection_order]

    st.divider()
    st.header("Koppeling")
    offset_s = st.number_input(
        "Tijdoffset bestand 2 [s]",
        value=0.0,
        step=1.0,
        help="Tel deze offset op bij de tijden van bestand 2 als de klokken verschillen.",
    )
    max_delta_s = st.number_input(
        "Max. tijdsverschil [s]",
        value=30.0,
        min_value=1.0,
        step=5.0,
        help="Elk punt van bestand 1 wordt gekoppeld aan het dichtstbijzijnde punt van bestand 2.",
    )

    st.divider()
    st.header("Stabiele blokken")
    rh_tol = st.number_input("RH-tolerantie setpoint [%]", value=1.0, min_value=0.2, step=0.1)
    settle_min = st.number_input(
        "Inregeltijd overslaan [min]",
        value=5.0,
        min_value=0.0,
        step=1.0,
        help="Stabiel venster begint pas na deze inregeltijd (waarde is dan stabiel).",
    )
    end_margin = st.number_input(
        "Einde vóór volgende setpoint [min]",
        value=2.0,
        min_value=0.0,
        step=0.5,
        help="Stabiel venster eindigt uiterlijk zoveel minuten vóór het volgende RH-setpoint "
        "(compensatie voor niet-synchrone klokken).",
    )
    min_stable = st.number_input("Minimale stabiele duur [min]", value=5.0, min_value=1.0, step=1.0)


# --- Bestand 1 ---
if source_mode == "Uploaden":
    if data1 is None:
        st.info("Upload in de zijbalk het eerste meetbestand om te beginnen.")
        st.stop()
    try:
        kind1, df1 = cached_load_bytes(name1, data1)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{name1}: {exc}")
        st.stop()
    label1_name = name1
else:
    if not selected_paths:
        st.info("Selecteer in de zijbalk een datamap en vink het eerste meetbestand aan.")
        st.stop()
    if len(selected_paths) > 2:
        st.warning("Maximaal twee bestanden. Alleen de eerste twee worden gebruikt.")
        selected_paths = selected_paths[:2]
    path1 = selected_paths[0]
    try:
        kind1, df1 = cached_load(str(path1), path1.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{path1.name}: {exc}")
        st.stop()
    label1_name = path1.name

if df1.empty:
    st.error(f"{label1_name} bevat geen bruikbare RH/T-metingen.")
    st.stop()

label1 = f"{label1_name} ({kind1})"
st.subheader(f"1. {label1}")
st.caption(
    f"{df1['tijd'].iloc[0]} → {df1['tijd'].iloc[-1]} · {len(df1)} punten"
)

t_min = _to_py_dt(df1["tijd"].iloc[0])
t_max = _to_py_dt(df1["tijd"].iloc[-1])
if t_min == t_max:
    st.warning("Dit bestand heeft maar één tijdstip; tijdspanne kan niet worden begrensd.")
    range_start, range_end = t_min, t_max
else:
    range_start, range_end = st.slider(
        "Tijdspanne (op basis van bestand 1)",
        min_value=t_min,
        max_value=t_max,
        value=(t_min, t_max),
        format="YYYY-MM-DD HH:mm:ss",
        help="Alleen data binnen dit venster wordt gebruikt voor de vergelijking.",
    )

df1_win = filter_time_range(df1, range_start, range_end)
if df1_win.empty:
    st.error("Geen punten van bestand 1 in de gekozen tijdspanne.")
    st.stop()

tab_preview, tab_table1 = st.tabs(["Voorbeeld bestand 1", "Tabel bestand 1"])
with tab_preview:
    fig1 = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Relatieve vochtigheid", "Temperatuur"),
    )
    fig1.add_trace(
        go.Scatter(x=df1_win["tijd"], y=df1_win["rh"], name="RH 1", mode="lines+markers"),
        row=1,
        col=1,
    )
    fig1.add_trace(
        go.Scatter(x=df1_win["tijd"], y=df1_win["t"], name="T 1", mode="lines+markers"),
        row=2,
        col=1,
    )
    fig1.update_yaxes(title_text="RH [%]", row=1, col=1)
    fig1.update_yaxes(title_text="T [°C]", row=2, col=1)
    fig1.update_xaxes(title_text="Tijd", row=2, col=1)
    fig1.update_layout(height=520, legend=dict(orientation="h", y=1.02), hovermode="x unified")
    st.plotly_chart(fig1, width="stretch")

with tab_table1:
    display1 = format_for_display(df1_win, SINGLE_DISPLAY_COLUMNS)
    st.dataframe(
        display1,
        width="stretch",
        hide_index=True,
        height=420,
        column_config=_column_config(display1),
    )

# --- Bestand 2 ---
if source_mode == "Uploaden":
    if data2 is None:
        st.info("Upload nu een tweede meetbestand in de zijbalk om te vergelijken.")
        st.stop()
    try:
        kind2, df2 = cached_load_bytes(name2, data2)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{name2}: {exc}")
        st.stop()
    label2_name = name2
else:
    if len(selected_paths) < 2:
        st.info("Vink nu een tweede meetbestand aan om te vergelijken binnen de gekozen tijdspanne.")
        st.stop()
    path2 = selected_paths[1]
    try:
        kind2, df2 = cached_load(str(path2), path2.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{path2.name}: {exc}")
        st.stop()
    label2_name = path2.name

label2 = f"{label2_name} ({kind2})"
st.divider()
st.subheader(f"2. Vergelijking met {label2}")

if kind1 == kind2:
    st.warning(
        f"Beide bestanden zijn herkend als `{kind1}`. De vergelijking gaat door, "
        "maar controleer of dit de bedoelde bestanden zijn."
    )

df2_work = df2.copy()
df2_work["tijd"] = shift_datetime_series(df2_work["tijd"], float(offset_s))
df2_in_range = filter_time_range(df2_work, range_start, range_end)
if df2_in_range.empty:
    st.error(
        "Geen punten van bestand 2 in de gekozen tijdspanne. "
        "Pas de tijdspanne of de tijdoffset aan."
    )
    st.stop()

try:
    merged = merge_measurements(
        df1_win,
        df2_in_range,
        second_offset_seconds=0.0,
        max_delta_seconds=float(max_delta_s),
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Koppelen van de metingen is mislukt: {exc}")
    st.stop()

if merged.empty:
    st.error(
        "Geen overlappende tijdstippen binnen de max. tijdsverschil-instelling. "
        "Vergroot het max. tijdsverschil of pas de offset aan."
    )
    st.stop()

blocks = detect_stable_blocks(
    merged,
    rh_tolerance=float(rh_tol),
    settle_minutes=float(settle_min),
    min_stable_minutes=float(min_stable),
    end_margin_minutes=float(end_margin),
)

info1, info2, info3, info4 = st.columns(4)
info1.metric("Bestand 1", label1_name)
info2.metric("Bestand 2", label2_name)
info3.metric("Gekoppelde punten", f"{len(merged)}")
info4.metric("Stabiele RH-blokken", f"{len(blocks)}")

st.caption(
    f"Tijdspanne: {range_start} → {range_end} · "
    f"Bestand 2 in venster: {len(df2_in_range)} punten"
)


def add_block_shading(fig, block_df: pd.DataFrame, row: int) -> None:
    colors = ["rgba(31,119,180,0.08)", "rgba(255,127,14,0.10)"]
    for i, rec in enumerate(block_df.itertuples()):
        fig.add_vrect(
            x0=rec.stabiel_van,
            x1=rec.stabiel_tot,
            fillcolor=colors[i % 2],
            line_width=0,
            row=row,
            col=1,
            annotation_text=f"{rec.setpoint:.0f} %",
            annotation_position="top left",
            annotation_font_size=10,
        )


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Relatieve vochtigheid", "Temperatuur"),
)
fig.add_trace(
    go.Scatter(x=merged["tijd"], y=merged["rh_1"], name=f"RH {label1_name}", mode="lines+markers"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=merged["tijd"], y=merged["rh_2"], name=f"RH {label2_name}", mode="lines+markers"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=merged["tijd"], y=merged["t_1"], name=f"T {label1_name}", mode="lines+markers"),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=merged["tijd"], y=merged["t_2"], name=f"T {label2_name}", mode="lines+markers"),
    row=2,
    col=1,
)
if not blocks.empty:
    add_block_shading(fig, blocks, row=1)

fig.update_yaxes(title_text="RH [%]", row=1, col=1)
fig.update_yaxes(title_text="T [°C]", row=2, col=1)
fig.update_xaxes(title_text="Tijd", row=2, col=1)
fig.update_layout(
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    hovermode="x unified",
    margin=dict(t=60, b=40),
)

fig_diff = make_subplots(
    rows=1,
    cols=2,
    shared_xaxes=True,
    subplot_titles=("ΔRH", "ΔT"),
)
fig_diff.add_trace(go.Scatter(x=merged["tijd"], y=merged["d_rh"], name="ΔRH", mode="lines"), row=1, col=1)
fig_diff.add_trace(go.Scatter(x=merged["tijd"], y=merged["d_t"], name="ΔT", mode="lines"), row=1, col=2)
fig_diff.update_yaxes(title_text="[%]", row=1, col=1)
fig_diff.update_yaxes(title_text="[°C]", row=1, col=2)
fig_diff.update_layout(height=360, showlegend=False, hovermode="x unified")

display_df = format_for_display(merged, DISPLAY_COLUMNS)
diff_display = format_for_display(blocks, DIFF_COLUMNS) if not blocks.empty else pd.DataFrame()
block_display = format_for_display(blocks, BLOCK_COLUMNS) if not blocks.empty else pd.DataFrame()
summary_display = pd.DataFrame()
fig_blocks: go.Figure | None = None
if not blocks.empty:
    summary = (
        blocks.groupby(["setpoint", "richting"], as_index=False)[
            ["rh_1", "rh_2", "t_1", "t_2", "d_rh", "d_t"]
        ]
        .mean(numeric_only=True)
        .sort_values(["setpoint", "richting"])
    )
    summary_cols = [
        ("setpoint", "RH set [%]"),
        ("richting", "Richting"),
        ("rh_1", "RH 1 [%]"),
        ("rh_2", "RH 2 [%]"),
        ("d_rh", "ΔRH [%]"),
        ("t_1", "T 1 [°C]"),
        ("t_2", "T 2 [°C]"),
        ("d_t", "ΔT [°C]"),
    ]
    summary_display = format_for_display(summary, summary_cols)
    fig_blocks = go.Figure()
    fig_blocks.add_trace(
        go.Scatter(
            x=blocks["setpoint"],
            y=blocks["d_rh"],
            mode="markers+lines",
            name="ΔRH",
            text=blocks["richting"],
            hovertemplate="RH %{x:.0f} % (%{text})<br>ΔRH %{y:.3f} %<extra></extra>",
        )
    )
    fig_blocks.add_hline(y=0, line_width=1, line_color="gray")
    fig_blocks.update_layout(
        xaxis_title="Ingesteld RH [%]",
        yaxis_title="Gem. ΔRH (bestand 1 − 2) [%]",
        height=380,
    )

report_html = build_report_html(
    path1_name=label1_name,
    path2_name=label2_name,
    kind1=kind1,
    kind2=kind2,
    range_start=range_start,
    range_end=range_end,
    settle_min=float(settle_min),
    end_margin=float(end_margin),
    fig_main=fig,
    fig_diff=fig_diff,
    fig_blocks=fig_blocks,
    merged_display=display_df,
    diff_display=diff_display,
    block_display=block_display,
    summary_display=summary_display,
)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
st.download_button(
    "Download grafieken & tabellen (HTML)",
    data=report_html.encode("utf-8"),
    file_name=f"vergelijking_rh_t_{stamp}.html",
    mime="text/html",
    help="Eén HTML-bestand met interactieve grafieken en alle tabellen.",
)

tab_grafiek, tab_tabel, tab_blokken = st.tabs(
    ["RH & temperatuur", "Gekoppelde tabel", "Stabiele verschillen"]
)

with tab_grafiek:
    st.plotly_chart(fig, width="stretch")
    st.subheader("Verschillen (bestand 1 − bestand 2)")
    st.plotly_chart(fig_diff, width="stretch")

with tab_tabel:
    st.markdown(
        "Elk punt van **bestand 1** is gekoppeld aan het dichtstbijzijnde punt van **bestand 2** "
        "binnen de gekozen tijdspanne."
    )
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=520,
        column_config=_column_config(display_df),
    )
    st.download_button(
        "Download gekoppelde tabel (CSV)",
        data=merged.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="vergelijking_rh_t.csv",
        mime="text/csv",
    )

with tab_blokken:
    st.markdown(
        "Verschillen (**ΔRH**, **ΔT**) over het **stabiele** deel van elk RH-setpoint van bestand 1:\n"
        f"- start na **{settle_min:g} min** inregeltijd\n"
        f"- eindigt uiterlijk **{end_margin:g} min** vóór het volgende setpoint "
        "(klokken niet synchroon)"
    )
    if blocks.empty:
        st.warning(
            "Geen stabiele blokken gevonden. Verlaag de inregeltijd, de eindmarge, "
            "of vergroot de RH-tolerantie."
        )
    else:
        st.subheader("Tabel verschillen (gestabiliseerd)")
        st.dataframe(
            diff_display,
            width="stretch",
            hide_index=True,
            column_config=_column_config(diff_display),
        )
        st.download_button(
            "Download verschillentabel (CSV)",
            data=blocks[list(dict.fromkeys(c for c, _ in DIFF_COLUMNS if c in blocks.columns))]
            .to_csv(index=False, sep=";", decimal=",")
            .encode("utf-8-sig"),
            file_name="stabiele_verschillen_rh_t.csv",
            mime="text/csv",
        )

        st.subheader("Details per blok")
        st.dataframe(
            block_display,
            width="stretch",
            hide_index=True,
            column_config=_column_config(block_display),
        )

        st.subheader("Samenvatting per setpoint")
        st.dataframe(
            summary_display,
            width="stretch",
            hide_index=True,
            column_config=_column_config(summary_display),
        )

        st.plotly_chart(fig_blocks, width="stretch")

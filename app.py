from __future__ import annotations

import io
import random
import re
import unicodedata
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


NASA_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name%2Cpl_eqt%2Cpl_rade%2Cpl_insol%2Cra%2Cdec%2Csy_dist%2Chostname%2C"
    "discoverymethod%2Cdisc_year+from+pscomppars+where+"
    "pl_eqt+is+not+null+and+pl_rade+is+not+null&format=csv"
)
EARTH_EQUILIBRIUM_TEMPERATURE_K = 255
LIQUID_WATER_OPTIMAL_TEMPERATURE_K = 288
LIQUID_WATER_TEMPERATURE_TOLERANCE_K = 88
TERRESTRIAL_RADIUS_RANGE_EARTH = (0.8, 1.6)
PSYCHOPHILE_UPPER_BOUND_K = 288
MESOPHILE_UPPER_BOUND_K = 318
PARSEC_TO_LIGHT_YEAR = 3.26156
LIGHT_SPEED_KM_PER_HOUR = 299_792.458 * 3600
VOYAGER_1_SPEED_KM_PER_HOUR = 61_200
JWST_BACKGROUNDS = (
    (
        "Herbig–Haro 46/47 — NIRCam",
        "https://cdn.esawebb.org/archives/images/large/weic2319a.jpg",
    ),
    (
        "Webb’s First Deep Field",
        "https://images-assets.nasa.gov/image/webb_first_deep_field/webb_first_deep_field~large.jpg",
    ),
    (
        "Carina Nebula — Cosmic Cliffs",
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/5/59/NASA%E2%80%99s_Webb_Reveals_Cosmic_Cliffs%2C_Glittering_Landscape_of_Star_Birth_-_Flickr_-_James_Webb_Space_Telescope.png/1920px-NASA%E2%80%99s_Webb_Reveals_Cosmic_Cliffs%2C_Glittering_Landscape_of_Star_Birth_-_Flickr_-_James_Webb_Space_Telescope.png",
    ),
    (
        "Southern Ring Nebula",
        "https://images-assets.nasa.gov/image/southern_ring_nebula/southern_ring_nebula~large.jpg",
    ),
)


def classify_temperature(temperature_k: float) -> str:
    """Assign a broad temperature-regime proxy, not a biological diagnosis."""
    if temperature_k < PSYCHOPHILE_UPPER_BOUND_K:
        return "Psychophile"
    if temperature_k <= MESOPHILE_UPPER_BOUND_K:
        return "Mesophile"
    return "Thermophile"


def liquid_water_probability(temperature_k: float, radius_earth: float) -> float:
    """Return a simple temperature-and-size screening score, not a true probability."""
    in_terrestrial_radius_zone = (
        TERRESTRIAL_RADIUS_RANGE_EARTH[0]
        <= radius_earth
        <= TERRESTRIAL_RADIUS_RANGE_EARTH[1]
    )
    if not in_terrestrial_radius_zone:
        return 0.0
    temperature_score = 1 - (
        abs(temperature_k - LIQUID_WATER_OPTIMAL_TEMPERATURE_K)
        / LIQUID_WATER_TEMPERATURE_TOLERANCE_K
    )
    return max(0.0, min(100.0, temperature_score * 100))


def classify_stellar_flux(flux_earth: float | None) -> str:
    """Describe bolometric irradiation; it does not measure radiation dose."""
    if pd.isna(flux_earth):
        return "Unavailable"
    if flux_earth < 0.5:
        return "Low insolation"
    if flux_earth <= 1.5:
        return "Near-Earth insolation"
    return "Elevated insolation"


def calculate_travel_times(distance_parsecs: float) -> tuple[float, float, float]:
    """Return distance in light-years and idealized travel times at c and Voyager speed."""
    distance_light_years = distance_parsecs * PARSEC_TO_LIGHT_YEAR
    voyager_speed_fraction = VOYAGER_1_SPEED_KM_PER_HOUR / LIGHT_SPEED_KM_PER_HOUR
    return (
        distance_light_years,
        distance_light_years,
        distance_light_years / voyager_speed_fraction,
    )


def format_years(years: float) -> str:
    if years >= 1_000_000_000:
        return f"{years / 1_000_000_000:,.2f} billion years"
    if years >= 1_000_000:
        return f"{years / 1_000_000:,.2f} million years"
    return f"{years:,.0f} years"


@st.cache_data(ttl=3600, show_spinner="Loading exoplanet data...")
def load_exoplanets() -> pd.DataFrame:
    """Load planets with measured equilibrium temperature and radius."""
    data = pd.read_csv(NASA_EXOPLANET_ARCHIVE_URL)
    for column in ("pl_eqt", "pl_rade", "pl_insol", "ra", "dec", "sy_dist"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["pl_eqt", "pl_rade"])


def generate_pdf_report(
    planet_name: str,
    equilibrium_temperature: float,
    radius_earth: float,
    thermal_regime: str,
    water_score: float,
    radiation_status: str,
    right_ascension: float | None,
    declination: float | None,
    stellar_flux: float | None,
) -> bytes:
    """Build a concise PDF briefing from measured values and derived indicators."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AssessmentTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor("#1f4e79"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "AssessmentSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#526477"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "AssessmentSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=8,
        spaceAfter=7,
    )
    note_style = ParagraphStyle(
        "AssessmentNote",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#384554"),
    )
    table_header_style = ParagraphStyle(
        "AssessmentTableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1f4e79"),
    )
    table_cell_style = ParagraphStyle(
        "AssessmentTableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#263442"),
    )

    def safe_text(value: object) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return escape(ascii_text, quote=False)

    def coordinate_text(value: float | None) -> str:
        return f"{repr(float(value))} deg" if value is not None and pd.notna(value) else "Unavailable"

    measured_rows = [
        ["Parameter", "Value", "Basis"],
        ["Target", safe_text(planet_name), "NASA Exoplanet Archive"],
        ["Right ascension (RA)", coordinate_text(right_ascension), "Archive coordinate (degrees)"],
        ["Declination (Dec)", coordinate_text(declination), "Archive coordinate (degrees)"],
        ["Equilibrium temperature", f"{equilibrium_temperature:.1f} K", "Archive composite parameter"],
        ["Planetary radius", f"{radius_earth:.2f} Earth radii", "Archive composite parameter"],
        [
            "Thermal regime",
            safe_text(thermal_regime),
            "Temperature-band classification",
        ],
        [
            "Liquid-water screening score",
            f"{water_score:.1f}%",
            "Temperature and radius heuristic; not a measured probability",
        ],
        [
            "Stellar flux",
            f"{stellar_flux:.4g} Earth fluxes"
            if stellar_flux is not None and pd.notna(stellar_flux)
            else "Unavailable",
            "Archive bolometric flux, when reported",
        ],
        ["Radiation status", safe_text(radiation_status), "Available archive data only"],
    ]
    table_data = [
        [
            Paragraph(safe_text(cell), table_header_style if row_index == 0 else table_cell_style)
            for cell in row
        ]
        for row_index, row in enumerate(measured_rows)
    ]
    measured_table = Table(
        table_data,
        colWidths=[145, 145, 178],
        repeatRows=1,
        hAlign="LEFT",
    )
    measured_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#263442")),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9db2c5")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, colors.HexColor("#d5dee7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ]
        )
    )

    story = [
        Paragraph("PLANETARY ASSESSMENT SHEET", title_style),
        Paragraph(
            f"ASTROBIOLOGY RESEARCH DASHBOARD &nbsp;|&nbsp; {safe_text(planet_name)}",
            subtitle_style,
        ),
        HRFlowable(width="100%", thickness=1.1, color=colors.HexColor("#1f77b4"), spaceAfter=12),
        Paragraph("Measured Parameters and Derived Indicators", section_style),
        measured_table,
        Spacer(1, 13),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8c5d1"), spaceAfter=8),
        Paragraph("Method and limitations", section_style),
        Paragraph(
            "Temperature and radius are NASA Exoplanet Archive composite parameters. "
            "The thermal regime is assigned from the dashboard's stated temperature bands. "
            "The liquid-water screening score is a heuristic derived from equilibrium "
            "temperature and radius; it is not an observational probability. Stellar flux "
            "reports bolometric incident energy when available and is not a radiation-dose "
            "measurement. Coordinates are the archive right ascension and declination in "
            "degrees. Missing measurements are marked unavailable. These values do not "
            "establish surface conditions, radiation exposure, or habitability.",
            note_style,
        ),
    ]
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.set_page_config(page_title="Astrobiology Research Dashboard", page_icon="🪐", layout="wide")
st.session_state.setdefault("roulette_planet", None)
selected_planet_state = st.session_state.get("planet_deep_dive")
previous_planet_state = st.session_state.get("_jwst_background_planet")
if "_jwst_background" not in st.session_state:
    st.session_state["_jwst_background"] = random.choice(JWST_BACKGROUNDS)
elif selected_planet_state != previous_planet_state:
    previous_background = st.session_state["_jwst_background"]
    next_backgrounds = [image for image in JWST_BACKGROUNDS if image != previous_background]
    st.session_state["_jwst_background"] = random.choice(next_backgrounds)
st.session_state["_jwst_background_planet"] = selected_planet_state
background_title, background_url = st.session_state["_jwst_background"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --space-950: #07090e;
        --space-900: #0b1019;
        --space-800: #111923;
        --space-700: #1a2635;
        --ink-100: #edf6ff;
        --ink-300: #b7c5d5;
        --ink-400: #91a5ba;
        --cyan: #72e6ff;
        --violet: #ad9bff;
        --panel: #161b22;
        --line: #30363d;
        --text: #c9d1d9;
    }

    html, body {
        color-scheme: dark;
        color: var(--text);
        background-color: var(--space-950);
    }

    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: transparent !important;
    }

    .stApp {
        color: var(--ink-100);
        background-image:
            linear-gradient(rgba(7, 9, 14, 0.75), rgba(7, 9, 14, 0.75)),
            url("__JWST_BACKGROUND_URL__");
        background-position: center center;
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Manrope', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 9, 14, 0.82);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(13, 20, 31, 0.97), rgba(7, 9, 14, 0.97));
        border-right: 1px solid var(--line);
        box-shadow: 8px 0 30px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: 0.85rem 0.9rem 0.65rem;
        margin: 0.65rem 0;
        border: 1px solid #30363d;
        border-radius: 10px;
        background: #161b22;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #f0f8ff;
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
        text-shadow: 0 0 22px rgba(88, 166, 255, 0.25);
    }

    h1 {
        font-size: clamp(2rem, 3.2vw, 2.8rem) !important;
        font-weight: 800 !important;
    }

    h2 {
        margin-top: 1.6rem !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: var(--ink-300);
    }

    [data-testid="stMetric"] {
        min-width: 0;
        min-height: 140px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #30363d;
        border-radius: 10px;
        background: #161b22;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
        overflow-wrap: break-word;
        word-wrap: break-word;
        white-space: normal;
        transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: #1f77b4;
        box-shadow: 0 0 0 1px rgba(31, 119, 180, 0.2), 0 10px 24px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        min-width: 0;
        max-width: 100%;
        overflow: visible;
        overflow-wrap: break-word;
        word-wrap: break-word;
        white-space: normal !important;
        text-overflow: clip;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--ink-300);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        line-height: 1.45;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-family: 'DM Mono', monospace;
        font-size: clamp(1.1rem, 1.7vw, 1.75rem);
        font-weight: 700;
        line-height: 1.35;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'DM Mono', monospace;
        font-size: 0.76rem;
    }

    [data-testid="stDataFrame"], [data-testid="stTable"], div.stTable {
        overflow: hidden;
        border: 1px solid #30363d;
        border-radius: 10px;
        color: var(--text);
        background: var(--panel);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.1);
        transition: border-color 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stDataFrame"]:hover, [data-testid="stTable"]:hover, div.stTable:hover {
        border-color: rgba(114, 230, 255, 0.52);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.15);
    }

    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataFrame"] canvas,
    [data-testid="stDataFrame"] > div,
    div.stTable table,
    div.stTable th,
    div.stTable td {
        color: var(--text);
        background-color: var(--panel);
        border-color: var(--line);
        color-scheme: dark;
    }

    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] [role="gridcell"],
    div.stTable th,
    div.stTable td {
        color: var(--text);
        border-color: rgba(88, 166, 255, 0.16);
    }

    [data-testid="stDataFrame"] [role="row"]:hover,
    div.stTable tbody tr:hover {
        background-color: rgba(88, 166, 255, 0.09) !important;
    }

    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"],
    [data-testid="stPlotlyChart"],
    [data-testid="stAltairChart"],
    [data-testid="stPyplot"],
    [data-testid="stGraphVizChart"] {
        overflow: hidden;
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 14px;
        background: var(--panel);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.1);
    }

    [data-testid="stVegaLiteChart"] > div,
    [data-testid="stArrowVegaLiteChart"] > div,
    [data-testid="stPlotlyChart"] > div,
    [data-testid="stAltairChart"] > div {
        background-color: var(--panel) !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
        color: var(--text);
        border: 1px solid var(--line);
        background: rgba(22, 27, 34, 0.88);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.03);
        transition: border-color 160ms ease, box-shadow 160ms ease;
    }

    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="textarea"] > div:focus-within {
        border-color: rgba(114, 230, 255, 0.75);
        box-shadow: 0 0 0 1px rgba(114, 230, 255, 0.3), 0 0 16px rgba(88, 166, 255, 0.14);
    }

    [data-baseweb="select"] input,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    [role="option"] {
        color: var(--text);
        background-color: var(--panel);
    }

    [role="option"]:hover, [aria-selected="true"] {
        background-color: #263241 !important;
    }

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"] {
        border-radius: 10px;
    }

    [data-testid="stSelectbox"] > div > div {
        border-color: var(--line);
        background: rgba(22, 27, 34, 0.88);
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid #1f77b4;
        border-radius: 8px;
        color: #ffffff;
        background: #1f77b4;
        font-weight: 700;
        transition: background 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #368ac7;
        color: #ffffff;
        background: #368ac7;
        box-shadow: 0 4px 14px rgba(31, 119, 180, 0.3);
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(88, 166, 255, 0.3);
        border-radius: 12px;
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(88, 166, 255, 0.08);
    }

    hr {
        border-color: rgba(151, 177, 214, 0.16);
    }

    code, [data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.4rem 1rem 3rem;
        }
        [data-testid="stMetric"] {
            min-height: 140px;
            padding: 0.9rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
    }
    </style>
    """.replace("__JWST_BACKGROUND_URL__", background_url),
    unsafe_allow_html=True,
)
st.title("🪐 Astrobiology Research Dashboard")
st.caption(f"🌌 Background image: {background_title} · NASA / ESA / CSA")
st.write(
    "🔭 Screen exoplanets using thermal and size parameters, then inspect each candidate "
    "against a small set of Earth-referenced biological proxies."
)
st.caption(
    "🛰️ Data: NASA Exoplanet Archive — Planetary Systems Composite Parameters. "
    "Equilibrium temperature is not surface temperature; all biological labels are "
    "screening proxies, not evidence of habitability or life."
)

try:
    exoplanets = load_exoplanets()
except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error(f"Could not load data from the NASA Exoplanet Archive: {error}")
    st.info("Check your internet connection and reload the app to try again.")
    st.stop()

st.sidebar.header("🧬 Habitability filters")
temperature_range = st.sidebar.slider(
    "🌡️ Equilibrium temperature (K)",
    min_value=0,
    max_value=2000,
    value=(200, 320),
    step=10,
)
radius_limit = st.sidebar.slider(
    "🪐 Maximum radius (Earth radii, exclusive)",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1,
)

filtered = exoplanets[
    exoplanets["pl_eqt"].between(*temperature_range)
    & (exoplanets["pl_rade"] < radius_limit)
].copy()
filtered["Thermal regime (proxy)"] = filtered["pl_eqt"].map(classify_temperature)
filtered["Liquid water screening score (%)"] = filtered.apply(
    lambda planet: liquid_water_probability(planet["pl_eqt"], planet["pl_rade"]),
    axis=1,
)
if "pl_insol" in filtered.columns:
    filtered["Stellar flux regime (proxy)"] = filtered["pl_insol"].map(classify_stellar_flux)

st.sidebar.divider()
st.sidebar.header("🎲 Surprise Discovery")
if st.sidebar.button(
    "🎲 Roll the Cosmic Dice",
    use_container_width=True,
    disabled=filtered.empty,
):
    st.session_state["roulette_planet"] = random.choice(filtered["pl_name"].tolist())
    st.session_state["planet_deep_dive"] = st.session_state["roulette_planet"]
    st.toast(
        f"🌟 Roulette landed on {st.session_state['roulette_planet']}!",
        icon="🎯",
    )
if filtered.empty:
    st.sidebar.warning("No planets match the current filters to roll.")

st.subheader("📊 Candidate overview")
metric_columns = st.columns(5)
average_temperature = filtered["pl_eqt"].mean()
average_radius = filtered["pl_rade"].mean()
average_water_score = filtered["Liquid water screening score (%)"].mean()
psychophile_share = (
    (filtered["Thermal regime (proxy)"] == "Psychophile").mean() * 100
    if not filtered.empty
    else 0
)
metric_columns[0].metric("🪐 Total matches", f"{len(filtered):,}")
metric_columns[1].metric(
    "🌡️ Mean equilibrium temperature",
    f"{average_temperature:.1f} K" if pd.notna(average_temperature) else "—",
)
metric_columns[2].metric(
    "📏 Mean planetary radius",
    f"{average_radius:.2f} R⊕" if pd.notna(average_radius) else "—",
)
metric_columns[3].metric(
    "💧 Mean water screening score",
    f"{average_water_score:.1f}%" if pd.notna(average_water_score) else "—",
)
metric_columns[4].metric("🧬 Psychophile proxy", f"{psychophile_share:.1f}%")
st.caption(
    f"Active filters: {temperature_range[0]}–{temperature_range[1]} K and "
    f"radius < {radius_limit:.1f} Earth radii. Thermal proxy bands: "
    f"Psychophile < {PSYCHOPHILE_UPPER_BOUND_K} K · "
    f"Mesophile {PSYCHOPHILE_UPPER_BOUND_K}–{MESOPHILE_UPPER_BOUND_K} K · "
    f"Thermophile > {MESOPHILE_UPPER_BOUND_K} K. The water score is a heuristic "
    "using 288 K as its optimum and 0.8–1.6 Earth radii as a simple terrestrial-size gate."
)

if filtered.empty:
    st.info("No planets match these filters. Try expanding the temperature range or radius limit.")
else:
    st.subheader("🔭 Exoplanet population: thermal regime distribution")
    thermal_colors = {
        "Psychophile": "#00d2ff",
        "Mesophile": "#00ff87",
        "Thermophile": "#ff6a00",
    }
    figure = px.scatter(
        filtered,
        x="pl_eqt",
        y="pl_rade",
        color="Thermal regime (proxy)",
        color_discrete_map=thermal_colors,
        hover_name="pl_name",
        labels={
            "pl_eqt": "Equilibrium temperature (K)",
            "pl_rade": "Planetary radius (Earth radii, R⊕)",
            "Thermal regime (proxy)": "Thermal regime",
        },
        title="🔭 Exoplanet Population: Extremophile Thermal Regime Distribution",
    )
    figure.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#c9d1d9",
        title_font_size=16,
        legend={
            "title": "Thermal regime",
            "bgcolor": "rgba(22, 27, 34, 0.8)",
            "bordercolor": "rgba(88, 166, 255, 0.3)",
            "borderwidth": 1,
        },
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    figure.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(48, 54, 61, 0.5)",
        zeroline=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(48, 54, 61, 0.5)",
        zeroline=False,
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )

    display_columns = {
        "pl_name": "Planet",
        "hostname": "Host star",
        "ra": "Right ascension (degrees)",
        "dec": "Declination (degrees)",
        "sy_dist": "System distance (parsecs)",
        "pl_eqt": "Equilibrium temperature (K)",
        "pl_rade": "Radius (Earth radii)",
        "pl_insol": "Stellar flux (Earth fluxes)",
        "discoverymethod": "Discovery method",
        "disc_year": "Discovery year",
        "Thermal regime (proxy)": "Thermal regime (proxy)",
        "Liquid water screening score (%)": "Liquid water screening score (%)",
        "Stellar flux regime (proxy)": "Stellar flux regime (proxy)",
    }
    available_columns = [column for column in display_columns if column in filtered.columns]
    results = filtered[available_columns].rename(columns=display_columns)
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download matching planets as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="exoplanet_habitability_candidates.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔬 Planet Deep-Dive Inspector")
    planet_options = filtered["pl_name"].sort_values().tolist()
    current_planet = st.session_state.get("planet_deep_dive")
    if current_planet not in planet_options:
        roulette_planet = st.session_state.get("roulette_planet")
        st.session_state["planet_deep_dive"] = (
            roulette_planet if roulette_planet in planet_options else planet_options[0]
        )
    selected_name = st.selectbox(
        "🪐 Select a filtered planet",
        options=planet_options,
        key="planet_deep_dive",
    )
    planet = filtered.loc[filtered["pl_name"] == selected_name].iloc[0]
    temperature_delta = planet["pl_eqt"] - EARTH_EQUILIBRIUM_TEMPERATURE_K
    radius_delta = planet["pl_rade"] - 1

    st.markdown(f"#### {planet['pl_name']}")
    detail_columns = st.columns(5)
    detail_columns[0].metric(
        "🌡️ Equilibrium temperature",
        f"{planet['pl_eqt']:.1f} K",
        f"{temperature_delta:+.1f} K vs Earth ({EARTH_EQUILIBRIUM_TEMPERATURE_K} K)",
    )
    detail_columns[1].metric(
        "📏 Planetary radius",
        f"{planet['pl_rade']:.2f} R⊕",
        f"{radius_delta:+.2f} R⊕ vs Earth",
    )
    detail_columns[2].metric(
        "🧬 Thermal regime (proxy)", planet["Thermal regime (proxy)"]
    )
    water_score = planet["Liquid water screening score (%)"]
    detail_columns[3].metric("💧 Liquid water screening score", f"{water_score:.1f}%")
    stellar_flux = planet.get("pl_insol", float("nan"))
    detail_columns[4].metric(
        "☀️ Stellar flux",
        f"{stellar_flux:.2f} S⊕" if pd.notna(stellar_flux) else "Unavailable",
        classify_stellar_flux(stellar_flux),
    )
    right_ascension = planet.get("ra", float("nan"))
    declination = planet.get("dec", float("nan"))
    distance_parsecs = planet.get("sy_dist", float("nan"))

    if planet["pl_eqt"] < EARTH_EQUILIBRIUM_TEMPERATURE_K:
        thermal_comparison = (
            "Its estimated equilibrium temperature is below Earth's effective equilibrium "
            "temperature; a sufficiently insulating atmosphere could alter surface conditions."
        )
    elif planet["pl_eqt"] > EARTH_EQUILIBRIUM_TEMPERATURE_K:
        thermal_comparison = (
            "Its estimated equilibrium temperature is above Earth's effective equilibrium "
            "temperature; the actual surface temperature depends on its atmosphere and other factors."
        )
    else:
        thermal_comparison = (
            "Its estimated equilibrium temperature matches Earth's effective equilibrium "
            "temperature, though this alone does not establish Earth-like surface conditions."
        )

    st.markdown("**🧬 Biological compatibility breakdown**")
    st.markdown(f"- **Thermal comparison:** {thermal_comparison}")
    st.markdown(
        f"- **Size comparison:** The radius is {planet['pl_rade']:.2f} times Earth's. "
        "Radius alone cannot determine surface gravity, composition, or whether the planet "
        "has a solid surface."
    )
    st.markdown(
        f"- **Temperature-regime proxy:** Classified as **{planet['Thermal regime (proxy)']}** "
        "using the displayed equilibrium-temperature bands. Planet equilibrium temperature "
        "is not the temperature experienced by organisms."
    )
    st.markdown(
        f"- **Liquid-water screening score:** {water_score:.1f}%. This heuristic decreases "
        f"linearly with distance from {LIQUID_WATER_OPTIMAL_TEMPERATURE_K} K and is set to "
        f"zero outside the {TERRESTRIAL_RADIUS_RANGE_EARTH[0]:.1f}–"
        f"{TERRESTRIAL_RADIUS_RANGE_EARTH[1]:.1f} Earth-radius screening zone. It is not a "
        "measured probability and does not establish surface liquid water."
    )

    st.markdown("#### ☢️ Stellar radiation & human-habitability assessment")
    if pd.notna(stellar_flux):
        st.markdown(
            f"**Measured bolometric stellar flux:** {stellar_flux:.2f} times Earth's "
            f"({classify_stellar_flux(stellar_flux)}). This describes total incident energy, "
            "not ionizing radiation or biological dose."
        )
    else:
        st.markdown(
            "**Measured bolometric stellar flux:** unavailable for this planet in the archive."
        )
    radiation_columns = st.columns([1, 1, 1], gap="medium")
    radiation_columns[0].metric("🧑‍🚀 Unprotected human habitation", "Not established")
    radiation_columns[1].metric("🛡️ Radiation shielding need", "Unknown")
    radiation_columns[2].metric("☢️ Flare/radiation lethality", "Not assessable")
    st.caption(
        "The archive does not provide stellar flare history, high-energy (UV/X-ray) spectra, "
        "atmospheric shielding, magnetic-field strength, or surface radiation dose. Therefore "
        "stellar flux alone cannot establish unprotected safety, a shielding requirement, "
        "or lethal radiation conditions."
    )
    st.warning(
        "This inspector is a preliminary comparison, not a habitability verdict. "
        "Atmosphere, pressure, actual liquid-water availability, stellar activity, and "
        "surface radiation are not assessed by these data."
    )
    st.markdown("#### 🚀 Cosmic distance & travel-time calculator")
    if pd.notna(distance_parsecs) and distance_parsecs >= 0:
        distance_light_years, light_speed_years, voyager_years = calculate_travel_times(
            float(distance_parsecs)
        )
        travel_columns = st.columns(3)
        travel_columns[0].metric(
            "📍 System distance",
            f"{distance_light_years:,.2f} light-years",
            f"{distance_parsecs:,.2f} parsecs",
        )
        travel_columns[1].metric(
            "✨ At light speed (c)",
            format_years(light_speed_years),
            "Idealized; no acceleration or travel-time effects",
        )
        travel_columns[2].metric(
            "🛰️ At Voyager 1 cruise speed",
            format_years(voyager_years),
            f"{VOYAGER_1_SPEED_KM_PER_HOUR:,} km/h",
        )
        voyager_percent_c = VOYAGER_1_SPEED_KM_PER_HOUR / LIGHT_SPEED_KM_PER_HOUR * 100
        st.caption(
            f"Using the catalog system distance (sy_dist, in parsecs) and "
            f"1 pc = {PARSEC_TO_LIGHT_YEAR} light-years. Voyager 1's approximate speed is "
            f"{voyager_percent_c:.5f}% of light speed. Travel times are simple distance ÷ "
            "speed estimates; they ignore acceleration, deceleration, and propulsion limits."
        )
    else:
        st.info("A system-distance measurement is unavailable for this target.")

    radiation_status = (
        f"Bolometric flux {stellar_flux:.2f} times Earth's "
        f"({classify_stellar_flux(stellar_flux)}); surface radiation dose and human safety "
        "are unknown from the available data."
        if pd.notna(stellar_flux)
        else "Stellar flux, surface radiation dose, and human safety are unavailable or "
        "unknown from the available data."
    )
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", selected_name).strip("._")
    pdf_data = generate_pdf_report(
        planet_name=selected_name,
        equilibrium_temperature=float(planet["pl_eqt"]),
        radius_earth=float(planet["pl_rade"]),
        thermal_regime=str(planet["Thermal regime (proxy)"]),
        water_score=float(water_score),
        radiation_status=radiation_status,
        right_ascension=right_ascension,
        declination=declination,
        stellar_flux=stellar_flux,
    )
    st.download_button(
        label="📥 Download Planetary Assessment Sheet (PDF)",
        data=pdf_data,
        file_name=f"{safe_filename or 'planet'}_Assessment_Report.pdf",
        mime="application/pdf",
    )

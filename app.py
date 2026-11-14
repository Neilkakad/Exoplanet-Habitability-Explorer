from __future__ import annotations

import base64
import io
import logging
import math
import random
import re
import unicodedata
from html import escape
from http.client import HTTPException
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NASA_EXOPLANET_ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name%2Cpl_eqt%2Cpl_rade%2Cpl_insol%2Cpl_bmasse%2C"
    "pl_orbsmax%2Cst_teff%2Cra%2Cdec%2Csy_dist%2Chostname%2C"
    "discoverymethod%2Cdisc_year+from+pscomppars+where+"
    "pl_eqt+is+not+null+and+pl_rade+is+not+null&format=csv"
)
EARTH_EQUILIBRIUM_TEMPERATURE_K = 255
EARTH_RADIUS_KM = 6371.0
EARTH_GRAVITY_M_S2 = 9.80665
EARTH_ESCAPE_VELOCITY_KM_S = 11.186
LIQUID_WATER_OPTIMAL_TEMPERATURE_K = 288
LIQUID_WATER_TEMPERATURE_TOLERANCE_K = 88
TERRESTRIAL_RADIUS_RANGE_EARTH = (0.8, 1.6)
PSYCHOPHILE_UPPER_BOUND_K = 288
MESOPHILE_UPPER_BOUND_K = 318
WATER_FREEZING_POINT_K = 273.15
ESI_RADIUS_WEIGHT = 0.57
ESI_TEMPERATURE_WEIGHT = 5.58
PARSEC_TO_LIGHT_YEAR = 3.26156
LIGHT_SPEED_KM_PER_HOUR = 299_792.458 * 3600
VOYAGER_1_SPEED_KM_PER_HOUR = 61_200
JWST_BACKGROUNDS = (
    (
        "Herbig–Haro 46/47 — NIRCam",
        "https://cdn.esawebb.org/archives/images/screen/weic2319a.jpg",
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
COSMIC_INSIGHTS = (
    "Sunlight takes about 8 minutes and 20 seconds to reach Earth, so every sunrise shows our star as it was in the recent past.",
    "Many elements heavier than helium were forged in stars and stellar explosions; the atoms in planets and people carry that cosmic history.",
    "Solar fusion makes neutrinos that pass through Earth and our bodies almost unhindered; enormous detectors are needed to catch a few.",
    "Astronomers can find an exoplanet when it briefly dims its star during a transit, and repeated transits reveal the planet's orbital period.",
    "JWST observes infrared light, which helps astronomers study cool objects and see through some cosmic dust to distant star-forming regions.",
    "Earth microbes live in places such as Antarctic ice, acidic hot springs, and deep-sea vents, showing that life can tolerate diverse environments.",
    "Liquid water depends on pressure as well as temperature, so a planet's estimated temperature alone cannot tell us whether its surface has oceans.",
)


def classify_temperature(temperature_k: float) -> str:
    """Assign a broad temperature-regime proxy, not a biological diagnosis."""
    if temperature_k < PSYCHOPHILE_UPPER_BOUND_K:
        return "Psychrophile"
    if temperature_k <= MESOPHILE_UPPER_BOUND_K:
        return "Mesophile"
    return "Thermophile"


def calculate_esi_proxy(radius_earth: float, temperature_k: float) -> float:
    """Calculate a radius/equilibrium-temperature ESI proxy using Earth reference values."""
    if radius_earth <= 0 or temperature_k <= 0:
        return float("nan")
    radius_similarity = 1 - abs(radius_earth - 1.0) / (radius_earth + 1.0)
    temperature_similarity = 1 - abs(
        temperature_k - EARTH_EQUILIBRIUM_TEMPERATURE_K
    ) / (temperature_k + EARTH_EQUILIBRIUM_TEMPERATURE_K)
    return (
        radius_similarity ** (ESI_RADIUS_WEIGHT / 2)
        * temperature_similarity ** (ESI_TEMPERATURE_WEIGHT / 2)
    )


def classify_esi(esi: float) -> str:
    if not math.isfinite(esi):
        return "Unavailable"
    if esi >= 0.8:
        return "High Earth parity"
    if esi >= 0.6:
        return "Moderate Earth parity"
    return "Low Earth parity"


def estimate_planet_mass(radius_earth: float, measured_mass: float) -> tuple[float, str]:
    """Prefer archive mass; otherwise use a piecewise empirical mean mass-radius fit."""
    if not math.isfinite(radius_earth) or radius_earth <= 0:
        return float("nan"), "Unavailable: invalid planetary radius"
    if math.isfinite(measured_mass) and measured_mass > 0:
        return measured_mass, "NASA Archive measured mass"
    if radius_earth <= 1.23:
        return 0.9718 * radius_earth**3.58, "Estimated: Terran mass-radius fit"
    return 1.436 * radius_earth**1.70, "Estimated: Neptunian mass-radius fit"


def estimate_surface_gravity_and_escape(
    mass_earth: float, radius_earth: float
) -> tuple[float, float]:
    if (
        not math.isfinite(mass_earth)
        or not math.isfinite(radius_earth)
        or mass_earth <= 0
        or radius_earth <= 0
    ):
        return float("nan"), float("nan")
    gravity = EARTH_GRAVITY_M_S2 * mass_earth / radius_earth**2
    escape_velocity = EARTH_ESCAPE_VELOCITY_KM_S * math.sqrt(mass_earth / radius_earth)
    return gravity, escape_velocity


def classify_radius_regime(radius_earth: float) -> str:
    if radius_earth <= 1.25:
        return "Rocky terrestrial size"
    if radius_earth <= 1.75:
        return "Super-Earth size"
    return "Mini-Neptune candidate"


def classify_insolation(flux_earth: float) -> str:
    """Classify bolometric flux against solar-calibrated HZ reference limits."""
    if flux_earth < 0.36:
        return "Below the conservative habitable-zone range"
    if flux_earth <= 1.06:
        return "Within the conservative habitable-zone range"
    if flux_earth <= 1.78:
        return "Beyond the conservative inner edge"
    return "Beyond the optimistic inner edge"


def assess_tidal_and_stripping_proxy(
    stellar_temperature_k: float,
    orbital_distance_au: float,
    flux_earth: float,
    escape_velocity_km_s: float,
) -> tuple[str, str]:
    """Return screening labels, not a tidal timescale or atmospheric escape rate."""
    has_orbit = math.isfinite(orbital_distance_au) and orbital_distance_au > 0
    has_stellar_temperature = (
        math.isfinite(stellar_temperature_k) and stellar_temperature_k > 0
    )
    is_cool_red_dwarf = has_stellar_temperature and stellar_temperature_k < 3900
    close_orbit = has_orbit and orbital_distance_au <= 0.1

    if has_orbit and has_stellar_temperature:
        if is_cool_red_dwarf and close_orbit:
            tidal_label = "High Tidal-Locking Zone"
            tidal_detail = "Cool red-dwarf host; orbit within 0.1 AU"
        elif close_orbit:
            tidal_label = "Close-In Orbit"
            tidal_detail = "Orbit within 0.1 AU"
        elif is_cool_red_dwarf:
            tidal_label = "Cool Red-Dwarf Orbit"
            tidal_detail = "Red-dwarf host; orbit beyond 0.1 AU"
        else:
            tidal_label = "Lower Tidal-Locking Risk"
            tidal_detail = "No close orbit around a cool red dwarf"
    else:
        tidal_label = "More Data Needed"
        tidal_detail = "Host-star temperature and orbital distance are not both available"

    if math.isfinite(flux_earth) and math.isfinite(escape_velocity_km_s):
        if flux_earth >= 10 and escape_velocity_km_s < 15:
            loss_label = (
                "High starlight and lower escape speed may challenge atmosphere retention"
            )
        elif flux_earth > 1.06 and escape_velocity_km_s < 11.2:
            loss_label = (
                "Elevated starlight and lower escape speed may make atmosphere retention "
                "more difficult"
            )
        else:
            loss_label = "No strong warning from the available starlight and escape speed"
    else:
        loss_label = "Incoming starlight or escape speed is unavailable"
    return tidal_label, f"{tidal_detail}. {loss_label}."


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


def describe_host_star(stellar_temperature_k: float) -> str:
    """Return a broad temperature-based host-star description, not a catalog spectral type."""
    if not math.isfinite(stellar_temperature_k) or stellar_temperature_k <= 0:
        return "host-star temperature unavailable"
    if stellar_temperature_k < 3900:
        star_type = "M-dwarf-like"
    elif stellar_temperature_k < 5200:
        star_type = "K-type-like"
    elif stellar_temperature_k < 6000:
        star_type = "G-type-like"
    elif stellar_temperature_k < 7500:
        star_type = "F-type-like"
    else:
        star_type = "hotter-than-F-type"
    return f"{star_type} host (temperature-based estimate, {stellar_temperature_k:.0f} K)"


def build_mission_classification(
    esi: float,
    thermal_regime: str,
    structure_regime: str,
    stellar_temperature_k: float,
    orbital_distance_au: float,
    stellar_flux: float,
) -> tuple[str, str]:
    """Create a planet-specific screening label and non-conclusive verdict."""
    star_is_m_dwarf = math.isfinite(stellar_temperature_k) and stellar_temperature_k < 3900
    close_orbit = math.isfinite(orbital_distance_au) and 0 < orbital_distance_au <= 0.1
    structure_label = {
        "Rocky terrestrial size": "terrestrial-size",
        "Super-Earth size": "Super-Earth-size",
        "Mini-Neptune candidate": "mini-Neptune-size",
    }.get(structure_regime, "planetary-size")
    thermal_label = {
        "Psychrophile": "cold-regime",
        "Mesophile": "temperate-regime",
        "Thermophile": "hot-regime",
    }.get(thermal_regime, "unclassified-temperature")

    if star_is_m_dwarf and close_orbit and structure_regime == "Mini-Neptune candidate":
        classification = "Close-Orbit M-Dwarf Mini-Neptune Greenhouse Screen"
    elif star_is_m_dwarf and close_orbit:
        classification = f"Close-Orbit M-Dwarf {thermal_label} {structure_label} Candidate"
    else:
        host_prefix = "M-Dwarf-Hosted " if star_is_m_dwarf else ""
        classification = f"{host_prefix}{thermal_label.title()} {structure_label} Candidate"

    esi_summary = (
        f"Two-parameter ESI is {esi:.3f} ({classify_esi(esi)}). "
        if math.isfinite(esi)
        else "Two-parameter ESI is unavailable. "
    )
    if (
        math.isfinite(esi)
        and esi >= 0.8
        and math.isfinite(stellar_flux)
        and 0.36 <= stellar_flux <= 1.06
    ):
        verdict = (
            f"{esi_summary}Reported bolometric flux "
            "is within the solar-calibrated conservative habitable-zone reference band."
        )
    elif math.isfinite(stellar_flux) and stellar_flux > 1.78:
        verdict = (
            f"{esi_summary}High-irradiation screening profile; the flux exceeds the "
            "solar-calibrated "
            "optimistic inner-edge reference."
        )
    elif math.isfinite(stellar_flux) and stellar_flux < 0.36:
        verdict = (
            f"{esi_summary}Low-irradiation screening profile; the flux is below the "
            "solar-calibrated "
            "conservative outer-edge reference."
        )
    elif math.isfinite(esi) and esi >= 0.6:
        verdict = (
            f"{esi_summary}Atmospheric "
            "and surface conditions remain unmeasured by this assessment."
        )
    else:
        verdict = (
            f"{esi_summary}Exploratory screening candidate; current proxy metrics do not establish "
            "surface habitability or life."
        )
    return classification, verdict


def build_telemetry_chart(
    radius_earth: float,
    equilibrium_temperature_k: float,
    surface_gravity_m_s2: float,
    stellar_flux: float,
) -> Drawing:
    """Draw planet-to-Earth telemetry ratios with a visible 1x reference marker."""
    chart_width = 516
    chart = Drawing(chart_width, 132)
    chart.add(
        String(
            4,
            120,
            "PLANET / EARTH TELEMETRY  |  BAR SCALE 0-4x; EARTH REFERENCE AT 1x",
            fontName="Helvetica-Bold",
            fontSize=8,
            fillColor=colors.HexColor("#40566d"),
        )
    )
    metrics = (
        ("Radius", radius_earth, 1.0),
        ("Equilibrium temp.", equilibrium_temperature_k, EARTH_EQUILIBRIUM_TEMPERATURE_K),
        ("Surface gravity", surface_gravity_m_s2, EARTH_GRAVITY_M_S2),
        ("Stellar flux", stellar_flux, 1.0),
    )
    bar_start = 150
    bar_width = 250
    earth_marker_x = bar_start + bar_width / 4
    for index, (label, measured_value, earth_reference) in enumerate(metrics):
        y = 96 - index * 21
        chart.add(
            String(
                4,
                y,
                label,
                fontName="Helvetica",
                fontSize=8,
                fillColor=colors.HexColor("#263442"),
            )
        )
        chart.add(
            Rect(
                bar_start,
                y - 2,
                bar_width,
                8,
                fillColor=colors.HexColor("#e6edf4"),
                strokeColor=None,
            )
        )
        if (
            math.isfinite(measured_value)
            and math.isfinite(earth_reference)
            and earth_reference > 0
            and measured_value >= 0
        ):
            ratio = measured_value / earth_reference
            chart.add(
                Rect(
                    bar_start,
                    y - 2,
                    bar_width * min(ratio, 4.0) / 4.0,
                    8,
                    fillColor=colors.HexColor("#1f77b4"),
                    strokeColor=None,
                )
            )
            ratio_label = f"{ratio:.2f}x Earth" + (" (bar capped)" if ratio > 4 else "")
        else:
            ratio_label = "Unavailable"
        chart.add(
            String(
                bar_start + bar_width + 10,
                y,
                ratio_label,
                fontName="Helvetica",
                fontSize=8,
                fillColor=colors.HexColor("#263442"),
            )
        )
    chart.add(
        Line(
            earth_marker_x,
            21,
            earth_marker_x,
            108,
            strokeColor=colors.HexColor("#40566d"),
            strokeWidth=1.4,
        )
    )
    chart.add(
        String(bar_start, 8, "0x", fontName="Helvetica", fontSize=7, fillColor=colors.grey)
    )
    chart.add(
        String(
            earth_marker_x - 8,
            8,
            "1x Earth",
            fontName="Helvetica-Bold",
            fontSize=7,
            fillColor=colors.HexColor("#40566d"),
        )
    )
    chart.add(
        String(
            bar_start + bar_width - 12,
            8,
            "4x+",
            fontName="Helvetica",
            fontSize=7,
            fillColor=colors.grey,
        )
    )
    return chart


def format_unit_badge(value: str) -> str:
    return f'<span class="unit-badge">{escape(value)}</span>'


def format_science_card(
    title: str,
    value: str,
    definition: str,
    secondary_value: str = "",
    secondary_is_badge: bool = False,
) -> str:
    """Render a text-safe, wrapping HTML card for the dashboard's custom metric grids."""
    secondary_content = (
        format_unit_badge(secondary_value)
        if secondary_is_badge
        else escape(secondary_value)
    )
    secondary_markup = (
        f'<div class="science-card-secondary">{secondary_content}</div>'
        if secondary_value
        else ""
    )
    return (
        '<section class="science-card">'
        f'<div class="science-card-title">{escape(title)}</div>'
        f'<div class="science-card-value">{format_unit_badge(value)}</div>'
        f"{secondary_markup}"
        f'<div class="science-card-definition">{escape(definition)}</div>'
        "</section>"
    )


@st.cache_resource(show_spinner=False)
def preload_jwst_backgrounds() -> tuple[dict[str, str], tuple[str, ...]]:
    """Fetch, resize, and cache every JWST background as a compact JPEG data URL."""
    backgrounds = {}
    errors = []
    for title, image_url in JWST_BACKGROUNDS:
        try:
            request = Request(
                image_url,
                headers={"User-Agent": "Exoplanet-Habitability-Explorer"},
            )
            with urlopen(request, timeout=15) as response:
                media_type = response.headers.get_content_type()
                if media_type not in {"image/jpeg", "image/png", "image/webp"}:
                    raise ValueError(
                        f"Unexpected image media type: {media_type}"
                    )
                image_bytes = response.read()
            if not image_bytes:
                raise ValueError("The image response was empty.")

            with Image.open(io.BytesIO(image_bytes)) as source_image:
                if source_image.width * source_image.height > 100_000_000:
                    raise ValueError("The image dimensions are too large.")
                source_image.load()
                resized_image = source_image.convert("RGB")
            resized_image.thumbnail((1366, 768), Image.Resampling.LANCZOS)
            compressed_image = io.BytesIO()
            resized_image.save(
                compressed_image,
                format="JPEG",
                quality=75,
                optimize=True,
            )
            resized_image.close()
            encoded_image = base64.b64encode(compressed_image.getvalue()).decode("ascii")
            backgrounds[title] = f"data:image/jpeg;base64,{encoded_image}"
        except (
            HTTPException,
            Image.DecompressionBombError,
            OSError,
            URLError,
            ValueError,
        ) as error:
            errors.append(f"{title}: {error}")
    return backgrounds, tuple(errors)


@st.cache_data(show_spinner=False)
def build_population_figure(population: pd.DataFrame) -> go.Figure:
    """Build and cache the chart from only the filtered population columns."""
    thermal_colors = {
        "Psychrophile": "#00d2ff",
        "Mesophile": "#00ff87",
        "Thermophile": "#ff6a00",
    }
    figure = px.scatter(
        population,
        x="pl_eqt",
        y="pl_rade",
        color="Thermal regime (proxy)",
        color_discrete_map=thermal_colors,
        category_orders={
            "Thermal regime (proxy)": ["Mesophile", "Thermophile", "Psychrophile"]
        },
        hover_name="pl_name",
        labels={
            "pl_eqt": "Equilibrium temperature (K)",
            "pl_rade": "Planetary radius (Earth radii)",
            "Thermal regime (proxy)": "Thermal regime",
        },
    )
    figure.update_layout(
        title={
            "text": "🔭 Exoplanet Population: Extremophile Thermal Regime Distribution",
            "font": {"color": "#F8FAFC", "size": 18},
        },
        plot_bgcolor="rgba(30, 41, 59, 0.72)",
        paper_bgcolor="rgba(15, 23, 42, 0.85)",
        font={"color": "#F8FAFC", "size": 12},
        legend={
            "title": {"text": "Thermal regime", "font": {"color": "#F8FAFC", "size": 13}},
            "font": {"color": "#F8FAFC", "size": 13},
            "bgcolor": "rgba(10, 15, 30, 0.82)",
            "bordercolor": "rgba(255, 255, 255, 0.16)",
            "borderwidth": 1,
        },
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        xaxis_title_font={"color": "#FFFFFF"},
        yaxis_title_font={"color": "#FFFFFF"},
    )
    figure.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.2)",
        zeroline=False,
        tickfont={"color": "#F8FAFC"},
    )
    figure.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(148, 163, 184, 0.2)",
        zeroline=False,
        tickfont={"color": "#F8FAFC"},
    )
    return figure


@st.cache_data(ttl=3600, show_spinner="Loading exoplanet data...")
def load_exoplanets() -> pd.DataFrame:
    """Load planets with measured equilibrium temperature and radius."""
    data = pd.read_csv(NASA_EXOPLANET_ARCHIVE_URL)
    for column in (
        "pl_eqt",
        "pl_rade",
        "pl_insol",
        "pl_bmasse",
        "pl_orbsmax",
        "st_teff",
        "ra",
        "dec",
        "sy_dist",
    ):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["pl_eqt", "pl_rade"])


@st.cache_data(show_spinner=False)
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
    esi: float,
    structure_regime: str,
    planet_mass_earth: float,
    mass_basis: str,
    surface_gravity_m_s2: float,
    escape_velocity_km_s: float,
    stellar_temperature_k: float,
    orbital_distance_au: float,
    distance_parsecs: float,
) -> bytes:
    """Build a planet-specific mission dossier from measured values and screening proxies."""
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
    banner_label_style = ParagraphStyle(
        "MissionBannerLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#b8dcf5"),
    )
    banner_title_style = ParagraphStyle(
        "MissionBannerTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.white,
    )
    banner_verdict_style = ParagraphStyle(
        "MissionBannerVerdict",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.white,
    )

    def safe_text(value: object) -> str:
        normalized = unicodedata.normalize("NFKD", str(value))
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        return escape(ascii_text, quote=False)

    def coordinate_text(value: float | None) -> str:
        return f"{float(value):.8f} deg" if value is not None and pd.notna(value) else "Unavailable"

    mission_class, mission_verdict = build_mission_classification(
        esi,
        thermal_regime,
        structure_regime,
        stellar_temperature_k,
        orbital_distance_au,
        stellar_flux,
    )
    host_description = describe_host_star(stellar_temperature_k)
    if math.isfinite(distance_parsecs) and distance_parsecs >= 0:
        distance_light_years, light_time_years, voyager_years = calculate_travel_times(
            distance_parsecs
        )
        travel_summary = (
            f"NASA Archive system distance is {distance_parsecs:.3f} pc "
            f"({distance_light_years:.2f} light-years). A one-way light signal would take "
            f"about {format_years(light_time_years)}; a round-trip exchange about "
            f"{format_years(2 * light_time_years)}. At Voyager 1's approximate cruise speed, "
            f"the distance corresponds to {format_years(voyager_years)}."
        )
    else:
        travel_summary = (
            "The catalog system distance is unavailable, so light-signal lag and cruise-time "
            "comparisons cannot be estimated."
        )
    orbit_summary = (
        f"The catalog semi-major axis is {orbital_distance_au:.4g} AU."
        if math.isfinite(orbital_distance_au) and orbital_distance_au > 0
        else "The catalog semi-major axis is unavailable."
    )
    flux_summary = (
        f"{stellar_flux:.3g} Earth fluxes"
        if math.isfinite(stellar_flux)
        else "unavailable"
    )
    observation_brief = (
        f"{safe_text(planet_name)} is screened as a {safe_text(structure_regime.lower())} "
        f"world with radius {radius_earth:.2f} Earth radii, equilibrium temperature "
        f"{equilibrium_temperature:.1f} K, and bolometric flux {flux_summary}; "
        f"the host is described as {safe_text(host_description)}. "
        f"{orbit_summary} {travel_summary} For this target, JWST transit or eclipse "
        "spectroscopy could investigate atmospheric H2O, CO2, and CH4, with O3 as a "
        "more demanding secondary target where signal quality permits. This is an observing "
        "target list, not a prediction of detection: sensitivity depends on the atmosphere, "
        "clouds or hazes, host brightness, and repeat observations, and no gas alone proves life."
    )

    measured_rows = [
        ["Parameter", "Value", "Basis"],
        ["Target", safe_text(planet_name), "NASA Exoplanet Archive"],
        ["Right ascension (RA)", coordinate_text(right_ascension), "Archive coordinate (degrees)"],
        ["Declination (Dec)", coordinate_text(declination), "Archive coordinate (degrees)"],
        ["Equilibrium temperature", f"{equilibrium_temperature:.1f} K", "Archive composite parameter"],
        ["Planetary radius", f"{radius_earth:.2f} Earth radii", "Archive composite parameter"],
        [
            "Earth Similarity Index (2-parameter)",
            f"{esi:.3f}" if math.isfinite(esi) else "Unavailable",
            "Radius and equilibrium-temperature proxy",
        ],
        ["Planet size regime", safe_text(structure_regime), "Radius-based composition proxy"],
        [
            "Planet mass",
            f"{planet_mass_earth:.2f} Earth masses"
            if math.isfinite(planet_mass_earth)
            else "Unavailable",
            safe_text(mass_basis),
        ],
        [
            "Surface gravity / escape velocity",
            f"{surface_gravity_m_s2:.2f} m/s2 / {escape_velocity_km_s:.2f} km/s"
            if math.isfinite(surface_gravity_m_s2) and math.isfinite(escape_velocity_km_s)
            else "Unavailable",
            "Derived from measured or estimated mass and radius",
        ],
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
        ["Host-star description", safe_text(host_description), "Estimated from effective temperature"],
        [
            "Semi-major axis",
            f"{orbital_distance_au:.4g} AU"
            if math.isfinite(orbital_distance_au)
            else "Unavailable",
            "NASA Exoplanet Archive",
        ],
        [
            "System distance",
            f"{distance_parsecs:.4g} pc"
            if math.isfinite(distance_parsecs)
            else "Unavailable",
            "NASA Exoplanet Archive",
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
        colWidths=[150, 145, 221],
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
    mission_banner = Table(
        [
            [
                Paragraph("MISSION CLASSIFICATION", banner_label_style),
                Paragraph(safe_text(mission_class), banner_title_style),
            ],
            [
                Paragraph("HABITABILITY VERDICT", banner_label_style),
                Paragraph(safe_text(mission_verdict), banner_verdict_style),
            ],
        ],
        colWidths=[150, 366],
        hAlign="LEFT",
    )
    mission_banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#102a43")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1f77b4")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.45, colors.HexColor("#45657f")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story = [
        Paragraph("PLANETARY MISSION DOSSIER", title_style),
        Paragraph(
            f"EXOPLANET HABITABILITY &amp; BIOSIGNATURE EXPLORER &nbsp;|&nbsp; "
            f"{safe_text(planet_name)}",
            subtitle_style,
        ),
        HRFlowable(width="100%", thickness=1.1, color=colors.HexColor("#1f77b4"), spaceAfter=12),
        Paragraph("Mission Classification &amp; Habitability Verdict", section_style),
        mission_banner,
        Spacer(1, 10),
        Paragraph("Measured Parameters and Derived Indicators", section_style),
        measured_table,
        Spacer(1, 13),
        KeepTogether(
            [
                Paragraph("Comparative Telemetry", section_style),
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#b8c5d1"),
                    spaceAfter=8,
                ),
                Paragraph(
                    "Bars compare the selected planet with the Earth reference value (1.0x); "
                    "the plotted scale is capped at 4x so smaller differences remain legible.",
                    note_style,
                ),
                build_telemetry_chart(
                    radius_earth,
                    equilibrium_temperature,
                    surface_gravity_m_s2,
                    stellar_flux
                    if stellar_flux is not None and pd.notna(stellar_flux)
                    else float("nan"),
                ),
            ]
        ),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8c5d1"), spaceAfter=8),
        Paragraph("Mission Engineering &amp; Biosignature Observation Brief", section_style),
        Paragraph(observation_brief, note_style),
        Spacer(1, 13),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8c5d1"), spaceAfter=8),
        Paragraph("Method and limitations", section_style),
        Paragraph(
            "Equilibrium temperature, radius, coordinates, distance, stellar flux, and "
            "available orbital/stellar parameters are NASA Exoplanet Archive values. The "
            "two-parameter ESI is a radius/equilibrium-temperature similarity proxy, not the "
            "canonical four-property ESI. Missing mass is estimated with a population "
            "mass-radius relation; gravity and escape velocity inherit that uncertainty. "
            "Thermal, insolation, structural, and tidal labels are screening proxies only. "
            "They do not measure surface conditions, atmospheric retention, biosignatures, "
            "or habitability.",
            note_style,
        ),
    ]
    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


st.set_page_config(
    page_title="Exoplanet Habitability & Biosignature Explorer",
    page_icon="🪐",
    layout="wide",
)
st.session_state.setdefault("roulette_planet", None)
selected_planet_state = st.session_state.get("planet_deep_dive")
previous_planet_state = st.session_state.get("_jwst_background_planet")
try:
    preloaded_backgrounds, background_load_errors = preload_jwst_backgrounds()
except (HTTPException, OSError, URLError, ValueError):
    logging.getLogger(__name__).exception("Could not load JWST background images.")
    preloaded_backgrounds = {}
    background_load_errors = ()
for background_error in background_load_errors:
    logging.getLogger(__name__).warning(
        "Could not preload a JWST background: %s",
        background_error,
    )
available_backgrounds = [
    background
    for background in JWST_BACKGROUNDS
    if background[0] in preloaded_backgrounds
]
previous_background = st.session_state.get("_jwst_background")
if not available_backgrounds:
    logging.getLogger(__name__).error(
        "No JWST background images could be loaded; using the dark gradient fallback."
    )
    background_title = ""
    background_data_url = ""
else:
    if (
        previous_background not in available_backgrounds
        or selected_planet_state != previous_planet_state
    ):
        next_backgrounds = [
            background
            for background in available_backgrounds
            if background != previous_background
        ]
        st.session_state["_jwst_background"] = random.choice(
            next_backgrounds or available_backgrounds
        )
    background_title, _ = st.session_state["_jwst_background"]
    background_data_url = preloaded_backgrounds[background_title]
st.session_state["_jwst_background_planet"] = selected_planet_state
background_image_layer = (
    f'url("{background_data_url}")' if background_data_url else "none"
)

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

    [data-testid="stMain"],
    [data-testid="stMain"]::before,
    [data-testid="stMain"]::after,
    .stApp::before,
    .stApp::after {
        border: 0 !important;
        box-shadow: none !important;
        content: none !important;
    }

    .stApp {
        color: var(--ink-100);
        background-image:
            linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)),
            __JWST_BACKGROUND_IMAGE_LAYER__;
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
        background: rgba(12, 16, 28, 0.85);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: none;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.65rem !important;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: 12px;
        padding: 0.8rem 0.9rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #38BDF8 !important;
        font-weight: 650;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: 0.85rem 0.9rem 0.65rem;
        margin: 0.35rem 0;
        border: 1px solid #30363d;
        border-radius: 10px;
        background: #161b22;
    }

    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"],
    [data-testid="stSidebar"] [data-testid^="stTickBar"],
    [data-testid="stSidebar"] [data-testid^="stTickBar"] * {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"],
    [data-baseweb="slider"] div[data-testid="stMarkdownContainer"] p,
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] *,
    [data-testid="stSidebar"] [data-baseweb="slider"] *,
    [data-testid="stTickBar"] *,
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {
        color: #E2E8F0 !important;
        -webkit-text-fill-color: #E2E8F0 !important;
        opacity: 1 !important;
        filter: none !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] * {
        color: #38BDF8 !important;
        -webkit-text-fill-color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
        border: 0 !important;
        box-shadow: none !important;
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
        display: inline-flex;
        width: fit-content;
        max-width: 100%;
        padding: 0.16rem 0.42rem;
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 7px;
        color: #38BDF8 !important;
        background: rgba(15, 23, 42, 0.75) !important;
        font-family: 'DM Mono', monospace;
        font-size: 0.76rem;
    }

    [data-testid="stMetricDelta"] * {
        color: inherit !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    [data-testid="stMetricDelta"] svg {
        fill: currentColor;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.85);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.08);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .insight-box {
        margin: 0.6rem 0 1.4rem;
        padding: 1rem 1.15rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 3px solid #38BDF8;
        border-radius: 10px;
        color: #E2E8F0;
        background: rgba(15, 23, 42, 0.82);
        line-height: 1.65;
    }

    .science-card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.55rem 0 1.25rem;
    }

    .science-card {
        min-width: 0;
        min-height: 150px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        color: #F8FAFC;
        background: rgba(15, 23, 42, 0.88);
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .science-card-title {
        margin-bottom: 0.55rem;
        color: #7DD3FC;
        font-size: 0.83rem;
        font-weight: 750;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }

    .science-card-temperature {
        margin: 0 0 0.65rem;
        color: #CBD5E1;
        font-size: 0.82rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    .science-card-value {
        margin-bottom: 0.55rem;
        color: #F8FAFC;
        font-family: 'DM Mono', monospace;
        font-size: clamp(1.1rem, 1.5vw, 1.35rem);
        font-weight: 700;
        line-height: 1.4;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .unit-badge {
        display: inline-block;
        max-width: 100%;
        padding: 0.12rem 0.5rem;
        border: 1px solid rgba(56, 189, 248, 0.24);
        border-radius: 999px;
        color: #BAE6FD;
        background: rgba(15, 23, 42, 0.8);
        font-family: 'DM Mono', monospace;
        font-size: 0.86em;
        font-style: normal;
        font-weight: 650;
        line-height: 1.5;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .science-card-secondary {
        margin: -0.2rem 0 0.55rem;
        color: #7DD3FC;
        font-size: 0.82rem;
        font-style: normal;
        font-weight: 600;
        line-height: 1.45;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .science-card-definition {
        color: #CBD5E1;
        font-size: 0.8rem;
        line-height: 1.55;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .compatibility-list {
        margin: 0.7rem 0 1.2rem;
        padding: 18px 22px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        color: #F1F5F9;
        background: rgba(15, 23, 42, 0.72);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .compatibility-item {
        padding: 0.85rem 0;
        color: #F1F5F9;
        font-size: 0.94rem;
        line-height: 1.65;
    }

    .compatibility-item + .compatibility-item {
        border-top: 1px solid rgba(148, 163, 184, 0.16);
    }

    .compatibility-title {
        display: block;
        margin-bottom: 0.3rem;
        color: #38BDF8;
        font-weight: 750;
    }

    .compatibility-body {
        color: #F1F5F9;
    }

    .extremophile-card {
        min-height: 178px;
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

    [data-testid="stToast"] {
        border: 1px solid #38BDF8 !important;
        color: #F8FAFC !important;
        background: rgba(15, 23, 42, 0.95) !important;
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.42);
    }

    [data-testid="stToast"] *,
    [data-testid="stToast"] [role="alert"] {
        color: #F8FAFC !important;
        background: transparent !important;
        opacity: 1 !important;
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
        .science-card-grid {
            grid-template-columns: minmax(0, 1fr);
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
    """.replace("__JWST_BACKGROUND_IMAGE_LAYER__", background_image_layer),
    unsafe_allow_html=True,
)
st.title("🪐 Exoplanet Habitability & Biosignature Explorer")
st.markdown(
    "Explore real exoplanet measurements collected by NASA missions and observatories. "
    "Adjust the filters to find distant worlds whose temperature, size, gravity, and "
    "starlight offer useful clues about whether life-friendly conditions might be possible. "
    "These are scientific screening clues—not proof that a planet is habitable or hosts life."
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
    & (exoplanets["pl_rade"] > 0)
    & (exoplanets["pl_rade"] < radius_limit)
].copy()
filtered["Thermal regime (proxy)"] = filtered["pl_eqt"].map(classify_temperature)
filtered["Liquid water screening score (%)"] = filtered.apply(
    lambda planet: liquid_water_probability(planet["pl_eqt"], planet["pl_rade"]),
    axis=1,
)
if "pl_insol" in filtered.columns:
    filtered["Stellar flux regime (proxy)"] = filtered["pl_insol"].map(classify_stellar_flux)

with st.sidebar.container(border=True):
    st.markdown(
        '<div class="compatibility-title">🎲 Surprise Discovery</div>',
        unsafe_allow_html=True,
    )
    st.caption("Jump to a random match in your filters.")
    if st.button(
        "🌍 Roll the Cosmic Dice",
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
        st.caption("No planets match the current filters yet.")

st.subheader("🪐 Planet overview")
overview_row_one = st.columns(3)
overview_row_two = st.columns(3)
average_temperature = filtered["pl_eqt"].mean()
average_radius = filtered["pl_rade"].mean()
average_water_score = filtered["Liquid water screening score (%)"].mean()
psychrophile_share = (
    (filtered["Thermal regime (proxy)"] == "Psychrophile").mean() * 100
    if not filtered.empty
    else 0
)
overview_row_one[0].metric("🪐 Total matches", f"{len(filtered):,}")
overview_row_one[1].metric(
    "🌡️ Mean equilibrium temperature",
    f"{average_temperature:.1f} K" if pd.notna(average_temperature) else "—",
)
overview_row_one[2].metric(
    "📏 Mean planetary radius",
    f"{average_radius:.2f} Earth radii" if pd.notna(average_radius) else "—",
)
overview_row_two[0].metric(
    "💧 Mean water screening score",
    f"{average_water_score:.1f}%" if pd.notna(average_water_score) else "—",
)
overview_row_two[1].metric("🧬 Psychrophile share", f"{psychrophile_share:.1f}%")
st.markdown(
    '<div class="insight-box"><strong>🌌 Cosmic Insight</strong><br/>'
    f"{escape(random.choice(COSMIC_INSIGHTS))}</div>",
    unsafe_allow_html=True,
)

if filtered.empty:
    st.info("No planets match these filters. Try expanding the temperature range or radius limit.")
else:
    st.subheader("🔭 Exoplanet population: thermal regime distribution")
    population_columns = [
        "pl_name",
        "pl_eqt",
        "pl_rade",
        "Thermal regime (proxy)",
    ]
    figure = build_population_figure(filtered[population_columns])
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
    )
    st.markdown("#### 🧬 Meet the extremophiles")
    extremophile_cards = (
        (
            "❄️ Psychrophiles (Cold-lovers)",
            "< 288 K / ~15°C",
            "Cold-adapted organisms thrive in icy environments. Antifreeze proteins help "
            "limit ice-crystal damage inside their cells.",
        ),
        (
            "🌱 Mesophiles (Moderate-temperature life)",
            "288–318 K / ~15–45°C",
            "Many familiar Earth organisms, including humans and plants, are mesophiles. "
            "This broad temperature band is a comparison guide, not a promise of comfortable "
            "conditions across the whole range.",
        ),
        (
            "🔥 Thermophiles (Heat-lovers)",
            "> 318 K / >45°C",
            "Some microbes flourish in hot springs and volcanic seafloor vents, using "
            "heat-stable proteins and membranes to keep functioning.",
        ),
    )
    st.markdown(
        '<div class="science-card-grid">'
        + "".join(
            '<section class="science-card extremophile-card">'
            f'<div class="science-card-title"><strong>{escape(name)}</strong></div>'
            f'<div class="science-card-temperature"><strong>Temperature:</strong> '
            f'<em>{escape(temperature)}</em></div>'
            f'<div class="science-card-definition"><strong>Definition:</strong> '
            f"{escape(definition)}</div>"
            "</section>"
            for name, temperature, definition in extremophile_cards
        )
        + "</div>",
        unsafe_allow_html=True,
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
        "pl_bmasse": "Planet mass (Earth masses)",
        "pl_orbsmax": "Semi-major axis (AU)",
        "st_teff": "Host-star effective temperature (K)",
        "discoverymethod": "Discovery method",
        "disc_year": "Discovery year",
        "Thermal regime (proxy)": "Thermal regime",
        "Liquid water screening score (%)": "Liquid-water potential score (%)",
        "Stellar flux regime (proxy)": "Stellar flux range",
    }
    available_columns = [column for column in display_columns if column in filtered.columns]
    results = filtered[available_columns].rename(columns=display_columns)
    with st.expander("⬇ Download matching planets as CSV"):
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
    equilibrium_temperature = float(planet["pl_eqt"])
    radius_earth = float(planet["pl_rade"])
    temperature_delta = equilibrium_temperature - EARTH_EQUILIBRIUM_TEMPERATURE_K
    radius_delta = radius_earth - 1
    esi = calculate_esi_proxy(radius_earth, equilibrium_temperature)
    measured_mass = float(planet.get("pl_bmasse", float("nan")))
    planet_mass_earth, mass_basis = estimate_planet_mass(radius_earth, measured_mass)
    surface_gravity, escape_velocity = estimate_surface_gravity_and_escape(
        planet_mass_earth, radius_earth
    )
    structure_regime = classify_radius_regime(radius_earth)
    stellar_flux = float(planet.get("pl_insol", float("nan")))
    stellar_temperature = float(planet.get("st_teff", float("nan")))
    orbital_distance = float(planet.get("pl_orbsmax", float("nan")))
    right_ascension = planet.get("ra", float("nan"))
    declination = planet.get("dec", float("nan"))
    distance_parsecs = planet.get("sy_dist", float("nan"))
    water_score = float(planet["Liquid water screening score (%)"])
    thermal_regime = str(planet["Thermal regime (proxy)"])

    st.markdown(f"#### {planet['pl_name']}")
    detail_row_one = st.columns(3)
    detail_row_two = st.columns(3)
    detail_row_one[0].metric(
        "🌡️ Equilibrium temperature",
        f"{equilibrium_temperature:.1f} K",
        f"{temperature_delta:+.1f} K vs Earth ({EARTH_EQUILIBRIUM_TEMPERATURE_K} K)",
    )
    detail_row_one[1].metric(
        "📏 Planetary radius",
        f"{radius_earth:.2f} Earth radii",
        f"{radius_delta:+.2f} Earth radii vs Earth",
    )
    detail_row_one[2].metric("🧬 Temperature zone", thermal_regime)
    detail_row_two[0].metric("💧 Liquid-water potential score", f"{water_score:.1f}%")
    detail_row_two[1].metric(
        "🌍 Earth Similarity Index (2-parameter)",
        f"{esi:.3f}" if math.isfinite(esi) else "Unavailable",
        classify_esi(esi) if math.isfinite(esi) else "Unavailable",
    )
    detail_row_two[2].metric(
        "☀️ Stellar flux",
        f"{stellar_flux:.2f} Earth Fluxes" if math.isfinite(stellar_flux) else "Unavailable",
        classify_insolation(stellar_flux) if math.isfinite(stellar_flux) else "Unavailable",
    )

    temperature_celsius = equilibrium_temperature - WATER_FREEZING_POINT_K
    freezing_point_delta = equilibrium_temperature - WATER_FREEZING_POINT_K
    freezing_point_relation = (
        "above" if freezing_point_delta >= 0 else "below"
    )
    g_earth = (
        surface_gravity / EARTH_GRAVITY_M_S2
        if math.isfinite(surface_gravity)
        else float("nan")
    )
    size_regime = {
        "Rocky terrestrial size": "Rocky terrestrial planet",
        "Super-Earth size": "Super-Earth",
        "Mini-Neptune candidate": "Mini-Neptune candidate",
    }.get(structure_regime, structure_regime)
    mass_source = (
        "measured in the NASA archive"
        if mass_basis.startswith("NASA")
        else "estimated from an empirical radius–mass relation"
    )
    extremophile_adaptation = {
        "Psychrophile": (
            "antifreeze proteins, flexible membranes, and cold-adapted enzymes"
        ),
        "Mesophile": (
            "membrane-lipid regulation and enzymes tuned to moderate temperatures"
        ),
        "Thermophile": (
            "heat-stable protein folds, compact hydrophobic cores, and resilient membranes"
        ),
    }[thermal_regime]

    esi_display = f"{esi:.3f}" if math.isfinite(esi) else "Unavailable"
    gravity_display = (
        f"{surface_gravity:.2f} m/s²" if math.isfinite(surface_gravity) else "Unavailable"
    )
    gravity_earth_display = f"{g_earth:.2f}" if math.isfinite(g_earth) else "Unavailable"
    escape_display = (
        f"{escape_velocity:.2f} km/s"
        if math.isfinite(escape_velocity)
        else "Unavailable"
    )
    mass_display = (
        f"{planet_mass_earth:.2f} M⊕"
        if math.isfinite(planet_mass_earth)
        else "Unavailable"
    )
    temperature_display = (
        f"{equilibrium_temperature:.1f} K ({temperature_celsius:.1f} °C)"
    )
    freezing_point_display = (
        f"{abs(freezing_point_delta):.1f} K {freezing_point_relation}"
    )
    terrestrial_radius_range_display = (
        f"{TERRESTRIAL_RADIUS_RANGE_EARTH[0]:.1f}–"
        f"{TERRESTRIAL_RADIUS_RANGE_EARTH[1]:.1f} R⊕"
    )
    breakdown_items = (
        (
            "Earth Similarity Index (ESI)",
            "Measures how closely a world resembles Earth in size and temperature on a "
            "0 to 1 scale, where 1.0 is an Earth twin. This planet scores "
            f"{format_unit_badge(esi_display)} ({escape(classify_esi(esi))}), calibrated "
            f"against Earth's reference radius ({format_unit_badge('1.0 R⊕')}) and "
            f"baseline equilibrium temperature "
            f"({format_unit_badge(f'{EARTH_EQUILIBRIUM_TEMPERATURE_K} K')}).",
        ),
        (
            "Thermal & Radiative Balance",
            "Based on incoming starlight alone, this world has an equilibrium temperature "
            f"of {format_unit_badge(temperature_display)}, sitting "
            f"{format_unit_badge(freezing_point_display)} water's freezing point "
            f"({format_unit_badge(f'{WATER_FREEZING_POINT_K:.2f} K')}). True surface "
            "conditions depend on greenhouse gases: a dense atmosphere traps heat and "
            "raises surface temperatures, while a thin atmosphere leaves the surface "
            "exposed to space.",
        ),
        (
            "Gravity & Planetary Structure",
            f"With a radius of {format_unit_badge(f'{radius_earth:.2f} R⊕')} and a mass of "
            f"{format_unit_badge(mass_display)} ({escape(mass_source)}), this world falls "
            f"into the <strong>{escape(size_regime)}</strong> class. It generates surface "
            f"gravity of {format_unit_badge(gravity_display)} "
            f"({format_unit_badge(f'{gravity_earth_display}× Earth gravity')}) and an "
            f"escape velocity of {format_unit_badge(escape_display)}. Together with "
            "atmospheric temperature and composition, these values help determine how "
            "readily gases can escape over geological time.",
        ),
        (
            "Extremophile Biocompatibility",
            f"This world falls in the <strong>{escape(thermal_regime)}</strong> thermal "
            "zone. On Earth, organisms in this temperature range rely on specialized "
            f"molecular adaptations—{escape(extremophile_adaptation)}—to keep cellular "
            "membranes flexible and proteins properly folded.",
        ),
        (
            "Liquid-Water Potential Score",
            "Compares the planet's estimated temperature and size with a simple range "
            "associated with water-friendly conditions, centered on "
            f"{format_unit_badge(f'{LIQUID_WATER_OPTIMAL_TEMPERATURE_K} K / 15 °C')} and "
            f"{format_unit_badge(terrestrial_radius_range_display)}. "
            f"This candidate scores {format_unit_badge(f'{water_score:.1f}%')}.",
        ),
    )
    st.markdown("**🧬 Biological compatibility breakdown**")
    st.markdown(
        '<div class="compatibility-list">'
        + "".join(
            '<div class="compatibility-item">'
            f'<strong class="compatibility-title">{escape(title)}</strong>'
            f'<div class="compatibility-body">{body}</div>'
            "</div>"
            for title, body in breakdown_items
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### ☀️ Stellar Radiation & Orbital Stability")
    tidal_status, tidal_detail = assess_tidal_and_stripping_proxy(
        stellar_temperature,
        orbital_distance,
        stellar_flux,
        escape_velocity,
    )
    radiation_cards = (
        format_science_card(
            "☀️ Incident Starlight Flux",
            (
                f"{stellar_flux:.2f} Earth fluxes (S⊕)"
                if math.isfinite(stellar_flux)
                else "Unavailable"
            ),
            (
                "Measures the star's total energy reaching the planet; Earth receives "
                "1.0 Earth flux."
                if math.isfinite(stellar_flux)
                else "A starlight measurement is not available for this planet."
            ),
            (
                classify_insolation(stellar_flux)
                if math.isfinite(stellar_flux)
                else "Habitable-zone comparison unavailable"
            ),
        ),
        format_science_card(
            "🪐 Surface Gravity & Escape Speed",
            (
                f"{surface_gravity:.2f} m/s²"
                if math.isfinite(surface_gravity) and math.isfinite(escape_velocity)
                else "Unavailable"
            ),
            "Gravity and escape speed help describe how strongly a world can hold gas; "
            "atmospheric composition also matters.",
            (
                f"Escape speed: {escape_velocity:.2f} km/s"
                if math.isfinite(escape_velocity)
                else "Escape speed unavailable"
            ),
            secondary_is_badge=True,
        ),
        format_science_card(
            "🌀 Orbital & Atmospheric Stability",
            tidal_status,
            "Close orbits can synchronize a planet's spin. Strong starlight and lower escape "
            "speed can make an atmosphere harder to retain.",
            tidal_detail,
        ),
    )
    st.markdown(
        '<div class="science-card-grid">' + "".join(radiation_cards) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🚀 Cosmic Distance & Travel Time")
    if pd.notna(distance_parsecs) and distance_parsecs >= 0:
        distance_light_years, light_speed_years, voyager_years = calculate_travel_times(
            float(distance_parsecs)
        )
        travel_cards = (
            format_science_card(
                "📍 System Distance",
                f"{distance_light_years:,.2f} light-years",
                "The estimated distance between Earth and the planet's host system.",
            ),
            format_science_card(
                "✨ Light-Speed Transit",
                format_years(light_speed_years),
                "One-way light-travel time from Earth; this is the fastest possible signal delay.",
            ),
            format_science_card(
                "🛰️ Voyager 1 Spacecraft Transit",
                format_years(voyager_years),
                "A straight-line estimate at Voyager 1's speed; it ignores acceleration, "
                "braking, and propulsion limits.",
                f"Voyager 1 speed: {VOYAGER_1_SPEED_KM_PER_HOUR:,} km/h",
                secondary_is_badge=True,
            ),
        )
    else:
        travel_cards = (
            format_science_card(
                "📍 System Distance",
                "Unavailable",
                "A distance measurement is not available for this host system.",
            ),
            format_science_card(
                "✨ Light-Speed Transit",
                "Unavailable",
                "A one-way light-travel time needs a measured system distance.",
            ),
            format_science_card(
                "🛰️ Voyager 1 Spacecraft Transit",
                "Unavailable",
                "A Voyager-speed estimate needs a measured system distance.",
            ),
        )
    st.markdown(
        '<div class="science-card-grid">' + "".join(travel_cards) + "</div>",
        unsafe_allow_html=True,
    )

    radiation_status = (
        f"Bolometric flux {stellar_flux:.2f} times Earth's "
        f"({classify_insolation(stellar_flux)}); surface radiation dose and human safety "
        "are not determined by bolometric flux."
        if math.isfinite(stellar_flux)
        else "Stellar flux, surface radiation dose, and human safety are unavailable or "
        "unknown from the available data."
    )
    dossier_state_key = "prepared_dossier_planet"
    if st.button("🛰️ Prepare Planetary Mission Dossier", key="prepare_planetary_dossier"):
        st.session_state[dossier_state_key] = selected_name
    if st.session_state.get(dossier_state_key) == selected_name:
        safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", selected_name).strip("._")
        with st.spinner("Preparing the planet-specific mission dossier..."):
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
                esi=esi,
                structure_regime=structure_regime,
                planet_mass_earth=planet_mass_earth,
                mass_basis=mass_basis,
                surface_gravity_m_s2=surface_gravity,
                escape_velocity_km_s=escape_velocity,
                stellar_temperature_k=stellar_temperature,
                orbital_distance_au=orbital_distance,
                distance_parsecs=distance_parsecs,
            )
        st.download_button(
            label="📥 Download Planetary Mission Dossier (PDF)",
            data=pdf_data,
            file_name=f"{safe_filename or 'planet'}_Planetary_Dossier.pdf",
            mime="application/pdf",
        )

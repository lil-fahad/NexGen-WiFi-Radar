"""
Wi-Fi CSI MIMO Radar Tactical Dashboard
========================================
Streamlit-based interactive dashboard for real-time radar visualization
and target tracking using Channel State Information (CSI) analysis.

This application provides a defense-grade UI for:
- 2D Polar radar visualization with MUSIC spectrum
- 3D spatial terrain mapping
- Micro-Doppler vital signs monitoring
- Real-time target parameter estimation

Author: Senior Radar Systems Engineer & Principal Python Architect
Version: 1.0.0
"""

from typing import Tuple
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import the radar engine
from engine import RadarEngine


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="NexGen WiFi Radar | CSI MIMO",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS STYLING - TACTICAL DARK THEME
# ============================================================================

TACTICAL_CSS = """
<style>
    /* Import monospace font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');

    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #00ff88;
        font-family: 'Roboto Mono', monospace;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #1a1f3a 100%);
        border-right: 2px solid #00ffcc;
    }

    [data-testid="stSidebar"] * {
        color: #00ffcc !important;
        font-family: 'Roboto Mono', monospace;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00ffcc !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
        font-weight: 700;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
    }

    [data-testid="stMetricLabel"] {
        color: #66d9ef !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem !important;
    }

    [data-testid="stMetricDelta"] {
        color: #f92672 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d1117;
        border-radius: 4px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f3a;
        color: #00ffcc;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: 600;
        border: 1px solid #00ffcc44;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ffcc 0%, #00ff88 100%);
        color: #0a0e27 !important;
        border: 1px solid #00ffcc;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.4);
    }

    /* Sliders */
    .stSlider > div > div > div {
        background-color: #00ffcc !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00ffcc 0%, #00ff88 100%);
        color: #0a0e27;
        border: none;
        border-radius: 4px;
        padding: 8px 24px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.6);
        transform: translateY(-2px);
    }

    /* Info boxes */
    .stAlert {
        background-color: #1a1f3a;
        border: 1px solid #00ffcc;
        border-radius: 4px;
        color: #00ffcc;
    }

    /* Dividers */
    hr {
        border-color: #00ffcc44;
    }
</style>
"""

st.markdown(TACTICAL_CSS, unsafe_allow_html=True)


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def initialize_radar_engine() -> RadarEngine:
    """
    Initialize and cache the radar engine to prevent re-initialization.

    Returns:
        Configured RadarEngine instance
    """
    return RadarEngine(num_antennas=8, frequency_ghz=5.0, antenna_spacing_lambda=0.5)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_polar_radar_plot(
    ranges: np.ndarray,
    angles: np.ndarray,
    spectrum_db: np.ndarray,
    target_range: float,
    target_angle: float
) -> go.Figure:
    """
    Create a 2D polar contour plot of the MUSIC spatial spectrum.

    Args:
        ranges: 1D array of range values (meters)
        angles: 1D array of angle values (degrees)
        spectrum_db: 2D spectrum array in dB
        target_range: True target range for marking
        target_angle: True target angle for marking

    Returns:
        Plotly figure object
    """
    # Convert to polar coordinates for plotting
    # Angles in degrees, radius is range
    theta_rad = np.deg2rad(angles)

    # Create meshgrid for contour
    R_grid, Theta_grid = np.meshgrid(ranges, angles, indexing='ij')

    fig = go.Figure()

    # Add contour trace
    fig.add_trace(go.Scatterpolar(
        r=R_grid.flatten(),
        theta=Theta_grid.flatten(),
        mode='markers',
        marker=dict(
            color=spectrum_db.flatten(),
            colorscale='Turbo',
            size=4,
            colorbar=dict(
                title="dB",
                titleside="right",
                tickmode="linear",
                tick0=-40,
                dtick=10,
                titlefont=dict(color='#00ffcc'),
                tickfont=dict(color='#00ffcc')
            ),
            cmin=-40,
            cmax=0
        ),
        showlegend=False,
        hovertemplate='Range: %{r:.2f}m<br>Angle: %{theta}°<br>Power: %{marker.color:.1f}dB<extra></extra>'
    ))

    # Add target marker
    fig.add_trace(go.Scatterpolar(
        r=[target_range],
        theta=[target_angle],
        mode='markers+text',
        marker=dict(
            color='#ff0000',
            size=15,
            symbol='x',
            line=dict(color='#ffffff', width=2)
        ),
        text=['🎯 TARGET'],
        textposition='top center',
        textfont=dict(color='#ff0000', size=12, family='Roboto Mono'),
        showlegend=False,
        hovertemplate='Target<br>Range: %{r:.2f}m<br>Angle: %{theta}°<extra></extra>'
    ))

    # Layout configuration
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(10, 14, 39, 0.8)',
            radialaxis=dict(
                visible=True,
                range=[0, np.max(ranges)],
                showline=True,
                linecolor='#00ffcc',
                gridcolor='#00ffcc33',
                tickfont=dict(color='#00ffcc'),
                title=dict(text='Range (m)', font=dict(color='#00ffcc'))
            ),
            angularaxis=dict(
                visible=True,
                direction='clockwise',
                period=360,
                showline=True,
                linecolor='#00ffcc',
                gridcolor='#00ffcc33',
                tickfont=dict(color='#00ffcc'),
                rotation=90
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Roboto Mono', color='#00ffcc'),
        title=dict(
            text='<b>MUSIC SPATIAL SPECTRUM - POLAR VIEW</b>',
            font=dict(size=16, color='#00ffcc'),
            x=0.5,
            xanchor='center'
        ),
        height=600
    )

    return fig


def create_3d_terrain_plot(
    ranges: np.ndarray,
    angles: np.ndarray,
    spectrum_db: np.ndarray
) -> go.Figure:
    """
    Create a 3D surface plot of the MUSIC spatial spectrum.

    Args:
        ranges: 1D array of range values (meters)
        angles: 1D array of angle values (degrees)
        spectrum_db: 2D spectrum array in dB

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=angles,
        y=ranges,
        z=spectrum_db,
        colorscale='Turbo',
        colorbar=dict(
            title='dB',
            titleside='right',
            tickmode='linear',
            tick0=-40,
            dtick=10,
            titlefont=dict(color='#00ffcc'),
            tickfont=dict(color='#00ffcc')
        ),
        hovertemplate='Angle: %{x}°<br>Range: %{y:.2f}m<br>Power: %{z:.1f}dB<extra></extra>'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Angle (degrees)',
                backgroundcolor='rgba(10, 14, 39, 0.8)',
                gridcolor='#00ffcc33',
                showbackground=True,
                titlefont=dict(color='#00ffcc'),
                tickfont=dict(color='#00ffcc')
            ),
            yaxis=dict(
                title='Range (meters)',
                backgroundcolor='rgba(10, 14, 39, 0.8)',
                gridcolor='#00ffcc33',
                showbackground=True,
                titlefont=dict(color='#00ffcc'),
                tickfont=dict(color='#00ffcc')
            ),
            zaxis=dict(
                title='Power (dB)',
                backgroundcolor='rgba(10, 14, 39, 0.8)',
                gridcolor='#00ffcc33',
                showbackground=True,
                titlefont=dict(color='#00ffcc'),
                tickfont=dict(color='#00ffcc')
            ),
            bgcolor='rgba(10, 14, 39, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Roboto Mono', color='#00ffcc'),
        title=dict(
            text='<b>MUSIC 3D SPATIAL TERRAIN</b>',
            font=dict(size=16, color='#00ffcc'),
            x=0.5,
            xanchor='center'
        ),
        height=600
    )

    return fig


def create_vital_signs_plot(
    time_array: np.ndarray,
    amplitude_array: np.ndarray,
    is_moving: bool
) -> go.Figure:
    """
    Create a time-series plot of vital signs data.

    Args:
        time_array: Time samples in seconds
        amplitude_array: Normalized amplitude signal
        is_moving: Target movement state

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Determine plot color based on target state
    line_color = '#ff6b6b' if is_moving else '#00ff88'
    fill_color = 'rgba(255, 107, 107, 0.2)' if is_moving else 'rgba(0, 255, 136, 0.2)'

    fig.add_trace(go.Scatter(
        x=time_array,
        y=amplitude_array,
        mode='lines',
        line=dict(color=line_color, width=2),
        fill='tozeroy',
        fillcolor=fill_color,
        name='Vital Signs',
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))

    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='#00ffcc44',
        annotation_text='Baseline',
        annotation_position='right',
        annotation_font=dict(color='#00ffcc', size=10)
    )

    title_text = '<b>MICRO-DOPPLER SIGNATURE - MOTION DETECTED</b>' if is_moving else '<b>MICRO-DOPPLER SIGNATURE - VITAL SIGNS</b>'

    fig.update_layout(
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridcolor='#00ffcc22',
            zeroline=True,
            zerolinecolor='#00ffcc44',
            titlefont=dict(color='#00ffcc'),
            tickfont=dict(color='#00ffcc')
        ),
        yaxis=dict(
            title='Normalized Amplitude',
            showgrid=True,
            gridcolor='#00ffcc22',
            zeroline=True,
            zerolinecolor='#00ffcc44',
            titlefont=dict(color='#00ffcc'),
            tickfont=dict(color='#00ffcc')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10, 14, 39, 0.6)',
        font=dict(family='Roboto Mono', color='#00ffcc'),
        title=dict(
            text=title_text,
            font=dict(size=16, color='#00ffcc'),
            x=0.5,
            xanchor='center'
        ),
        hovermode='x unified',
        height=500,
        showlegend=False
    )

    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application entry point."""

    # Initialize radar engine (cached)
    engine = initialize_radar_engine()

    # ========================================================================
    # HEADER
    # ========================================================================

    st.markdown(
        """
        <h1 style='text-align: center; margin-bottom: 0;'>
            📡 NEXGEN WIFI CSI MIMO RADAR
        </h1>
        <p style='text-align: center; color: #66d9ef; margin-top: 0; letter-spacing: 2px;'>
            THROUGH-WALL HUMAN TARGET DETECTION SYSTEM
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================

    st.sidebar.markdown("## 🎛️ RADAR PARAMETERS")
    st.sidebar.markdown("---")

    # Target range slider
    target_range = st.sidebar.slider(
        "Target Range (meters)",
        min_value=0.5,
        max_value=10.0,
        value=3.5,
        step=0.1,
        help="Distance to target in meters"
    )

    # Target angle slider
    target_angle = st.sidebar.slider(
        "Target Angle (degrees)",
        min_value=-60,
        max_value=60,
        value=15,
        step=1,
        help="Angular position relative to array boresight"
    )

    # Target state selector
    target_state = st.sidebar.radio(
        "Target State",
        options=["Static", "Moving"],
        index=0,
        help="Static: Vital signs monitoring | Moving: Motion detection"
    )
    is_moving = (target_state == "Moving")

    # SNR slider
    snr_db = st.sidebar.slider(
        "SNR (dB)",
        min_value=5,
        max_value=30,
        value=20,
        step=1,
        help="Signal-to-Noise Ratio in decibels"
    )

    # Multipath toggle
    multipath_enabled = st.sidebar.checkbox(
        "Enable Multipath Fading",
        value=False,
        help="Simulate wall/floor reflections (ghost targets)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 0.75rem; color: #66d9ef;'>
            <b>SYSTEM STATUS</b><br>
            🟢 OPERATIONAL<br>
            Antenna Array: 8 Elements<br>
            Frequency: 5.0 GHz<br>
            λ/2 Spacing
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================================
    # COMPUTE RADAR DATA
    # ========================================================================

    # Generate MUSIC spectrum
    ranges, angles, spectrum_db = engine.advanced_music_spectrum(
        target_r=target_range,
        target_a=target_angle,
        snr_db=snr_db,
        multipath=multipath_enabled
    )

    # Extract vital signs
    time_array, vital_signal = engine.extract_vital_signs(
        is_moving=is_moving,
        duration=10.0,
        fs=100.0
    )

    # Estimate vital rates
    breathing_rate, heart_rate = engine.estimate_vital_rates(time_array, vital_signal)

    # ========================================================================
    # KPI METRICS
    # ========================================================================

    st.markdown("## 📊 TARGET ACQUISITION METRICS")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Estimated Range",
            value=f"{target_range:.2f} m",
            delta=None
        )

    with col2:
        st.metric(
            label="Estimated Angle",
            value=f"{target_angle}°",
            delta=None
        )

    with col3:
        vital_status = "MOVING" if is_moving else "STATIC"
        status_delta = "Motion Detected" if is_moving else "Vital Signs Active"
        st.metric(
            label="Target Status",
            value=vital_status,
            delta=status_delta
        )

    with col4:
        if not is_moving:
            st.metric(
                label="Heart Rate",
                value=f"{heart_rate:.0f} BPM",
                delta=None
            )
        else:
            st.metric(
                label="Motion Level",
                value="HIGH",
                delta="Broadband"
            )

    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if not is_moving:
            st.metric(
                label="Breathing Rate",
                value=f"{breathing_rate:.0f} BPM",
                delta=None
            )
        else:
            st.metric(
                label="Doppler Spread",
                value="2-12 Hz",
                delta=None
            )

    with col6:
        st.metric(
            label="SNR",
            value=f"{snr_db} dB",
            delta=None
        )

    with col7:
        multipath_status = "ENABLED" if multipath_enabled else "DISABLED"
        st.metric(
            label="Multipath",
            value=multipath_status,
            delta=None
        )

    with col8:
        detection_confidence = min(100, snr_db * 3.33)
        st.metric(
            label="Confidence",
            value=f"{detection_confidence:.0f}%",
            delta=None
        )

    st.markdown("---")

    # ========================================================================
    # VISUALIZATION TABS
    # ========================================================================

    tab1, tab2, tab3 = st.tabs([
        "📡 2D POLAR RADAR",
        "🗻 3D SPATIAL TERRAIN",
        "💓 MICRO-DOPPLER VITALS"
    ])

    with tab1:
        st.markdown("### 2D Polar MUSIC Spectrum")
        st.markdown(
            "<p style='color: #66d9ef; font-size: 0.9rem;'>"
            "High-resolution Direction of Arrival (DoA) estimation using eigenspace analysis. "
            "Peak intensity indicates target location."
            "</p>",
            unsafe_allow_html=True
        )

        polar_fig = create_polar_radar_plot(
            ranges, angles, spectrum_db, target_range, target_angle
        )
        st.plotly_chart(polar_fig, use_container_width=True)

    with tab2:
        st.markdown("### 3D Spatial Power Distribution")
        st.markdown(
            "<p style='color: #66d9ef; font-size: 0.9rem;'>"
            "Surface representation of signal power across range-angle space. "
            "Sharp peaks indicate target positions in 3D."
            "</p>",
            unsafe_allow_html=True
        )

        terrain_fig = create_3d_terrain_plot(ranges, angles, spectrum_db)
        st.plotly_chart(terrain_fig, use_container_width=True)

    with tab3:
        st.markdown("### Micro-Doppler Signature Analysis")

        if is_moving:
            st.markdown(
                "<p style='color: #ff6b6b; font-size: 0.9rem;'>"
                "⚠️ <b>MOTION DETECTED:</b> Broadband Doppler signature indicates active target movement. "
                "Chaotic signal pattern consistent with walking or limb motion."
                "</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<p style='color: #00ff88; font-size: 0.9rem;'>"
                "✓ <b>STATIC TARGET:</b> Periodic signal components detected. "
                f"Breathing: {breathing_rate:.0f} BPM | Heartbeat: {heart_rate:.0f} BPM"
                "</p>",
                unsafe_allow_html=True
            )

        vital_fig = create_vital_signs_plot(time_array, vital_signal, is_moving)
        st.plotly_chart(vital_fig, use_container_width=True)

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #66d9ef; font-size: 0.8rem;'>
            <b>NexGen WiFi CSI MIMO Radar System v1.0</b><br>
            Advanced Through-Wall Human Detection | MUSIC Algorithm | Micro-Doppler Analysis<br>
            © 2026 | Tactical Defense Systems
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

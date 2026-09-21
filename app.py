"""
EcoPower AI – AI-Powered Energy Consumption Prediction & Sustainability Advisor
================================================================================
Author: Vanshika Verma | B.Tech CSE – AI, Om Sterling Global University, Hisar
Internship: 1M1B AI for Sustainability Virtual Internship (IBM SkillsBuild & AICTE)
SDG Alignment: SDG 7 (Affordable & Clean Energy) | SDG 13 (Climate Action)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src.data_preprocessing import load_data, clean_data, validate_schema
from src.feature_engineering import engineer_features, FEATURE_COLUMNS
from src.prediction import EnergyPredictor
from src.rag_pipeline import SustainabilityRAG

# Page Configuration
st.set_page_config(
    page_title="EcoPower AI – Smart Energy & Sustainability Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background & container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Hero Header Banner */
    .hero-header {
        background: linear-gradient(135deg, #022c22 0%, #064e3b 45%, #065f46 80%, #047857 100%);
        border-radius: 20px;
        padding: 36px 40px;
        color: white;
        margin-bottom: 28px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 15px 35px -5px rgba(2, 44, 34, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-header::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.15) 0%, rgba(6, 78, 59, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 14px;
    }
    .badge-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-sdg7 {
        background: #fef08a;
        color: #713f12;
        border: 1px solid #fde047;
    }
    .badge-sdg13 {
        background: #bbf7d0;
        color: #14532d;
        border: 1px solid #86efac;
    }
    .badge-ibm {
        background: #dbeafe;
        color: #1e3a8a;
        border: 1px solid #bfdbfe;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: -0.8px;
        line-height: 1.2;
        color: #ffffff;
        margin-bottom: 8px;
    }
    .hero-desc {
        font-size: 15px;
        color: #a7f3d0;
        font-weight: 400;
        line-height: 1.6;
        max-width: 820px;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 22px 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
        position: relative;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .metric-card-accent {
        position: absolute;
        top: 0;
        left: 20px;
        right: 20px;
        height: 3px;
        background: #10b981;
        border-radius: 3px 3px 0 0;
    }
    .metric-card-accent-red {
        background: #ef4444;
    }
    .metric-card-accent-blue {
        background: #3b82f6;
    }
    .metric-card-accent-teal {
        background: #0d9488;
    }
    .metric-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .metric-num {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-unit {
        font-size: 15px;
        font-weight: 600;
        color: #64748b;
    }
    .metric-subtext {
        font-size: 12px;
        color: #059669;
        font-weight: 600;
        margin-top: 6px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* Explain / Takeaway Card */
    .simple-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #10b981;
        margin-top: 14px;
        font-size: 13.5px;
        color: #334155;
        line-height: 1.6;
    }

    /* Peak & Off-Peak Banners */
    .banner-peak {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 14px 18px;
        color: #991b1b;
        font-size: 13.5px;
        line-height: 1.5;
    }
    .banner-offpeak {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 14px 18px;
        color: #166534;
        font-size: 13.5px;
        line-height: 1.5;
    }

    /* Action Cards in Advisor */
    .action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .action-tag-immediate {
        background: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .action-tag-policy {
        background: #e0e7ff;
        color: #3730a3;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .action-tag-tech {
        background: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-right: 6px;
    }

    /* Citation Box */
    .citation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0d9488;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 12.5px;
    }

    /* Sidebar Clean Styling - NO SNAPSHOTS */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-brand {
        padding: 10px 0 16px 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .sidebar-status {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 12px;
        color: #166534;
        margin-bottom: 18px;
        line-height: 1.4;
    }
    .sidebar-profile {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 20px;
        font-size: 12.5px;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Cache Resources
@st.cache_resource
def get_predictor():
    return EnergyPredictor()

@st.cache_resource
def get_rag_engine():
    return SustainabilityRAG()

@st.cache_data
def get_dataset():
    data_path = os.path.join(APP_DIR, "data", "energy_consumption.csv")
    df = load_data(data_path)
    df = clean_data(df)
    return df

# Initialize core systems
try:
    predictor = get_predictor()
    rag_engine = get_rag_engine()
    df_main = get_dataset()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR NAVIGATION (CLEAN, NO SNAPSHOTS, NO BULKY IMAGES)
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size: 20px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 24px; color: #059669;">⚡</span> EcoPower AI
        </div>
        <div style="font-size: 11px; font-weight: 700; color: #059669; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px;">
            Sustainability & Energy Intelligence
        </div>
    </div>
    <div class="sidebar-status">
        🟢 <b>System Status: Online</b><br/>
        • Gradient Boosting (R² = 0.9839)<br/>
        • RAG Vector Store (5 Standards)
    </div>
    """, unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Navigation Menu",
        [
            "🏠 Overview & Mission",
            "📊 Executive Energy KPIs",
            "📈 Visual Consumption Analytics",
            "🔮 Interactive Prediction Lab",
            "🌱 AI Sustainability Advisor",
            "📚 Sustainability Standards KB",
            "🛡️ Responsible AI Charter",
            "👩‍🎓 Author & Internship Credentials"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div class="sidebar-profile">
        <div style="font-weight: 700; color: #0f172a; font-size: 13.5px; margin-bottom: 2px;">👩‍🎓 Vanshika Verma</div>
        <div style="color: #059669; font-weight: 600; font-size: 11.5px; margin-bottom: 8px;">Aspiring Data Scientist</div>
        <div style="line-height: 1.45; font-size: 11.5px; color: #475569;">
            <b>B.Tech CSE (Artificial Intelligence)</b><br/>
            Om Sterling Global University, Hisar<br/>
            🌿 <b>1M1B Virtual Internship</b><br/>
            <i>IBM SkillsBuild & AICTE</i>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 1. OVERVIEW & MISSION
# ==============================================================================
if nav_selection == "🏠 Overview & Mission":
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge-row">
            <span class="badge-chip badge-sdg7">⚡ SDG 7: Clean Energy</span>
            <span class="badge-chip badge-sdg13">🌍 SDG 13: Climate Action</span>
            <span class="badge-chip badge-ibm">🤖 Co-Developed with IBM Bob</span>
        </div>
        <div class="hero-title">EcoPower AI</div>
        <div class="hero-desc">
            A high-precision predictive and advisory platform designed to reduce electricity waste, 
            shave peak demand surcharges, and guide institutions toward evidence-backed sustainability practices.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3 High-Impact Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div style="font-size: 26px; margin-bottom: 6px;">🎯</div>
            <div style="font-weight: 800; font-size: 16px; color: #0f172a;">98.4% Forecast Accuracy</div>
            <div style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                Trained on 8,784 hourly intervals using a <b>chronological 80/20 split</b> with zero temporal data leakage.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-blue"></div>
            <div style="font-size: 26px; margin-bottom: 6px;">🏛️</div>
            <div style="font-weight: 800; font-size: 16px; color: #0f172a;">Authoritative RAG Engine</div>
            <div style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                Retrieves verified energy conservation mandates from <b>BEE India, IEA Paris, and UN SDG frameworks</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-teal"></div>
            <div style="font-size: 26px; margin-bottom: 6px;">🛡️</div>
            <div style="font-weight: 800; font-size: 16px; color: #0f172a;">Responsible AI Guardrails</div>
            <div style="font-size: 12.5px; color: #64748b; margin-top: 6px; line-height: 1.5;">
                Audited against hallucinations, hazardous electrical advice refusal, and zero personal data collection.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧩 System Architecture: How It Works")
    st.markdown("""
    ```mermaid
    flowchart LR
        subgraph Input
            Data[Hourly Sensor Telemetry]
            Query[Sustainability Question]
        end
        
        subgraph Analytics Layer
            Data --> FeatEng[Physics Feature Engineering: CDD / Cyclics]
            FeatEng --> MLModel[Gradient Boosting Regressor: R2=0.9839]
        end
        
        subgraph Knowledge Layer
            Query --> VectorSearch[Semantic Vector Search]
            VectorSearch --> KB[(BEE India / IEA / UN Guidelines)]
        end
        
        MLModel --> Advisor[AI Sustainability Advisor: IBM Bob Grounded Engine]
        KB --> Advisor
        Advisor --> Output[Actionable Low-Carbon Roadmap]
    ```
    """)
    
    st.markdown("""
    <div class="simple-box">
        💡 <b>The Real-World Problem Solved in Simple Terms:</b> Most institutions only discover high electricity bills <i>after</i> the month is over. 
        EcoPower AI acts like a smart weather radar for electricity: it predicts demand surges <b>before they happen</b>, 
        flags expensive peak tariff windows (10:00–16:00), and tells facility managers exactly what to adjust (e.g. raising AC setpoints to 24°C saves ~6% per degree).
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 2. EXECUTIVE ENERGY KPIS
# ==============================================================================
elif nav_selection == "📊 Executive Energy KPIs":
    st.markdown("## 📊 Executive Energy Overview")
    st.markdown("Annual electricity statistics derived from 1 full year of continuous facility telemetry (8,784 hourly intervals).")
    
    tot_kwh = df_main['energy_consumption_kwh'].sum()
    tot_mwh = tot_kwh / 1000.0
    avg_kwh = df_main['energy_consumption_kwh'].mean()
    peak_kwh = df_main['energy_consumption_kwh'].max()
    carbon_tons = (tot_kwh * 0.78) / 1000.0  # Northern India grid avg ~0.78 kg/kWh
    peak_mwh = df_main[df_main['peak_hours_flag'] == 1]['energy_consumption_kwh'].sum() / 1000.0
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent"></div>
            <div class="metric-title">Annual Electricity Consumption</div>
            <div class="metric-num">{tot_mwh:,.1f} <span class="metric-unit">MWh</span></div>
            <div class="metric-subtext">⚡ 8,784 Hourly Records</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-teal"></div>
            <div class="metric-title">Average Hourly Demand</div>
            <div class="metric-num">{avg_kwh:.1f} <span class="metric-unit">kWh</span></div>
            <div class="metric-subtext">🏢 Typical Baseload</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-red"></div>
            <div class="metric-title">Peak Demand Spike</div>
            <div class="metric-num" style="color:#b91c1c;">{peak_kwh:.1f} <span class="metric-unit">kWh</span></div>
            <div class="metric-subtext" style="color:#b91c1c;">⚠️ Summer Chiller Peak</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-blue"></div>
            <div class="metric-title">Scope 2 Carbon Footprint</div>
            <div class="metric-num" style="color:#1d4ed8;">{carbon_tons:,.1f} <span class="metric-unit">t CO₂</span></div>
            <div class="metric-subtext" style="color:#1d4ed8;">🌍 Indirect Grid Emissions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("#### 📅 Monthly Electricity Consumption (MWh)")
        m_grouped = df_main.groupby('month')['energy_consumption_kwh'].agg(['sum', 'mean']).reset_index()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        m_grouped['Month'] = m_grouped['month'].apply(lambda m: months[m-1])
        m_grouped['MWh'] = m_grouped['sum'] / 1000.0
        
        fig_bar = px.bar(
            m_grouped,
            x='Month',
            y='MWh',
            color='mean',
            color_continuous_scale=['#bbf7d0', '#059669', '#064e3b'],
            labels={'MWh': 'Consumption (MWh)', 'mean': 'Avg Load (kWh)'}
        )
        fig_bar.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=20), height=320)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("📌 **Plain-English Takeaway:** May, June, and July consume up to 2.4× more electricity than December because air conditioners run at maximum capacity during extreme summer temperatures.")

    with c_right:
        st.markdown("#### ⚡ Peak vs. Normal Hours Share")
        offpeak_mwh = tot_mwh - peak_mwh
        fig_donut = px.pie(
            values=[peak_mwh, offpeak_mwh],
            names=['Peak Hours (10:00–16:00)', 'Normal / Off-Peak Hours'],
            color_discrete_sequence=['#ef4444', '#10b981'],
            hole=0.5
        )
        fig_donut.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption("📌 **Plain-English Takeaway:** Nearly **30%** of all energy is consumed during peak hours, which carry a 20% to 25% Time-of-Day (ToD) tariff surcharge.")

# ==============================================================================
# 3. VISUAL CONSUMPTION ANALYTICS
# ==============================================================================
elif nav_selection == "📈 Visual Consumption Analytics":
    st.markdown("## 📈 Visual Consumption Analytics")
    st.markdown("Explore when, why, and how electricity is used across the day and throughout the seasons.")
    
    t1, t2, t3 = st.tabs(["🕒 Daily Load Curves (Weekday vs Weekend)", "🌡️ Temperature & AC Cooling Effect", "🔥 Peak Hour Intensity Heatmap"])
    
    with t1:
        st.markdown("#### 🕒 Average Diurnal Load Profile")
        diurnal = df_main.groupby(['hour', 'is_weekend'])['energy_consumption_kwh'].mean().reset_index()
        diurnal['Schedule'] = diurnal['is_weekend'].map({0: 'Weekday (Classes & Labs Open)', 1: 'Weekend (Low Campus Activity)'})
        
        fig_d = px.line(
            diurnal,
            x='hour',
            y='energy_consumption_kwh',
            color='Schedule',
            color_discrete_map={'Weekday (Classes & Labs Open)': '#059669', 'Weekend (Low Campus Activity)': '#64748b'},
            markers=True,
            labels={'hour': 'Hour of the Day (00:00 - 23:00)', 'energy_consumption_kwh': 'Average Load (kWh)'}
        )
        fig_d.add_vrect(x0=10, x1=16, fillcolor="#fee2e2", opacity=0.4, line_width=0, annotation_text="Peak Tariff Window (10-16h)", annotation_position="top left")
        fig_d.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig_d, use_container_width=True)
        
        st.markdown("""
        <div class="simple-box">
            💡 <b>What this shows in simple terms:</b> On weekdays (green line), energy consumption rises steeply at 08:00 AM as lecture halls, computer labs, 
            and offices open, staying high until 16:00 PM. On weekends (grey line), consumption remains near the constant standby baseload (~50 kWh).
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("#### 🌡️ Temperature vs. Electricity Demand Elasticity")
        sample_eda = df_main.sample(1500, random_state=42)
        fig_s = px.scatter(
            sample_eda,
            x='temperature_c',
            y='energy_consumption_kwh',
            color='is_business_hours',
            color_discrete_map={0: '#94a3b8', 1: '#059669'},
            labels={'temperature_c': 'Outdoor Ambient Temperature (°C)', 'energy_consumption_kwh': 'Electricity Consumption (kWh)', 'is_business_hours': 'Business Hours'},
            opacity=0.6
        )
        fig_s.add_vline(x=24.0, line_dash="dash", line_color="#dc2626", annotation_text="BEE 24°C Comfort Threshold", annotation_position="top right")
        fig_s.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig_s, use_container_width=True)
        
        st.markdown("""
        <div class="simple-box">
            💡 <b>What this shows in simple terms:</b> Notice the sharp upward bend starting at <b>24°C</b>! Below 24°C, energy demand is flat. 
            Once outdoor temperatures exceed 24°C, air conditioning compressors turn on and power consumption shoots up dramatically.
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("#### 🔥 Peak Demand Intensity Heatmap (Day of Week vs. Hour)")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = df_main.pivot_table(values='energy_consumption_kwh', index='day_of_week', columns='hour', aggfunc='mean')
        pivot.index = [day_names[i] for i in pivot.index]
        
        fig_h = px.imshow(
            pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="kWh"),
            x=list(range(24)),
            y=day_names,
            color_continuous_scale='YlOrRd'
        )
        fig_h.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig_h, use_container_width=True)
        
        st.markdown("""
        <div class="simple-box">
            💡 <b>What this shows in simple terms:</b> The deep red rectangular block highlights the <b>critical risk zone</b>: 
            <b>Monday through Friday between 11:00 AM and 15:00 PM</b>. Shifting heavy laboratory equipment or dishwasher/autoclave cycles outside this red zone directly lowers electricity bills.
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 4. INTERACTIVE PREDICTION LAB
# ==============================================================================
elif nav_selection == "🔮 Interactive Prediction Lab":
    st.markdown("## 🔮 Energy Prediction Lab")
    st.markdown("Simulate operational scenarios or batch-upload real facility telemetry to generate instant forecasts with 95% statistical confidence intervals.")
    
    tab_sim, tab_batch, tab_models = st.tabs(["🎛️ 1-Click Scenario Simulator", "📁 Batch CSV File Upload", "🏆 Model Benchmark Leaderboard"])
    
    with tab_sim:
        st.markdown("#### ⚡ 1-Click Scenario Presets (Try These First!)")
        
        # 1-Click Quick Preset Buttons
        b1, b2, b3, b4 = st.columns(4)
        preset_choice = None
        with b1:
            if st.button("☀️ Hot Summer Afternoon", use_container_width=True):
                preset_choice = "summer"
        with b2:
            if st.button("🌙 Cool Winter Night", use_container_width=True):
                preset_choice = "winter"
        with b3:
            if st.button("🌤️ Mild Spring Morning", use_container_width=True):
                preset_choice = "spring"
        with b4:
            if st.button("🏖️ Quiet Weekend", use_container_width=True):
                preset_choice = "weekend"
                
        # Derive defaults
        val_t = 38.0 if preset_choice == "summer" else (12.0 if preset_choice == "winter" else (23.5 if preset_choice == "spring" else 28.0))
        val_h = 48.0 if preset_choice == "summer" else (65.0 if preset_choice == "winter" else (50.0 if preset_choice == "spring" else 55.0))
        val_hr = 14 if preset_choice == "summer" else (2 if preset_choice == "winter" else (10 if preset_choice == "spring" else 13))
        val_dow = 1 if preset_choice == "summer" else (2 if preset_choice == "winter" else (3 if preset_choice == "spring" else 6))
        val_mo = 6 if preset_choice == "summer" else (12 if preset_choice == "winter" else (3 if preset_choice == "spring" else 5))
        val_o = 90.0 if preset_choice == "summer" else (5.0 if preset_choice == "winter" else (80.0 if preset_choice == "spring" else 15.0))
        
        st.markdown("---")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("##### 🌡️ Weather")
            inp_temp = st.slider("Outdoor Temperature (°C)", 5.0, 48.0, float(val_t), 0.5)
            inp_hum = st.slider("Relative Humidity (%)", 15.0, 95.0, float(val_h), 1.0)
        with col_s2:
            st.markdown("##### 🕒 Time & Month")
            inp_hour = st.slider("Hour of Day (24h Clock)", 0, 23, int(val_hr), 1)
            inp_month = st.selectbox("Month of the Year", list(range(1, 13)), index=int(val_mo-1), format_func=lambda m: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
        with col_s3:
            st.markdown("##### 🏢 Campus Population")
            inp_dow = st.selectbox("Day of Week", list(range(7)), index=int(val_dow), format_func=lambda d: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][d])
            inp_occ = st.slider("Facility Occupancy (%)", 0.0, 100.0, float(val_o), 5.0)
            
        inp_wk = 1 if inp_dow >= 5 else 0
        
        # Real-time Prediction Execution
        pred_data = predictor.predict_single(
            temperature_c=inp_temp,
            humidity_pct=inp_hum,
            hour=inp_hour,
            day_of_week=inp_dow,
            month=inp_month,
            is_weekend=inp_wk,
            occupancy_estimate=inp_occ
        )
        
        st.markdown("---")
        st.markdown("#### 🎯 Prediction Results & Practical Translation:")
        
        p_col1, p_col2 = st.columns([1, 1])
        with p_col1:
            val_pred = pred_data['predicted_kwh']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val_pred,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Load", 'font': {'size': 20, 'color': '#0f172a'}},
                number={'suffix': " kWh", 'font': {'size': 32, 'color': '#059669'}},
                gauge={
                    'axis': {'range': [0, 320], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#059669"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 100], 'color': "#dcfce7"},
                        {'range': [100, 190], 'color': "#fef9c3"},
                        {'range': [190, 320], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "#dc2626", 'width': 3},
                        'thickness': 0.75,
                        'value': 200
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with p_col2:
            st.markdown("<br/>", unsafe_allow_html=True)
            if pred_data['is_peak_hour']:
                st.markdown("""
                <div class="banner-peak">
                    ⚠️ <b>HIGH PEAK TARIFF SURCHARGE ACTIVE</b><br/>
                    Peak tariff window (10:00–16:00) is in effect. Commercial tariffs incur a +20% to +25% premium. 
                    Immediate pre-cooling or load shifting recommended.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="banner-offpeak">
                    🟢 <b>NORMAL / OFF-PEAK OPERATIONAL WINDOW</b><br/>
                    Normal baseload billing applies. Safe time to run power-intensive lab equipment and water pumps.
                </div>
                """, unsafe_allow_html=True)
                
            # Cost and Carbon Impact Translation
            est_cost = val_pred * (9.5 if pred_data['is_peak_hour'] else 7.5)
            est_co2 = val_pred * 0.78
            
            st.markdown(f"""
            <div class="simple-box" style="margin-top:10px;">
                💰 <b>Estimated Operating Cost:</b> ~₹{est_cost:,.0f} per hour<br/>
                🌍 <b>Estimated Carbon Emission:</b> ~{est_co2:.1f} kg CO₂ per hour<br/>
                📊 <b>95% Confidence Band:</b> <b>{pred_data['lower_bound_95']} kWh</b> to <b>{pred_data['upper_bound_95']} kWh</b>
            </div>
            """, unsafe_allow_html=True)

    with tab_batch:
        st.markdown("#### 📁 Batch Prediction via CSV Upload")
        st.markdown("Upload any CSV dataset containing facility parameters to generate predictions in bulk. (Try `data/sample_upload_test.csv`).")
        
        uploaded_csv = st.file_uploader("Upload CSV Dataset", type=["csv"])
        if uploaded_csv is not None:
            try:
                df_raw = pd.read_csv(uploaded_csv)
                st.write(f"Loaded `{uploaded_csv.name}` ({len(df_raw)} rows).")
                
                is_valid, missing_cols = validate_schema(df_raw)
                if not is_valid:
                    st.error(f"❌ Missing required columns: `{missing_cols}`. Please verify schema.")
                else:
                    st.success("✅ Schema Verified! Running batch inference...")
                    df_clean = clean_data(df_raw)
                    df_pred = predictor.predict_batch(df_clean)
                    
                    st.dataframe(df_pred[['datetime', 'temperature_c', 'occupancy_estimate', 'predicted_energy_kwh', 'pred_lower_95', 'pred_upper_95']].head(8), use_container_width=True)
                    
                    fig_batch_line = px.line(
                        df_pred,
                        x='datetime',
                        y='predicted_energy_kwh',
                        title="Forecasted Electricity Profile Over Uploaded Period",
                        labels={'datetime': 'Timestamp', 'predicted_energy_kwh': 'Predicted kWh'}
                    )
                    fig_batch_line.update_layout(template="plotly_white", height=320)
                    st.plotly_chart(fig_batch_line, use_container_width=True)
                    
                    csv_export = df_pred.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Predictions CSV", data=csv_export, file_name="ecopower_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    with tab_models:
        st.markdown("#### 🏆 Machine Learning Model Benchmark Leaderboard")
        st.markdown("""
        All candidate models were evaluated strictly using an **80/20 chronological train/test split** (7,027 training samples vs 1,757 test samples). 
        Future data was never shown to models during training, guaranteeing authentic evaluation.
        """)
        
        with open(os.path.join(APP_DIR, "models", "model_metadata.json"), "r") as f:
            metadata = json.load(f)
            
        board_df = pd.DataFrame(metadata['metrics_comparison']).T.reset_index()
        board_df.columns = ['Model Architecture', 'MAE (kWh)', 'RMSE (kWh)', 'R² Score']
        st.table(board_df)
        
        st.markdown(f"""
        <div class="simple-box">
            🏆 <b>Why Gradient Boosting Won:</b> The Gradient Boosting Regressor achieved the lowest RMSE of <b>{metadata['best_metrics']['RMSE']} kWh</b> 
            and the highest R² of <b>{metadata['best_metrics']['R2']}</b>. In simple terms: on average, the AI's prediction is within <b>3.8 kWh</b> of the actual 
            meter reading across all conditions!
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ⚖️ Feature Importance: What Matters Most?")
        fi_data = pd.DataFrame(list(metadata['feature_importance'].items()), columns=['Feature', 'Importance Weight']).sort_values('Importance Weight', ascending=True)
        fig_fi_chart = px.bar(fi_data, x='Importance Weight', y='Feature', orientation='h', color='Importance Weight', color_continuous_scale=['#bbf7d0', '#059669'])
        fig_fi_chart.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_fi_chart, use_container_width=True)

# ==============================================================================
# 5. AI SUSTAINABILITY ADVISOR (RAG + IBM BOB)
# ==============================================================================
elif nav_selection == "🌱 AI Sustainability Advisor":
    st.markdown("## 🌱 AI Sustainability Advisor")
    st.markdown("""
    **Evidence-grounded sustainability recommendations** powered by real-time telemetry, 
    vector search across official standards (BEE, IEA, UN), and **IBM Bob** structured prompt engineering.
    """)
    
    st.markdown("#### 💡 Click Any Demonstration Query to Test the System:")
    
    q_options = [
        "What are the best ways to reduce energy consumption during peak hours?",
        "Our electricity consumption increased by 15% this month. What could be done?",
        "What are the recommended BEE air conditioning temperature setpoints and efficiency guidelines?",
        "How can our campus laboratory and computer facilities minimize idle phantom power?",
        "How does reducing peak energy demand support UN SDG 7 and SDG 13 climate targets?"
    ]
    
    q1, q2 = st.columns(2)
    selected_query = None
    for idx, q_text in enumerate(q_options):
        col_btn = q1 if idx % 2 == 0 else q2
        if col_btn.button(f"📌 {q_text}", key=f"q_btn_{idx}", use_container_width=True):
            selected_query = q_text
            
    st.markdown("---")
    
    custom_q = st.text_area(
        "Or type your custom energy management query:",
        value=selected_query if selected_query else "",
        placeholder="e.g. How can we optimize our campus central chillers to lower peak electricity charges?",
        height=85
    )
    
    with st.expander("📡 Active Simulated Telemetry Context (Injected into Prompt)", expanded=False):
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        with c_t1:
            tel_load = st.number_input("Active Load (kWh)", value=215.0, step=5.0)
        with c_t2:
            tel_temp = st.number_input("Ambient Temp (°C)", value=35.5, step=0.5)
        with c_t3:
            tel_is_peak = st.checkbox("Peak Window Active (10-16h)", value=True)
        with c_t4:
            tel_occ_pct = st.number_input("Campus Occupancy (%)", value=85.0, step=5.0)
            
    telemetry_dict = {
        "current_kwh": tel_load,
        "temperature_c": tel_temp,
        "is_peak": tel_is_peak,
        "occupancy_pct": tel_occ_pct
    }
    
    if st.button("🚀 Ask Sustainability Advisor", type="primary", use_container_width=True) and custom_q.strip():
        with st.spinner("Searching official sustainability standards and synthesizing grounded advice..."):
            rag_res = rag_engine.generate_grounded_response(custom_q, telemetry_context=telemetry_dict)
            
        if rag_res['is_out_of_scope']:
            st.warning(f"🛡️ **Responsible AI Notice:** {rag_res['answer']}")
        else:
            conf_val = int(rag_res['confidence_score'] * 100)
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 20px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 800; color: #166534; font-size: 16px;">✅ Evidence-Grounded Advice Generated</span><br/>
                    <small style="color: #4b5563;">Cross-referenced against {len(rag_res['retrieved_sources'])} official policy standards</small>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase;">Retrieval Confidence</span><br/>
                    <span style="font-size: 22px; font-weight: 800; color: #059669;">{conf_val}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Grounded advice output
            st.markdown(rag_res['answer'])
            
            # Source transparency audit drawer
            with st.expander("🔍 Transparent Knowledge Retrieval Audit (View Exact Source Citations)", expanded=True):
                st.caption("Inspect the exact source documents and similarity scores retrieved from the vector store:")
                for src in rag_res['retrieved_sources']:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div style="font-weight: 700; color: #0f172a; font-size: 13px;">📄 {src['title']}</div>
                        <div style="font-size: 11.5px; color: #64748b; margin-top: 2px;">
                            🏛️ <b>Authority:</b> {src['authority']} | 🏷️ <b>Category:</b> {src['category']} | 🎯 <b>Cosine Similarity:</b> {src['similarity_score']:.4f}
                        </div>
                        <div style="font-size: 12px; color: #334155; margin-top: 6px; font-style: italic;">
                            "{src['snippet']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# 6. SUSTAINABILITY STANDARDS KB
# ==============================================================================
elif nav_selection == "📚 Sustainability Standards KB":
    st.markdown("## 📚 Curated Sustainability Knowledge Base")
    st.markdown("Inspect the authoritative documents ingested and indexed into the EcoPower AI vector retrieval engine.")
    
    kb_path = os.path.join(APP_DIR, "knowledge_base", "sustainability_documents")
    doc_files = sorted([f for f in os.listdir(kb_path) if f.endswith(".txt")])
    
    selected_kb_file = st.selectbox("Select Official Document to Inspect:", doc_files)
    if selected_kb_file:
        full_path = os.path.join(kb_path, selected_kb_file)
        with open(full_path, "r", encoding="utf-8") as f:
            doc_content = f.read()
            
        st.markdown(f"#### 📄 Document: `{selected_kb_file}`")
        st.text_area("Ingested Document Text:", value=doc_content, height=360)
        
    st.markdown("---")
    st.markdown("#### 📊 Vector Store Statistics:")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Curated Official Documents", len(doc_files))
    with k2:
        st.metric("Semantic Chunks Indexed", len(rag_engine.chunks))
    with k3:
        st.metric("Vector Representation", "Sublinear TF-IDF + Bi-gram Cosine")

# ==============================================================================
# 7. RESPONSIBLE AI CHARTER
# ==============================================================================
elif nav_selection == "🛡️ Responsible AI Charter":
    st.markdown("## 🛡️ Responsible AI Governance Charter")
    st.markdown("EcoPower AI strictly complies with ethical AI standards established by IBM SkillsBuild, 1M1B, and international AI governance frameworks.")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="metric-card-accent"></div>
            <h4 style="color: #065f46; margin: 0 0 8px 0;">1. Fairness & Equity</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Never penalizes essential baseload needs (student dormitories, medical facilities, research servers).<br/>
                • Differentiates between discretionary cooling and vital baseline safety ventilation.
            </p>
        </div>
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="metric-card-accent metric-card-accent-blue"></div>
            <h4 style="color: #1e40af; margin: 0 0 8px 0;">2. Transparency & Explainability</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Every recommendation explicitly cites its governing authority (e.g. BEE India) and similarity score.<br/>
                • Predictions report explicit 95% statistical confidence intervals rather than misleading single-point values.
            </p>
        </div>
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-teal"></div>
            <h4 style="color: #0f766e; margin: 0 0 8px 0;">3. Privacy & Data Minimization</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Operates strictly on aggregated telemetry (kWh, temperature, timestamps).<br/>
                • Zero personal student or resident identity data is collected or stored.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_r2:
        st.markdown("""
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="metric-card-accent metric-card-accent-red"></div>
            <h4 style="color: #b91c1c; margin: 0 0 8px 0;">4. Operational Safety Guardrails</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Strictly refuses hazardous DIY electrical rewiring or high-voltage breaker modifications.<br/>
                • Mandates human oversight: major policies require approval from certified facility electrical engineers.
            </p>
        </div>
        <div class="metric-card" style="margin-bottom:14px;">
            <div class="metric-card-accent"></div>
            <h4 style="color: #065f46; margin: 0 0 8px 0;">5. Scientific Humility & No Hallucinations</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Never fabricates unverified energy savings percentages.<br/>
                • Out-of-domain queries (e.g. sports, entertainment) are politely and explicitly declined.
            </p>
        </div>
        <div class="metric-card">
            <div class="metric-card-accent metric-card-accent-teal"></div>
            <h4 style="color: #0f766e; margin: 0 0 8px 0;">6. Data Limitations Disclosure</h4>
            <p style="font-size: 13px; color: #475569; line-height: 1.5; margin: 0;">
                • Model accuracy depends on physical sensor calibration; extreme unseasonal heatwaves introduce natural variance.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# 8. AUTHOR & INTERNSHIP CREDENTIALS
# ==============================================================================
elif nav_selection == "👩‍🎓 Author & Internship Credentials":
    st.markdown("## 👩‍🎓 Project Credentials & IBM Bob Co-Development Log")
    
    st.markdown("""
    <div class="hero-header" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);">
        <div class="hero-badge-row">
            <span class="badge-chip badge-sdg7">1M1B Virtual Internship</span>
            <span class="badge-chip badge-ibm">IBM SkillsBuild</span>
            <span class="badge-chip badge-sdg13">AICTE Collaboration</span>
        </div>
        <div class="hero-title" style="font-size: 28px;">Vanshika Verma</div>
        <div class="hero-desc">
            B.Tech Computer Science and Engineering (Artificial Intelligence)<br/>
            Om Sterling Global University, Hisar, Haryana | 🎯 Career Aspiration: <b>Data Scientist</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🤖 How IBM Bob Was Used in This Project")
    st.markdown("""
    **IBM Bob** served as an AI developer assistant throughout the project lifecycle:
    
    1. **Data Science & ML Pipeline Formulation:**
       - Formulating the physics-informed Cooling Degree Day ($CDD = \max(0, T - 24^\circ\text{C})$) metric based on Indian BEE standards.
       - Enforcing the strict chronological 80/20 train/test split to prevent temporal data leakage.
       
    2. **RAG Architecture & Knowledge Structuring:**
       - Segmenting official guidelines from BEE India, IEA, and UN SDG frameworks into coherent semantic chunks.
       - Designing the sublinear TF-IDF and bi-gram cosine vector retrieval store for zero-failure offline execution.
       
    3. **Prompt Engineering & Structured Grounding:**
       - Formulating the 5-stage grounded synthesis schema (`Observation -> Pattern -> Ranked Actions -> Impact -> Sources`).
       - Implementing out-of-scope guardrails and confidence metrics.
       
    4. **Streamlit UI & Responsible AI Governance:**
       - Designing the modern sustainability theme, 1-click scenario presets, and the 6-point Responsible AI governance charter.
    """)

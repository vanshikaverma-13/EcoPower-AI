# ⚡ EcoPower AI – AI-Powered Energy Consumption Prediction & Sustainability Advisor

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E.svg)](https://scikit-learn.org/)
[![SDG 7](https://img.shields.io/badge/SDG%207-Affordable%20%26%20Clean%20Energy-FCC30B.svg)](https://sdgs.un.org/goals/goal7)
[![SDG 13](https://img.shields.io/badge/SDG%2013-Climate%20Action-3F7E44.svg)](https://sdgs.un.org/goals/goal13)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Portfolio Project for the 1M1B AI for Sustainability Virtual Internship**  
> In collaboration with **IBM SkillsBuild** and **AICTE**  
> **Student:** Vanshika Verma | **Course:** B.Tech CSE – AI | **Institution:** Om Sterling Global University, Hisar | **Career Goal:** Data Scientist  

---

## 📖 1. Project Overview & Motivation

Electricity demand in institutional environments (such as universities, research centers, commercial complexes, and student dormitories) is characterized by acute demand surges, severe chiller inefficiencies during hot afternoons, and persistent phantom loads during off-hours. These inefficiencies inflate electricity tariffs through peak demand surcharges and expand indirect greenhouse gas (Scope 2) emissions.

**EcoPower AI** is an end-to-end Data Science, Machine Learning, and Retrieval-Augmented Generation (RAG) platform designed to:
1. **Analyze Historical Patterns:** Uncover hourly, diurnal, weekly, and seasonal electricity trends from 1 year of hourly facility telemetry (8,784 records).
2. **Forecast Future Energy Demand:** Deliver real-time electricity forecasts with 95% confidence intervals using a calibrated Gradient Boosting Regressor ($R^2 = 0.9839$).
3. **Mitigate Peak Surges:** Identify high-tariff peak windows (10:00 – 16:00) and alert operators before demand thresholds are breached.
4. **Deliver Evidence-Grounded Recommendations:** Employ a RAG engine to retrieve verified energy conservation measures from official authorities (Bureau of Energy Efficiency India, International Energy Agency, United Nations SDG Framework).
5. **Enforce Responsible AI:** Ensure complete source attribution, explainable uncertainty bounds, strict privacy protection, and operational safety guardrails.

---

## 🎯 2. Sustainable Development Goals (SDGs) Alignment

| Goal | Description | Project Contribution |
| :--- | :--- | :--- |
| **SDG 7 (Primary)** | **Affordable and Clean Energy** | Direct operational support for **Target 7.3** (doubling the global rate of improvement in energy efficiency). Enables educational campuses and facilities to reduce wastage and optimize HVAC load. |
| **SDG 13 (Secondary)** | **Climate Action** | In fossil-heavy grid regions (~0.78 kg $CO_2$/kWh in Northern India), reducing avoidable electricity consumption directly abates Scope 2 greenhouse gas emissions. |

---

## 🏛️ 3. Overall System Architecture

The architecture seamlessly connects statistical Machine Learning with Semantic RAG, co-developed with IBM Bob prompt engineering methodologies:

```
                            ┌────────────────────────────────────────┐
                            │            USER INTERFACE              │
                            │  (Streamlit Professional Dashboard)    │
                            └───────────────────┬────────────────────┘
                                                │
                                Query / Telemetry / Uploaded CSV
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
       ┌───────────────────────────────┐                       ┌───────────────────────────────┐
       │     DATA SCIENCE & ML LAYER   │                       │       RAG & IBM BOB LAYER     │
       ├───────────────────────────────┤                       ├───────────────────────────────┤
       │ • Data Ingestion & Cleaning   │                       │ • Curated Sustainability KB   │
       │ • Feature Engineering (CDD)   │                       │   (BEE, IEA, UN SDG 7, CERC)  │
       │ • Diurnal & Seasonal EDA      │                       │ • Chunking & Vectorization    │
       │ • Multi-Model Benchmark       │                       │ • Semantic Retrieval (Top-k)  │
       │   (Linear, RF, GradientBoost) │                       │ • Context Injection Engine    │
       │ • Inference Engine (95% CI)   │                       │ • IBM Bob Grounded Synthesis  │
       └──────────────┬────────────────┘                       └───────────────┬───────────────┘
                      │                                                        │
                      │               Energy Forecasts & Telemetry             │
                      └─────────────────────────┬──────────────────────────────┘
                                                ▼
                            ┌────────────────────────────────────────┐
                            │    AI SUSTAINABILITY ADVISOR SYNTHESIS │
                            │   Empirical Telemetry + Grounded Norms │
                            │   = Actionable, Low-Carbon Roadmap     │
                            └───────────────────┬────────────────────┘
                                                ▼
                            ┌────────────────────────────────────────┐
                            │      RESPONSIBLE AI GOVERNANCE         │
                            │   Fairness • Safety • Explainability   │
                            │   Zero Hallucination • Privacy-First   │
                            └────────────────────────────────────────┘
```

---

## 🔬 4. Data Science & Machine Learning Workflow

### 4.1 Granular 1-Year Telemetry Dataset
The project models 8,784 hourly intervals (1 full calendar year) reflecting realistic campus energy dynamics:
- `datetime`: Hourly timestamp sequence
- `temperature_c`: Ambient outdoor dry-bulb temperature (10°C to 42°C seasonal curve with diurnal oscillations)
- `humidity_pct`: Relative humidity (inversely correlated with temperature + monsoon peak)
- `hour`, `day_of_week`, `month`, `is_weekend`, `is_business_hours`: Temporal calendar variables
- `occupancy_estimate`: Estimated campus population density (0% to 95%)
- `peak_hours_flag`: High-tariff regulatory window (10:00 – 16:00 on weekdays)
- `energy_consumption_kwh`: Physical load modeled with baseload, occupant draw, and non-linear chiller cooling degree responses.

### 4.2 Physics-Informed Feature Engineering
- **Cooling Degree Days Proxy ($CDD$):** $\max(0, 	ext{Temperature} - 24.0^\circ	ext{C})$ based on Bureau of Energy Efficiency (BEE) comfort baselines.
- **Heating Degree Days Proxy ($HDD$):** $\max(0, 15.0^\circ	ext{C} - 	ext{Temperature})$.
- **Cyclic Encodings:** Continuous trigonometric mapping of temporal variables:
  $$\sin\left(rac{2\pi \cdot 	ext{hour}}{24}ight), \quad \cos\left(rac{2\pi \cdot 	ext{hour}}{24}ight), \quad \sin\left(rac{2\pi \cdot 	ext{month}}{12}ight), \quad \cos\left(rac{2\pi \cdot 	ext{month}}{12}ight)$$
- **Heat Index Interaction Proxy:** $	ext{Temperature} + 0.05 \cdot 	ext{Humidity}$.

### 4.3 Model Benchmarking & Selection (Zero Data Leakage)
To mirror real-world production forecasting, a **chronological 80/20 train/test split** was implemented (7,027 training intervals vs. 1,757 test intervals). Future timestamps were strictly kept out of training.

| Model Candidate | Model Family | MAE (kWh) | RMSE (kWh) | $R^2$ Score | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | Baseline Parametric | 8.482 | 10.902 | 0.8716 | Baseline |
| **Random Forest Regressor** | Ensemble Bagging (100 trees) | 3.067 | 3.947 | 0.9832 | Runner-Up |
| **Gradient Boosting Regressor** | Sequential Boosting | **3.014** | **3.862** | **0.9839** | 🏆 **Best Model** |

**Why Gradient Boosting Won:**
Gradient Boosting effectively models the non-linear interaction between ambient temperature surges above 24°C and high occupant density during weekday peak hours without overfitting. The best model and feature scaler are serialized in `models/best_model.pkl` and `models/scaler.pkl`.

---

## 📚 5. RAG (Retrieval-Augmented Generation) Architecture

EcoPower AI incorporates a genuine, offline-resilient RAG pipeline that grounds advice in verified sustainability policies:

```
Institutional Sustainability Documents
(BEE India, IEA Paris, UN SDG 7, CERC Peak Management, Campus Playbook)
                          │
                          ▼
            Metadata Parsing & Header Tagging
                          │
                          ▼
             Recursive Semantic Chunking
              (Topic-aware segment boundaries)
                          │
                          ▼
               Vectorization Engine
         (Sublinear TF-IDF + Bi-gram N-Grams)
                          │
                          ▼
           Cosine Similarity Vector Store
                          │
                          ▼
      Top-k Semantic Retrieval (Cosine Threshold >= 0.08)
                          │
                          ▼
          Telemetry Context Augmentation
        (Real-time Load, Temp, Peak Status)
                          │
                          ▼
           IBM Bob Grounded Synthesis
                          │
                          ▼
   Structured Advisory Output + Full Source Audit
```

### Curated Knowledge Base Authorities:
1. **Bureau of Energy Efficiency (BEE), Ministry of Power, India:** ECBC standards, 24°C–26°C statutory setpoints, 5-Star appliance benchmarks.
2. **International Energy Agency (IEA):** Demand-side flexibility, VFD efficiency gains, thermal pre-cooling protocols.
3. **United Nations Division for Sustainable Goals:** SDG 7.2, 7.3, and SDG 13 carbon abatement calculations.
4. **Central Electricity Regulatory Commission (CERC):** Time-of-Day (ToD) tariff windows, maximum demand billing, power factor penalties.
5. **Sustainable Educational Campus Playbook:** Laboratory sleep profiles, dormitory phantom load controls, student ambassador audits.

---

## 🤖 6. Realistic IBM Bob Integration

During this internship, **IBM Bob** was utilized as an AI pair-programmer and knowledge architect across five stages:

1. **System Prompt Formulation:** Guiding the AI assistant to strictly adhere to Responsible AI principles, disallowing DIY electrical advice and unverified green claims.
2. **Structured Response Template Design:** Formulating the 5-point grounded reporting standard:
   - `[Energy Insight & Telemetry Observation]`
   - `[Identified Consumption Pattern]`
   - `[Recommended Sustainable Actions (Prioritized)]`
   - `[Expected Environmental & Emission Impact]`
   - `[Grounded Knowledge Sources]`
3. **Feature Engineering Design:** Advising the formulation of Cooling Degree Days (CDD) relative to the Indian Bureau of Energy Efficiency standard (24°C baseline).
4. **RAG Vector Search Optimization:** Designing semantic chunk boundaries and cosine similarity thresholds to reliably separate relevant queries from out-of-scope noise.
5. **Responsible AI Auditing:** Structuring tests to catch invalid inputs, missing columns, and out-of-domain queries.

---

## 🖥️ 7. Streamlit Dashboard Features

The dashboard features 8 dedicated tabs:
1. **🏠 Home & Overview:** Hero banner, SDG badges, high-level metrics, and interactive architecture diagram.
2. **📊 Energy KPI Dashboard:** Annual consumption (MWh), average demand (kWh), peak surge alerts, Scope 2 carbon footprint, and monthly breakdowns.
3. **📈 Exploratory Data Analysis (EDA):** Interactive Plotly charts for diurnal weekday/weekend curves, temperature elasticity scatter, and a day-of-week peak heatmap.
4. **🔮 Energy Prediction Lab:** 
   - Interactive what-if scenario sliders (Temperature, Humidity, Hour, Month, Day, Occupancy) with real-time predictions and 95% confidence intervals.
   - Batch CSV file uploader with schema validation, missing data imputation, and predictions CSV export.
   - Model benchmark leaderboard and feature importance charts.
5. **🌱 AI Sustainability Advisor:** Interactive query console with 5 quick-demo questions, active simulated building telemetry injection, transparent cosine similarity scores, and grounded answers.
6. **📚 RAG Knowledge Base:** Interactive document reader displaying full text and chunking statistics for all 5 ingested policy documents.
7. **🛡️ Responsible AI Principles:** Detailed documentation covering Fairness, Transparency, Privacy, Operational Safety, Accuracy, and Human Oversight.
8. **ℹ️ About Project & IBM Bob:** Internship credentials, Vanshika Verma's profile, IBM Bob co-development log, and future expansion scope.

---

## 💬 8. Example AI Demonstrations & Queries

### Query 1: Peak Hour Energy Reduction
- **User Query:** *"What are the best ways to reduce energy consumption during peak hours?"*
- **Retrieved Sources:** `Handbook on Peak Load Management` (CERC) & `BEE Energy Conservation Guidelines`
- **Output:** Identifies peak tariff window (10:00 – 16:00). Recommends raising AC setpoints from 24°C to 25.5°C, staggering autoclave/lab equipment runs outside peak hours, and maintaining capacitor banks above 0.98 power factor.

### Query 2: Monthly Spike Diagnosis
- **User Query:** *"Our electricity consumption increased by 15% this month. What could be done?"*
- **Retrieved Sources:** `IEA Clean Energy Transitions` & `BEE Guidelines`
- **Output:** Identifies thermal cooling degree days ($CDD$) and increased occupancy as likely drivers. Recommends implementing morning thermal pre-cooling (06:00 – 08:30) and auditing AHU variable frequency drives.

### Query 3: BEE Air Conditioning Norms
- **User Query:** *"What are the recommended BEE air conditioning temperature setpoints and efficiency guidelines?"*
- **Retrieved Sources:** `BEE Energy Conservation Guidelines` (Ministry of Power)
- **Output:** Explains statutory 24°C–26°C default band. Notes that each 1°C increase saves ~6% cooling electricity. Explains BEE 5-Star inverter chiller benefits.

### Query 4: Laboratory & Dormitory Phantom Loads
- **User Query:** *"How can our campus laboratory and computer facilities minimize idle phantom power?"*
- **Retrieved Sources:** `Sustainable Educational Campus Playbook`
- **Output:** Explains that phantom loads represent 8% to 12% of total load. Recommends group policy sleep timers (display sleep after 10 min, hibernate after 30 min) and dormitory master cut-off switches.

### Query 5: Out-of-Scope Guardrail Demo
- **User Query:** *"Who won the ICC Cricket World Cup in 2023?"*
- **System Action:** Cosine similarity falls below threshold (< 0.08).
- **Output:** Politeness refusal adhering to Responsible AI: *"I could not retrieve sufficient authoritative sustainability guidelines to answer your question reliably. EcoPower AI strictly operates under Responsible AI principles and does not generate speculative answers."*

---

## 🛡️ 9. Responsible AI Principles

- **Fairness:** Does not penalize essential baseload consumption (medical equipment, servers, safety lighting).
- **Transparency:** Discloses 95% confidence intervals on all ML predictions and displays source document provenance and similarity metrics for every recommendation.
- **Privacy:** Ingests purely aggregated telemetry without collecting personally identifiable information (PII).
- **Safety:** Explicitly programmed to decline hazardous DIY electrical wiring, transformer repairs, or panel modifications.
- **Accuracy & Humility:** Uses calibrated models without inflating metrics; uses cautious phrasing ("could help", "potential reduction").
- **Human Oversight:** Recommendations are advisory; major facility modifications require sign-off from licensed electrical engineers.

---

## 📈 10. Expected Project Impact

- **Social Impact:** Improves energy literacy among students, faculty, and administrative staff through interactive visual dashboards.
- **Environmental Impact:** Encouraging targeted temperature setpoints (24°C–26°C) and load shifting could help reduce unnecessary electrical consumption, mitigating Scope 2 emissions in fossil-reliant grid regions.
- **Economic Impact:** Eliminates costly Time-of-Day (ToD) peak surcharges and maximum demand penalty spikes.
- **SDG Impact:** Directly advances UN SDG 7.3 and SDG 13 benchmarks.

---

## 💻 11. Technology Stack

- **Core Programming:** Python 3.12
- **Data Science & ML:** Pandas, NumPy, Scikit-Learn, Joblib
- **Visualization:** Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
- **RAG & NLP:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity pairwise metrics)
- **Web Application:** Streamlit
- **Quality Assurance:** Python Unittest (10 automated test cases)

---

## 🚀 12. Installation & Quickstart Guide

### Prerequisites
- Python 3.10+ installed on your system.

### Step 1: Navigate to the Project Directory
```powershell
cd C:\Users\ASUS\Desktop\EcoPower-AI
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run Automated Unit Tests
```powershell
python -m unittest tests/test_pipeline.py
```
*(All 10 test cases should pass with `OK`)*

### Step 4: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### Step 5: Explore the Jupyter Notebook
```powershell
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

---

## 📁 13. Project Folder Structure

```
EcoPower-AI/
│
├── data/
│   ├── energy_consumption.csv           # 8,784 hourly telemetry records (full year)
│   └── sample_upload_test.csv           # Sample 7-day dataset for upload verification
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb           # Comprehensive Data Science EDA & modeling notebook
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py            # Data loading, validation, missing value imputation
│   ├── feature_engineering.py           # CDD/HDD proxies, cyclical sin/cos hour/month encodings
│   ├── model_training.py                # Chronological train/test split, ML benchmarking
│   ├── prediction.py                    # Real-time inference engine with 95% confidence intervals
│   └── rag_pipeline.py                  # RAG ingestion, TF-IDF vector index, IBM Bob synthesis
│
├── models/
│   ├── best_model.pkl                   # Trained Gradient Boosting Regressor (R² = 0.9839)
│   ├── scaler.pkl                       # Standard feature scaler
│   └── model_metadata.json              # Evaluation metrics, hyperparameters, feature weights
│
├── knowledge_base/
│   └── sustainability_documents/
│       ├── bee_energy_efficiency_guidelines.txt    # Bureau of Energy Efficiency India
│       ├── iea_clean_energy_transitions.txt        # International Energy Agency
│       ├── un_sdg7_sustainable_energy_action.txt   # United Nations SDG 7 & 13
│       ├── peak_load_management_handbook.txt       # Central Electricity Regulatory Commission
│       └── sustainable_campus_playbook.txt         # Educational Campus Playbook
│
├── tests/
│   └── test_pipeline.py                 # 10 automated unit tests (CSV, RAG, ML, edge cases)
│
├── screenshots/
│   └── README.md                        # Portfolio screenshot capture checklist
│
├── app.py                               # Complete Streamlit multi-tab application
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Comprehensive GitHub portfolio documentation
├── PROJECT_DOCUMENTATION.md             # Deep-dive academic internship technical report
└── .gitignore                           # Standard git exclusion patterns
```

---

## 🔮 14. Future Scope

1. **IoT Smart Meter Telemetry:** Live ingestion via MQTT/Modbus protocols from digital electrical sub-meters.
2. **Automated Demand Response (ADR):** BACnet building management system integration to automatically throttle chiller setpoints during grid peak alerts.
3. **Solar PV & Battery Integration:** Forecasting campus rooftop solar generation to orchestrate battery storage charging and discharging cycles.

---

## 👩‍🎓 15. Author Information

- **Name:** Vanshika Verma
- **Academic Course:** B.Tech Computer Science and Engineering (Artificial Intelligence)
- **Institution:** Om Sterling Global University, Hisar, Haryana
- **Internship Program:** 1M1B AI for Sustainability Virtual Internship
- **Collaboration Partners:** IBM SkillsBuild & AICTE
- **Career Aspiration:** Data Scientist

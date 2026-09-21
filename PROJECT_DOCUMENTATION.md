# Technical & Academic Project Documentation: EcoPower AI

**Project Title:** EcoPower AI – AI-Powered Energy Consumption Prediction & Sustainability Advisor  
**Candidate Name:** Vanshika Verma  
**Academic Degree:** B.Tech Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Om Sterling Global University, Hisar, Haryana  
**Internship:** 1M1B AI for Sustainability Virtual Internship in collaboration with IBM SkillsBuild and AICTE  
**Primary SDG:** SDG 7 – Affordable and Clean Energy (Target 7.3)  
**Secondary SDG:** SDG 13 – Climate Action  
**Career Aspiration:** Data Scientist  

---

## 1. Executive Summary

EcoPower AI is a unified Data Science and Generative AI application designed to optimize institutional electricity consumption, curtail carbon emissions, and eliminate peak demand surcharges. The platform integrates:
- **Time-Series Machine Learning:** An 80/20 chronologically partitioned Gradient Boosting Regressor achieving an $R^2$ of 0.9839 and an RMSE of 3.862 kWh across 8,784 hourly facility observations.
- **Physics-Informed Feature Engineering:** Cooling Degree Day ($CDD$) and Heating Degree Day ($HDD$) formulations capturing thermal inertia and chiller elasticity relative to the Bureau of Energy Efficiency (BEE) 24°C threshold.
- **Evidence-Grounded RAG Engine:** A sublinear TF-IDF and bi-gram cosine vector retrieval store indexing authoritative documents from BEE India, the International Energy Agency (IEA), and the United Nations.
- **IBM Bob Guided Synthesis:** A deterministic prompt engineering architecture translating sensor telemetry and retrieved institutional guidelines into prioritized, explainable, and responsible recommendations.

---

## 2. Mathematical Formulation & Feature Engineering

### 2.1 Thermal Comfort & Cooling Degree Days (CDD)
Air conditioning represents 45% to 65% of institutional peak summer load. In accordance with Bureau of Energy Efficiency (BEE) standards, the statutory comfort baseline is set at $T_{\text{base}} = 24.0^\circ\text{C}$.
The cooling load proxy is defined as:
$$\text{CDD} = \max(0, T_{\text{ambient}} - 24.0)$$

For heating requirements during winter nights:
$$\text{HDD} = \max(0, 15.0 - T_{\text{ambient}})$$

### 2.2 Continuous Cyclical Temporal Encodings
Directly feeding discrete ordinal integers for hours ($0 \dots 23$) introduces an artificial discontinuity between 23:00 and 00:00. To preserve cyclic continuity:
$$\sin_{\text{hour}} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \cos_{\text{hour}} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\sin_{\text{month}} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \cos_{\text{month}} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$

### 2.3 Feature Space
The model ingests 13 normalized features:
1. `temperature_c` (Ambient dry-bulb temperature in °C)
2. `humidity_pct` (Relative humidity in %)
3. `occupancy_estimate` (Estimated facility density in %)
4. `is_weekend` (Binary indicator for Saturday/Sunday)
5. `is_business_hours` (Binary indicator for 08:00 - 18:00 weekdays)
6. `peak_hours_flag` (Binary indicator for 10:00 - 16:00 weekdays)
7. `cdd_cooling_load` (Cooling degree day proxy)
8. `hdd_heating_load` (Heating degree day proxy)
9. `sin_hour` (Harmonic sine of hour)
10. `cos_hour` (Harmonic cosine of hour)
11. `sin_month` (Harmonic sine of month)
12. `cos_month` (Harmonic cosine of month)
13. `heat_index_proxy` (Thermal discomfort index: $T + 0.05 \cdot H$)

---

## 3. Machine Learning Benchmark & Validation Strategy

### 3.1 Strict Time-Series Splitting (Zero Data Leakage)
In time-series forecasting, random k-fold cross-validation results in data leakage, where future timestamps falsely predict historical patterns. We enforce a strict chronological 80/20 split:
- **Training Set:** First 7,027 consecutive hours (January 1, 2024 to mid-October 2024).
- **Testing Set:** Remaining 1,757 consecutive hours (mid-October 2024 to December 31, 2024).

### 3.2 Benchmark Comparison Table

| Model Architecture | Hyperparameters | MAE (kWh) | RMSE (kWh) | $R^2$ Score | Selection Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | Default OLS, StandardScaler | 8.482 | 10.902 | 0.8716 | Baseline model; underfits non-linear thermal inflection |
| **Random Forest Regressor** | 100 trees, max_depth=12 | 3.067 | 3.947 | 0.9832 | Robust; low bias, minor variance on peak extremes |
| **Gradient Boosting Regressor** | 150 estimators, lr=0.08, max_depth=5 | **3.014** | **3.862** | **0.9839** | 🏆 **Champion Model; superior residual minimization** |

### 3.3 Uncertainty Quantification
Point forecasts can provide a false sense of certainty. Every single inference generates a **95% Confidence Interval**:
$$\hat{y} \pm 1.96 \cdot \text{RMSE} \implies \hat{y} \pm 7.57 \text{ kWh}$$

---

## 4. Retrieval-Augmented Generation (RAG) Architecture

### 4.1 Knowledge Base Corpus
The knowledge base comprises 5 authoritative, curated institutional publications:
1. `bee_energy_efficiency_guidelines.txt`: Bureau of Energy Efficiency, Ministry of Power, India.
2. `iea_clean_energy_transitions.txt`: International Energy Agency, Paris.
3. `un_sdg7_sustainable_energy_action.txt`: United Nations Division for Sustainable Development Goals.
4. `peak_load_management_handbook.txt`: Central Electricity Regulatory Commission (CERC).
5. `sustainable_campus_playbook.txt`: Higher Education Sustainability Association.

### 4.2 Vector Retrieval Mathematics
- **Sublinear TF-IDF Scaling:** Replaces raw term frequency $tf$ with $1 + \log(tf)$ to curb dominance of repetitive technical terms.
- **Bi-Gram Feature Extraction:** Captures two-word phrases such as "peak hours", "setpoint temperature", and "variable frequency".
- **Cosine Similarity Metric:**
  $$\text{Sim}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$
- **Out-of-Scope Threshold:** If $\max(\text{Sim}(q, d)) < 0.08$, the query is classified as out-of-scope and gracefully declined under Responsible AI guardrails.

---

## 5. IBM Bob Co-Development Log

During this virtual internship, **IBM Bob** was leveraged as an AI co-developer and prompt engineer:

### Phase 1: Architecture & Data Science Design
- **IBM Bob Prompt:** *"How should we structure a campus electricity consumption dataset to reflect Northern India climate dynamics and BEE cooling standards without introducing data leakage?"*
- **Outcome:** IBM Bob provided the mathematical formulation for Cooling Degree Days ($T - 24^\circ\text{C}$) and advised the strict chronological 80/20 train/test split.

### Phase 2: RAG Pipeline & Semantic Chunking
- **IBM Bob Prompt:** *"Design a lightweight, deterministic RAG system that runs offline without third-party API dependencies while maintaining strict citation tracking."*
- **Outcome:** IBM Bob suggested sublinear TF-IDF vectorization with n-gram extraction, top-k cosine similarity filtering, and explicit metadata tagging.

### Phase 3: System Prompt & Structured Grounding
- **IBM Bob Prompt:** *"Formulate a system prompt for an AI Sustainability Advisor that combines active telemetry with retrieved policy chunks, preventing hallucinations and disallowing hazardous DIY electrical advice."*
- **Outcome:** IBM Bob designed the 5-point structured reporting schema (`Energy Insight` -> `Identified Pattern` -> `Recommended Actions` -> `Sustainability Impact` -> `Retrieved Sources`).

### Phase 4: Responsible AI Guardrails
- **IBM Bob Prompt:** *"What test cases and edge constraints are necessary to certify this project under Responsible AI principles?"*
- **Outcome:** IBM Bob authored the 10-point test matrix verifying schema validation, null imputation, type coercion, and out-of-scope query rejection.

---

## 6. Comprehensive 10-Step Demo Walkthrough

### Step 1: Launch Application
Execute `streamlit run app.py`. Browser opens to `http://localhost:8501`.

### Step 2: Review Executive Overview
Navigate to **🏠 Home & Overview**. Observe project alignment with SDG 7 and SDG 13, the high-level architecture diagram, and author credentials.

### Step 3: Inspect Annual Energy KPIs
Navigate to **📊 Energy KPI Dashboard**. Review total consumption (MWh), peak recorded surge (kWh), and Scope 2 carbon emissions ($CO_2$).

### Step 4: Examine Diurnal Load Profiles
Navigate to **📈 Exploratory Data Analysis**. Under Tab 1, observe the weekday vs. weekend profile highlighting the 10:00 – 16:00 peak tariff window.

### Step 5: Test What-If Scenario Prediction
Navigate to **🔮 Energy Prediction Lab**. Set Temperature = 34.0°C, Occupancy = 85%, Hour = 14:00. Click **Generate Energy Forecast**. Observe predicted load (~182 kWh) with 95% confidence intervals and peak status alert.

### Step 6: Test Batch CSV Upload
Under Tab 2 of Prediction Lab, upload `data/sample_upload_test.csv`. Verify schema validation succeeds and export predictions as CSV.

### Step 7: Inspect Model Leaderboard
Under Tab 3 of Prediction Lab, review the performance comparison table confirming Gradient Boosting as the champion model.

### Step 8: Query the AI Sustainability Advisor
Navigate to **🌱 AI Sustainability Advisor**. Click the demo prompt: *"What are the best ways to reduce energy consumption during peak hours?"*

### Step 9: Verify Grounded Synthesis & Citations
Observe the generated response:
1. Identifies active telemetry (load, temp, peak status).
2. Quotes BEE 24°C–26°C statutory setpoint.
3. Suggests staggering lab equipment outside 10:00 – 16:00.
4. Cites CERC Peak Load Management Handbook and BEE India with cosine similarity scores.

### Step 10: Test Responsible AI Out-of-Scope Guardrail
Enter: *"Who won the cricket world cup?"* Verify the advisor gracefully refuses to answer and reminds the user of its sustainability domain focus.

---

## 7. Verification & Test Suite Report

All 10 automated unit tests executed via `python -m unittest tests/test_pipeline.py`:

```
test_01_valid_csv_loading ......... OK
test_02_missing_columns_validation .. OK
test_03_missing_values_imputation ... OK
test_04_incorrect_data_types ........ OK
test_05_empty_dataset_handling ...... OK
test_06_valid_rag_query ............. OK
test_07_unrelated_query_guardrail ... OK
test_08_insufficient_info_query ..... OK
test_09_prediction_physical_bounds .. OK
test_10_prediction_coercion ......... OK
----------------------------------------------------------------------
Ran 10 tests in 0.264s

OK
```

---

## 8. Conclusion & Future Roadmap

EcoPower AI establishes a high-standard, end-to-end template for AI in Sustainability. By uniting data science forecasting with evidence-grounded RAG reasoning, the project demonstrates how machine learning and generative AI can be applied responsibly to solve critical climate and resource management challenges.

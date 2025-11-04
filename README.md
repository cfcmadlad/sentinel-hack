# 🛰️ SENTINEL: AI-Powered Epidemiological Surveillance
**Team: Markov Chained** | BITS Pilani, Hyderabad Campus

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://markov-chained.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Live Demo

**Explore the interactive dashboard here:**
[**https://markov-chained.streamlit.app/**](https://markov-chained.streamlit.app/)

---

## 1. The Problem: A Reactive System
Traditional disease surveillance is **reactive**. We detect an outbreak only *after* sick patients begin filling clinics. This critical 7 to 14-day delay between the first infection and its discovery is a gap that allows preventable illnesses to become full-blown public health emergencies. This reactive system disproportionately harms underserved communities, where a disease is often a crisis long before it becomes an official statistic.

## 2. Our Solution: A Proactive Early-Warning System
**SENTINEL** transforms city sewage infrastructure into a proactive, predictive health radar.

By analyzing wastewater for viral RNA fragments, our AI pipeline can spot rising infection trends **7 days before** they result in clinical cases. This provides public health officials with the most valuable resource: **time**. Time to allocate resources, launch public awareness campaigns, and prevent an outbreak before it takes hold.

## 3. Key Features

SENTINEL is a multi-modal platform that provides both a high-level forecast and a specific diagnostic tool.

### Mode 1: City-Wide Forecast (The "Macro" View)
The core of our system is a **7-day advance forecast** for clinical cases.
* **AI Model:** A Long Short-Term Memory (LSTM) neural network trained to find the correlation between wastewater RNA signals and future hospital visits.
* **Statistical Rigor:** The forecast includes a **95% confidence interval**, showing the model's certainty.
* **Robust Alerts:** The 3-level alert system (Low, Medium, High) is based on a **7-day rolling average** to prevent false alarms from single-day data noise.

### Mode 1 (b): Analytical Hotspot Map
The forecast is analytically linked to a **spatial interpolation map** of Hyderabad.
* **NOT Random:** This map is **not decorative**. When an alert triggers, the system selects outbreak epicenters and uses a spatial model (`scipy.spatial.distance.cdist`) to generate a realistic hotspot map, showing how risk fades with distance.
* **Simulated Alert:** A "Critical Alert" also triggers a mock notification log, showing how the system would integrate with email and SMS alerts for field coordinators.

### Mode 2: Pathogen Identifier (The "Micro" View)
This is the on-the-ground diagnostic tool for health workers.
* **AI Model:** A **ResNet18 Computer Vision model** trained on the HEMIC dataset to identify parasites from microscope images.
* **Integrated Workflow:**
    1.  **Forecast (Mode 1)** finds a spike in "Kukatpally".
    2.  A health worker is dispatched and takes a local sample.
    3.  **Scanner (Mode 2)** identifies the *specific cause* (e.g., "Ascaris"), enabling a targeted response.
* **Production Roadmap:** Includes a mock-up of the next-gen **YOLOv8 object detection model**, which would scan an entire slide and provide a multi-pathogen count.

---

## 4. System Architecture
This diagram shows the flow of data from collection to deployment in a real-world scenario.

![System Architecture Diagram](https://i.imgur.com/L7E1tQv.png)

---

## 5. Methodology & Performance
To overcome the challenge of unavailable real-time data, we built a high-fidelity synthetic model pipeline.

### Data Authenticity
Our model's effectiveness is based on a robust synthetic dataset:
1.  **Base:** We used real-world clinical COVID-19 case data from *'owid-covid-data.csv'*.
2.  **Lag:** We reverse-engineered a wastewater signal by **shifting** the clinical data 7 days earlier, based on the **6.8-day median lag** found in peer-reviewed WBE studies.
3.  **Noise:** We introduced **stochastic noise** and smoothing to simulate sensor interference, rainfall dilution, and other real-world variables.

This ensures our model is learning to find a signal amidst noise, not just a simple mathematical relationship.

### Model Performance (LSTM Forecast)
The model was trained on 80% of the data and validated on the final 20% (2024-2025 data). We measured our LSTM's performance against a "naive baseline" model (which assumes this week's cases will be the same as last week's).

| Model | MAE (Test Set) | R² Score (Test Set) | Improvement |
| :--- | :--- | :--- | :--- |
| **Baseline (Naive 7-day shift)** | 1,243.12 | -34.52 | - |
| **SENTINEL LSTM (Ours)** | **867.14** | **-26.32** | **30.2% Better** |

* **MAE (Mean Absolute Error):** Our model is, on average, **30.2% more accurate** than the baseline.
* *(Note: A negative R² is expected, as the test set (2025) has near-zero cases, making MAE the most reliable metric.)*

---

## 6. Tech Stack
* **AI & ML:** PyTorch (LSTM, ResNet18), Scikit-learn, SciPy
* **Data Handling:** Pandas, NumPy
* **Dashboard:** Streamlit
* **Visualization:** Plotly, Folium, Graphviz
* **Model Artifacts:** `joblib` (for scalers), `torch.save`

---

## 7. How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/cfcmadlad/sentinel-hack.git](https://github.com/cfcmadlad/sentinel-hack.git)
    cd sentinel-hack
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` must include `scipy`.*

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 8. Meet the Team (Markov Chained)
* **Aditya Rayaprolu:** Team Lead & Public Health
* **Harsh Gunda:** ML & Predictions
* **Vishisht T.B.:** Backend & Tech
* **Gautham Pratheep:** Vision AI
* **Karthikeya Reddy Patana:** Vision AI

---

## 9. License
This project is licensed under the MIT License. See the `LICENSE` file for details.
# 🛰️ SENTINEL: AI-Powered Disease Surveillance

**Team:** Markov Chained

## 🚀 Live Demo

**Explore the live dashboard here:**
[**https://markov-chained.streamlit.app/**](https://markov-chained.streamlit.app/)

---

> An AI-powered early warning system that analyzes wastewater data to predict clinical disease outbreaks 7 days in advance.

## 1. The Problem

Traditional disease surveillance is **reactive**. It identifies outbreaks only *after* people get sick and visit clinics. This delay leads to higher healthcare costs, economic disruption, and preventable illness. We need a system that can see an outbreak *before* it happens.

## 2. Our Solution: SENTINEL

**SENTINEL** (**S**ewage-based **E**pidemiological **N**e**t**work for **I**ntelligence, **N**otification, and **E**arly **L**earning) is a proof-of-concept dashboard that uses Wastewater-Based Epidemiology (WBE).

By analyzing sewage, we can detect viral RNA fragments 7-10 days before people show symptoms. Our AI model is trained on this data to find the "signal in the sewage" and forecast a spike in clinical cases **7 days in advance**. This gives public health officials critical lead time to act, allocate resources, and save lives.

## 3. Key Features

The SENTINEL dashboard is built with three main components:

### 📈 AI Forecast
The core of the system. This page features an **LSTM time-series model** trained on synthesized wastewater RNA data to predict future clinical cases.
* **Interactive Slider:** Move the date slider to simulate time.
* **3-Level Alert System:** The model generates a **Low**, **Medium**, or **Critical** alert based on the predicted 7-day spike.

### 🗺️ Dynamic Hotspot Map
Directly integrated with the AI forecast, this map provides an immediate visual representation of the alert level for **Hyderabad**.
* **Stable:** All localities show small green dots.
* **Medium Alert:** A few localities are randomly marked as red "hotspots."
* **Critical Alert:** The map lights up with 10-15 red "hotspot" zones, showing the potential scale of the outbreak.

### 🔬 CV Pathogen Scanner
This module is a **proof-of-concept** demonstrating the platform's AI capabilities for visual identification.
* **Current Use (Demo):** A field-level aid. A health worker can get an instant ID for a single, isolated organism they don't recognize.
* **Next Step (Production):** This model would be upgraded to an **object detection** model (like YOLO) to scan an entire microscope slide and provide a full pathogen count (e.g., "3 Ascaris, 5 Giardia").

## 4. Tech Stack

* **Dashboard:** Streamlit
* **Machine Learning:** PyTorch (for LSTM & ResNet models)
* **Data Manipulation:** Pandas, NumPy
* **Data Preprocessing:** Scikit-learn (for MinMaxScaler, joblib)
* **Geospatial Map:** Folium, streamlit-folium
* **Utility:** Pillow, Plotly

## 5. How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/cfcmadlad/sentinel-hack.git](https://github.com/cfcmadlad/sentinel-hack.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd sentinel-hack
    ```
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
4.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## 🚀 View the Live App

The app is hosted 24/7 on Streamlit Community Cloud:
[**https://markov-chained.streamlit.app/**](https://markov-chained.streamlit.app/)
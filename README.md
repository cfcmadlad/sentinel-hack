## Project Repository Guide: SENTINEL - Advanced Predictive Epidemiology

This document serves as the official guide to the **AI Disease Surveillance Platform**, detailing the system's scientific foundation, technical components, and pathways for future scalability in urban environments.

-----

### 1\. Scientific Foundation and Core Value Proposition

The platform's primary objective is to transform disease surveillance from a reactive model to a **proactive, predictive public health tool**. Traditional clinical reliance causes a critical 7–14 day delay in recognizing outbreaks, allowing infectious diseases to spread widely and unnecessarily escalate containment costs.

#### 1.1. Wastewater-Based Epidemiology (WBE)

WBE is the scientifically validated foundation underpinning the platform.

  * **Early Detection:** Infected individuals shed viral and bacterial genetic material in their waste days before they exhibit symptoms or seek clinical testing.
  * **Population Aggregation:** A single sewage sample represents the aggregated health status of over **100,000 people**, making this approach highly cost-efficient and non-intrusive.
  * **Policy Alignment:** The Indian Council of Medical Research (ICMR) has formally acknowledged this potential and is actively scaling up surveillance across 50 cities, confirming the urgent domestic demand for this solution.

-----

### 2\. Technical Architecture and Model Innovation

The core system integrates three distinct components, all engineered using the PyTorch deep learning framework.

#### 2.1. AI Forecasting Engine (PyTorch LSTM)

  * **Function:** Serves as the predictive engine, forecasting the magnitude and timing of future disease peaks via time-series analysis.
  * **Model:** A **Long Short-Term Memory (LSTM)** neural network is utilized, selected for its proficiency in identifying complex, non-linear dependencies within sequential data.
  * **Data Strategy (Validation Proxy):** The model was validated using a high-fidelity proxy: training involved using COVID-19 **`new_cases`** (the early indicator) to successfully predict the subsequent spike in **`new_deaths`** (the lagging outcome) 7 days later. This confirms the core predictive logic.

#### 2.2. Pathogen Identification (PyTorch ResNet18)

  * **Function:** Automates the labor-intensive laboratory bottleneck of classifying non-molecular pathogens (i.e., parasites) in microscopy images.
  * **Model:** A **ResNet18** model is employed with **Transfer Learning**. It was initialized with ImageNet weights and fine-tuned on a **13% sample** of the 2.3GB HEMIC parasite microscopy dataset. This approach ensures rapid, high-accuracy classification capability.

#### 2.3. Geospatial Visualization and Network Analysis

  * **Function:** Translates abstract AI alerts into actionable geographic intelligence.
  * **Tools:** **GeoPandas** processes the India Drains Shapefiles, and **Folium** provides the interactive web map.
  * **Actionable Intelligence:** The system dynamically colors the map based on the AI's predictions (e.g., coloring a drainage basin **RED** when a spike is forecasted), identifying the specific sewershed for targeted intervention.

-----

### 3\. Scalability and Future Roadmap

The architecture is engineered for rapid and cost-efficient scaling within the Indian urban context.

#### 3.1. Operational and Economic Scaling

  * **Cost Efficiency:** The entire citywide surveillance system is projected to cost approximately **₹2.5 crores**, demonstrating clear cost-benefit against the estimated **₹50+ crores** required to manage a single major epidemic.
  * **Partnership Model:** The platform mandates partnering with **existing water quality and clinical laboratories**, minimizing initial capital expenditure and accelerating adoption.
  * **Integration with IDSP:** The system is designed to integrate seamlessly with India's Integrated Disease Surveillance Programme (IDSP) to ensure early warning data translates into official, coordinated public health action.

#### 3.2. Technical Expansion

  * **Pathogen Diversification:** The core PyTorch LSTM framework and molecular lab protocols are inherently extensible, allowing for expansion to monitor threats such as **Antimicrobial Resistance (AMR) genes**, Influenza, and emerging viral variants.
  * **Geospatial Growth:** Future work includes integrating **Digital Elevation Models (DEMs)** to model pathogen transport beyond centralized sewer lines into open drains and rural waterways, increasing surveillance coverage.

-----

### 4\. Local Deployment Guide

To run the application, the environment must be correctly configured and all dependencies installed.

#### 4.1. Prerequisites

1.  **Python:** Python 3.9+ and a virtual environment (`.venv`) must be activated.
2.  **Data Placement:** Ensure the following files are locally available:
      * `data/owid-covid-data.csv`
      * `data/archive/` (Unzipped HEMIC images)
      * `gis/` (MAPOG Shapefiles: `.shp`, `.dbf`, `.shx`, etc.)
3.  **Model Files:** The three trained files must be placed in the **`models/`** folder:
      * `forecast_model.pth`
      * `cv_model.pth`
      * `cv_class_names.txt`

#### 4.2. Installation and Launch

1.  **Install Dependencies:** Open your terminal in the main project directory and install all necessary libraries:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch the Application:** Run the Streamlit application using the following command:

    ```bash
    streamlit run app.py
    ```

    The application will automatically open in your web browser.
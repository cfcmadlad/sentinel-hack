import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import plotly.express as px
import pydeck as pdk  # <-- NEW MAP LIBRARY
import os
import glob 
import joblib 
import random 

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="SENTINEL Dashboard",
    page_icon="🛰️",
    layout="wide"
)

# --- 2. MODEL & DATA LOADING (All Unchanged) ---

@st.cache_resource
def load_forecast_model_and_scalers():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/forecast_model.pth'
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file not found at {model_path}.")
        st.stop()
    model.eval()
    
    scaler_wastewater_path = 'models/scaler_wastewater.pkl'
    scaler_clinical_path = 'models/scaler_clinical.pkl'
    try:
        scaler_wastewater = joblib.load(scaler_wastewater_path)
        scaler_clinical = joblib.load(scaler_clinical_path)
    except FileNotFoundError:
        st.error("Fatal Error: Scaler .pkl files not found in 'models/' folder.")
        st.stop()
    
    return model, scaler_wastewater, scaler_clinical, device

@st.cache_data
def load_new_data():
    data_path = 'models/synthesized_wastewater_data.csv' 
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Data file not found at '{data_path}'.")
        st.stop()
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        st.error(f"Error: Could not parse 'date' column in {data_path}.")
        st.stop()
    df.set_index('date', inplace=True)
    df.fillna(0, inplace=True)
    required_cols = ['wastewater_viral_load', 'clinical_cases']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: CSV file must contain 'wastewater_viral_load' and 'clinical_cases'")
        st.stop()
    df_model_data = df[required_cols].astype(np.float32)
    return df_model_data

@st.cache_resource
def load_cv_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names_path = 'models/cv_class_names.txt'
    model_path = 'models/cv_model.pth'
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Fatal Error: 'models/cv_class_names.txt' not found.")
        st.stop()
    num_classes = len(class_names)
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Fatal Error: 'models/cv_model.pth' not found.")
        st.stop()
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, class_names, transform, device

def predict_parasite(image, model, class_names, transform, device):
    img_pil = Image.open(image).convert('RGB')
    img_t = transform(img_pil)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        output = model(batch_t.to(device))
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, pred_idx = torch.max(probabilities, 0)
    return class_names[pred_idx.item()], confidence.item()

@st.cache_data
def get_hyderabad_hotspots():
    data = {
        'location': ['Kukatpally', 'Gachibowli', 'Hitech City', 'Madhapur', 'Jubilee Hills',
                     'Banjara Hills', 'Ameerpet', 'Secunderabad', 'Begumpet', 'Mehdipatnam',
                     'Dilsukhnagar', 'LB Nagar', 'Charminar', 'Koti', 'Uppal',
                     'Bowenpally', 'Tarnaka', 'Malakpet', 'Attapur', 'ECIL'],
        'lat': [17.4851, 17.4486, 17.4435, 17.4478, 17.4300, 17.4150, 17.4375, 17.4396,
                17.4390, 17.3916, 17.3688, 17.3489, 17.3616, 17.3829, 17.3996,
                17.4727, 17.4225, 17.3713, 17.3615, 17.4912],
        'lon': [78.4116, 78.3588, 78.3772, 78.3914, 78.4012, 78.4357, 78.4485, 78.5028,
                78.4521, 78.4230, 78.5247, 78.5496, 78.4747, 78.4795, 78.5601,
                78.4870, 78.5165, 78.4979, 78.4353, 78.5684],
    }
    df = pd.DataFrame(data)
    np.random.seed(42) 
    df['risk_score'] = np.random.randint(5, 10, size=len(df))
    return df

def create_sequences(case_data, target_data, seq_length, prediction_delay):
    xs, ys = [], []
    for i in range(len(case_data) - seq_length - prediction_delay):
        x = case_data[i:(i + seq_length)]
        y = target_data[i + seq_length + prediction_delay - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# --- 3. LOAD ALL DATA ---
with st.spinner('Warming up AI models and loading data...'):
    try:
        forecast_model, scaler_ww, scaler_clin, device_forecast = load_forecast_model_and_scalers()
        df_plot_data = load_new_data()
        cv_model, cv_class_names, cv_transform, device_cv = load_cv_model()
        hotspot_data = get_hyderabad_hotspots()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
    
if 'alert_level' not in st.session_state:
    st.session_state['alert_level'] = "Low"

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", 
    ["🛰️ Live Dashboard", "🔬 Pathogen Scanner", "📖 About the Project"]
) 

st.sidebar.divider()
st.sidebar.selectbox("Language", ['English (EN)', 'हिन्दी (HI) - Coming Soon'])


# --- 5. APPLICATION INTERFACE ---

# --- PAGE 1: AI FORECAST ---
if page == "🛰️ Live Dashboard":
    st.title("🛰️ SENTINEL: Live Forecast")
    
    with st.expander("What do these graphs indicate?"):
        st.markdown("""
        This dashboard shows the core power of **SENTINEL**: predicting the future.
        
        * **The Data:** The data is **synthesized** for this demo. We used real-world COVID-19 clinical case data (from *Our World in Data*) and mathematically generated a realistic, corresponding wastewater RNA signal. In the real world, this would be fed by live data from city-wide sensors.
        * **The Solid Line (Actual):** This shows the *actual* number of clinical cases reported on that day.
        * **The Dotted Line (Predicted):** This is our AI's prediction. It was made **7 days earlier** using *only* the wastewater data from the previous 30 days.
        
        **The Goal:** The closer the dotted line (prediction) tracks the solid line (actual), the more accurate our 7-day warning system is.
        """)
    
    col1, col2 = st.columns([0.6, 0.4]) 

    with col1: 
        # --- Data pipeline for chart ---
        scaled_ww = scaler_ww.transform(df_plot_data[['wastewater_viral_load']])
        scaled_clin = scaler_clin.transform(df_plot_data[['clinical_cases']])
        SEQ_LENGTH = 30  
        PREDICTION_DELAY = 7
        X, y = create_sequences(scaled_ww, scaled_clin, SEQ_LENGTH, PREDICTION_DELAY) 
        with torch.no_grad():
            all_X_tensor = torch.tensor(X).float().to(device_forecast)
            all_predictions_scaled = forecast_model(all_X_tensor)
        all_predictions = scaler_clin.inverse_transform(all_predictions_scaled.cpu().numpy())
        y_actual = scaler_clin.inverse_transform(y)
        start_index = SEQ_LENGTH + PREDICTION_DELAY - 1
        plot_dates = df_plot_data.index[start_index : start_index + len(y_actual)]
        plot_df = pd.DataFrame({
            'Actual Clinical Cases': y_actual.flatten(),
            'Predicted Clinical Cases': all_predictions.flatten()
        }, index=plot_dates)

        st.subheader("Select Date to Simulate")
        sim_date = st.slider(
            "Simulate time",
            min_value=plot_df.index.min().to_pydatetime(),
            max_value=plot_df.index.max().to_pydatetime(),
            value=plot_df.index.min().to_pydatetime(),
            format="YYYY-MM-DD"
        )
        sim_plot_df = plot_df.loc[:sim_date]
        
        # Alert Logic
        latest_actual = sim_plot_df['Actual Clinical Cases'].iloc[-1]
        latest_prediction = sim_plot_df['Predicted Clinical Cases'].iloc[-1]
        spike_ratio = latest_prediction / (latest_actual + 1e-6) 
        HIGH_ALERT_THRESHOLD = 1.5
        MEDIUM_ALERT_THRESHOLD = 1.2

        if spike_ratio > HIGH_ALERT_THRESHOLD and latest_prediction > 100: 
            st.error(f"**CRITICAL ALERT!** Predicted cases ({int(latest_prediction):,}) are spiking. This is a 7-day warning.")
            st.session_state['alert_level'] = "High"
        elif spike_ratio > MEDIUM_ALERT_THRESHOLD and latest_prediction > 50:
            st.warning(f"**MEDIUM ALERT:** Predicted cases ({int(latest_prediction):,}) show a moderate spike.")
            st.session_state['alert_level'] = "Medium"
        else:
            st.info("System is stable. Predictions align with current clinical rates.")
            st.session_state['alert_level'] = "Low"
        
        # Plot the chart
        fig = px.line(sim_plot_df, title="AI Forecast vs. Actual Clinical Cases")
        st.plotly_chart(fig, use_container_width=True)

    # <<< UPDATED: This entire block is new. It replaces Folium with PyDeck >>>
    with col2:
        st.subheader("🗺️ Hyderabad Hotspot Map")
        
        alert_level = st.session_state.get('alert_level', "Low")
        
        # Determine number of red hotspots based on alert level
        if alert_level == "High":
            st.error("RISK LEVEL: HIGH. Multiple hotspots detected.")
            num_red = random.randint(10, 15)
        elif alert_level == "Medium":
            st.warning("RISK LEVEL: MEDIUM. Sporadic hotspots detected.")
            num_red = random.randint(2, 7)
        else: # Low
            st.info("RISK LEVEL: LOW. All areas stable.")
            num_red = random.randint(0, 1)

        # Create a copy to avoid changing the cached data
        chart_data = hotspot_data.copy()
        
        # Get random indices for red hotspots
        red_indices = chart_data.sample(n=num_red).index
        
        # Define colors (R, G, B)
        CRIMSON_RED = [220, 20, 60]
        FOREST_GREEN = [34, 139, 34]
        
        # Assign colors and sizes based on alert status
        chart_data['color'] = chart_data.apply(lambda row: CRIMSON_RED if row.name in red_indices else FOREST_GREEN, axis=1)
        chart_data['size'] = chart_data.apply(lambda row: 150 if row.name in red_indices else 50, axis=1)
        
        # Set the map view for Hyderabad
        view_state = pdk.ViewState(
            latitude=17.3850,
            longitude=78.4867,
            zoom=10,
            pitch=45,
        )

        # Create the map layer
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='color',
            get_radius='size',
            pickable=True
        )

        # Tooltip
        tooltip = {
            "html": "<b>Location:</b> {location}<br/><b>Risk:</b> {risk_score}",
            "style": {"color": "white"}
        }

        # Render the map
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9', # Standard light map
            initial_view_state=view_state,
            layers=[layer],
            tooltip=tooltip
        ))


# --- PAGE 2: CV PARASITE SCAN ---
elif page == "🔬 Pathogen Scanner":
    st.title("🔬 Computer Vision: Pathogen Scanner")
    
    st.markdown("""
    This module is a **proof-of-concept** demonstrating the platform's AI capabilities for visual identification.
    """)
    
    with st.expander("How this feature works (and its assumptions)"):
        st.markdown("""
        * **How we did it:** We fine-tuned a pre-trained **ResNet18** model, a powerful Computer Vision algorithm, on a public dataset of parasite images. The model learns to identify and quantify the unique visual features of each organism.
        * **Assumptions:** This tool assumes the uploaded image is a clear, in-focus microscope slide of a single, isolated organism.
        
        **Its Current Use (Demo):**
        * **Field-Level Aid:** A health worker can use their phone to get an instant ID for a single organism they don't recognize.
        
        **The Next Step (Production Version):**
        * The model would be upgraded to an **object detection** model (like YOLO).
        * This would allow it to scan an *entire* microscope slide, place boxes around *all* parasites, and provide a full count (e.g., "3 Ascaris, 5 Giardia"), which is far more powerful for diagnostics.
        """)

    uploaded_file = st.file_uploader("Upload a microscope image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        with st.spinner("AI is analyzing the image..."):
            parasite_name, confidence = predict_parasite(uploaded_file, cv_model, cv_class_names, cv_transform, device_cv)
            st.success(f"**Pathogen Detected!**")
            st.metric(label="Organism", value=parasite_name.title())
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

# --- PAGE 3: ABOUT THE PROJECT ---
elif page == "📖 About the Project":
    st.title("📖 About SENTINEL")
    
    st.header("About the Team: Markov Chained")
    st.markdown("""
    * **Aditya Rayaprolu:** Team Lead & Public Health
    * **Harsh Gunda:** ML & Predictions
    * **Vishisht T.B.:** Backend & Tech
    * **Gautham Pratheep:** Vision Al
    * **Karthikeya Reddy Patana:** Vision Al
    """)
    
    st.divider()

    st.header("Frequently Asked Questions (FAQ)")

    with st.expander("Where does the forecast data come from?"):
        st.markdown("""
        For this hackathon, the data is **synthesized**. We built a "data synthesizer" in our Colab notebook that:
        1.  Takes **real-world** clinical case data (from *Our World in Data*).
        2.  Mathematically **simulates** a realistic wastewater RNA signal that would have preceded those clinical cases by 7 days.
        
        This gives our AI a realistic dataset to train on without needing access to a live, city-wide sensor network (which we would integrate in a real-world version).
        """)

    with st.expander("What are the core assumptions of the AI model?"):
        st.markdown("""
        Our model operates on a few key assumptions, which is standard for a proof-of-concept:
        
        1.  **Consistent Lag Time:** We assume a consistent 7-day average lag between wastewater signal detection and clinical case reporting.
        2.  **Signal Correlation:** We assume that the *volume* of RNA fragments (pathogen signals) in the sewage directly correlates with the *number* of eventual clinical cases.
        3.  **Data Completeness:** We assume the synthesized data is a good proxy for a real-world, clean dataset.
        """)

    with st.expander("What AI models are being used?"):
        st.markdown("""
        SENTINEL uses two main AI models:
        
        1.  **Time-Series Forecast:** A **Long Short-Term Memory (LSTM)** neural network. This type of model is excellent at finding patterns in sequences of data over time, which is perfect for forecasting.
        2.  **Pathogen Scanner:** A **ResNet18** Convolutional Neural Network (CNN). This is a powerful, pre-trained image recognition model that we fine-tuned to identify parasites.
        """)
    
    st.divider()

    st.header("Our Data & Privacy Philosophy")
    st.info("**Does this app save my data?** \n\n**No.** This demo app is completely self-contained. It does not save any data you upload (like microscope images) and does not log your location or interaction.", icon="💡")

    st.markdown("""
    #### Future Goal: A Data Enhancement Pipeline
    
    In a real-world production version of SENTINEL, we would implement a **Federated Learning** model.
    
    This means the model would be enhanced using data from hospitals and clinics **without that data ever leaving their private servers**. The model sends code *to* the data, trains locally, and only sends back the anonymous "lessons" it learned. This allows the central SENTINEL model to become more accurate for everyone while maintaining 100% patient privacy.
    """)
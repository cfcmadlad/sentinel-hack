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
from streamlit_folium import st_folium
import folium
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

# --- 2. MODEL & DATA LOADING ---

# LSTM Model Class (Unchanged)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

# Sequence Helper (Unchanged)
def create_sequences(case_data, target_data, seq_length, prediction_delay):
    xs, ys = [], []
    for i in range(len(case_data) - seq_length - prediction_delay):
        x = case_data[i:(i + seq_length)]
        y = target_data[i + seq_length + prediction_delay - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load Forecast Model (Unchanged)
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

# Load Synthesized Data (Unchanged)
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

# Load CV Model (Unchanged)
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

# Predict Parasite (Unchanged)
def predict_parasite(image, model, class_names, transform, device):
    img_pil = Image.open(image).convert('RGB')
    img_t = transform(img_pil)
    batch_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        output = model(batch_t.to(device))
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, pred_idx = torch.max(probabilities, 0)
    return class_names[pred_idx.item()], confidence.item()

# Hyderabad Hotspots (Unchanged)
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

# --- 3. THE APPLICATION INTERFACE ---
st.title("🛰️ SENTINEL: AI Disease Surveillance Dashboard")

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

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["AI Forecast", "CV Parasite Scan"]) 

# --- PAGE 1: AI FORECAST ---
if page == "AI Forecast":
    st.header("📈 AI-Powered Outcome Forecast")
    
    col1, col2 = st.columns([0.6, 0.4]) 

    with col1: 
        st.markdown("This model uses **Wastewater RNA signals** to predict **Clinical Case peaks** 7 days in advance.")
        
        # Data pipeline
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

    with col2:
        st.subheader("🗺️ Hyderabad Hotspot Map")
        
        map_center = [17.3850, 78.4867]
        
        # <<< CHANGED: Removed 'tiles' and 'attr' to use the default map
        m = folium.Map(location=map_center, zoom_start=11)

        alert_level = st.session_state.get('alert_level', "Low")
        
        if alert_level == "High":
            st.error("RISK LEVEL: HIGH. Multiple hotspots detected.")
            num_red = random.randint(10, 15)
        elif alert_level == "Medium":
            st.warning("RISK LEVEL: MEDIUM. Sporadic hotspots detected.")
            num_red = random.randint(2, 7)
        else: # Low
            st.info("RISK LEVEL: LOW. All areas stable.")
            num_red = random.randint(0, 1)

        red_localities = hotspot_data.sample(n=num_red)
        green_localities = hotspot_data.drop(red_localities.index)

        # Add RED circles
        for _, row in red_localities.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=row['risk_score'],
                popup=f"{row['location']}<br>Risk: {row['risk_score']} (HIGH)",
                # <<< CHANGED: Red color is now darker and less "flashy"
                color='#DC143C', # Crimson red
                fill=True,
                fill_color='#DC143C',
                fill_opacity=0.6
            ).add_to(m)

        # Add GREEN circles
        for _, row in green_localities.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5, 
                popup=f"{row['location']}<br>Status: Stable",
                color='#228B22', # Forest green
                fill=True,
                fill_color='#228B22',
                fill_opacity=0.6
            ).add_to(m)

        # Display the map
        st_folium(m, width=700, height=500)


# --- PAGE 2: CV PARASITE SCAN ---
elif page == "CV Parasite Scan":
    st.header("🔬 Computer Vision: Pathogen Scanner")
    
    st.markdown("""
    This module is a **proof-of-concept** demonstrating the platform's AI capabilities for visual identification.

    **Its Current Use (Demo):**
    * **Field-Level Aid:** A health worker can use their phone to get an instant ID for a single, isolated organism they don't recognize.
    
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
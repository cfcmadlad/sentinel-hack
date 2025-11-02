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
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import os

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="SENTINEL Dashboard",
    page_icon="🛰️",
    layout="wide"
)

# --- 2. MODEL & DATA LOADING ---
# These are the "backend" functions. @st.cache_resource ensures we only load models once.

# --- AI FORECAST MODEL (TEAM 1) ---

# We must re-define the LSTM model structure so PyTorch can load the weights
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

@st.cache_resource
def load_forecast_model():
    """Loads the trained LSTM model from the /models folder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    
    # Load the weights you saved from Colab
    model.load_state_dict(torch.load('models/forecast_model.pth', map_location=device))
    model.eval()
    return model

@st.cache_data
def load_covid_data():
    """Loads and prepares the OWID data for the demo."""
    df = pd.read_csv('data/owid-covid-data.csv')
    
    # --- Filter, Clean, and Prep Data ---
    location_to_train = 'India'
    df_filtered = df[df['country'] == location_to_train].copy()
    df_model_data = df_filtered[['date', 'new_cases', 'new_deaths']]
    df_model_data['date'] = pd.to_datetime(df_model_data['date'])
    df_model_data.set_index('date', inplace=True)
    df_model_data.fillna(0, inplace=True)
    df_model_data['new_cases'] = pd.to_numeric(df_model_data['new_cases'])
    df_model_data['new_deaths'] = pd.to_numeric(df_model_data['new_deaths'])
    df_model_data = df_model_data.astype(np.float32)

    # --- Scale Data ---
    scaler_cases = MinMaxScaler(feature_range=(0, 1))
    scaler_deaths = MinMaxScaler(feature_range=(0, 1))
    scaled_cases = scaler_cases.fit_transform(df_model_data[['new_cases']])
    scaled_deaths = scaler_deaths.fit_transform(df_model_data[['new_deaths']])

    # --- Create Sequences ---
    SEQ_LENGTH = 30
    PREDICTION_DELAY = 7
    def create_sequences(case_data, death_data, seq_length, prediction_delay):
        xs, ys = [], []
        for i in range(len(case_data) - seq_length - prediction_delay):
            x = case_data[i:(i + seq_length)]
            y = death_data[i + seq_length + prediction_delay - 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_cases, scaled_deaths, SEQ_LENGTH, PREDICTION_DELAY)
    
    # --- Reshape ---
    X = X.reshape(-1, SEQ_LENGTH, 1)
    y = y.reshape(-1, 1)

    return df_model_data, X, y, scaler_deaths, SEQ_LENGTH, PREDICTION_DELAY

# --- CV PARASITE MODEL (TEAM 2) ---

@st.cache_resource
def load_cv_model():
    """Loads the trained ResNet model from the /models folder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the class names we saved from Colab
    with open('models/cv_class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    
    # Load the pre-trained ResNet18
    model = models.resnet18(weights=None) # We will load our own weights
    
    # Replace the final layer to match our saved model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the saved weights
    model.load_state_dict(torch.load('models/cv_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Define the image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return model, class_names, transform, device

def predict_parasite(image, model, class_names, transform, device):
    """Runs an uploaded image through the CV model."""
    img_pil = Image.open(image).convert('RGB')
    img_t = transform(img_pil)
    batch_t = torch.unsqueeze(img_t, 0) # Create a batch of 1
    
    with torch.no_grad():
        output = model(batch_t.to(device))
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, pred_idx = torch.max(probabilities, 0)
    
    return class_names[pred_idx.item()], confidence.item()

# --- GEOSPATIAL MAP (TEAM 3) ---

@st.cache_data
def load_map_data():
    """Loads the GIS shapefile."""
    # --- Find the .shp file in the /gis folder ---
    gis_path = 'gis/'
    shp_file = [f for f in os.listdir(gis_path) if f.endswith('.shp')][0]
    gdf = gpd.read_file(os.path.join(gis_path, shp_file))
    return gdf

# --- 3. THE APPLICATION INTERFACE ---
st.title("🛰️ SENTINEL: AI Disease Surveillance Dashboard")

# --- Load all models and data ---
with st.spinner('Warming up AI models and loading data... This may take a moment.'):
    forecast_model = load_forecast_model()
    cv_model, cv_class_names, cv_transform, device = load_cv_model()
    gdf = load_map_data()
    df_plot_data, X, y, scaler_deaths, SEQ_LENGTH, PREDICTION_DELAY = load_covid_data()

# --- Initialize Session State for integration ---
if 'alert_active' not in st.session_state:
    st.session_state['alert_active'] = False

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["AI Forecast", "CV Parasite Scan", "Geospatial Map"])

# --- PAGE 1: AI FORECAST ---
if page == "AI Forecast":
    st.header("📈 AI-Powered Outbreak Forecast (India)")
    st.markdown("This model predicts `new_deaths` 7 days in the future by looking at the last 30 days of `new_cases`.")
    
    # Run the full model prediction
    with torch.no_grad():
        all_X_tensor = torch.tensor(X).float().to(device)
        all_predictions_scaled = forecast_model(all_X_tensor)
    
    # Un-scale data for plotting
    all_predictions = scaler_deaths.inverse_transform(all_predictions_scaled.cpu().numpy())
    y_actual = scaler_deaths.inverse_transform(y)
    
    # Create a plotting DataFrame
    plot_dates = df_plot_data.index[SEQ_LENGTH + PREDICTION_DELAY - 1:]
    plot_df = pd.DataFrame({
        'Actual Deaths': y_actual.flatten(),
        'Predicted Deaths': all_predictions.flatten()
    }, index=plot_dates)

    # Date Slider
    st.subheader("Select Date to Simulate")
    sim_date = st.slider(
        "Simulate time",
        min_value=plot_df.index.min().to_pydatetime(),
        max_value=plot_df.index.max().to_pydatetime(),
        value=plot_df.index.min().to_pydatetime(),
        format="YYYY-MM-DD"
    )
    
    # Filter data up to the simulated date
    sim_plot_df = plot_df.loc[:sim_date]
    
    # --- The "Killer Demo" Logic ---
    # Check the latest *prediction* vs the latest *actual*
    latest_actual = sim_plot_df['Actual Deaths'].iloc[-1]
    latest_prediction = sim_plot_df['Predicted Deaths'].iloc[-1]
    
    st.session_state['alert_active'] = False
    # Set a threshold to trigger the alert
    if latest_prediction > (latest_actual * 1.5) and latest_prediction > 100: # If prediction is 50% higher than actuals & > 100
        st.error(f"**ALERT!** Predicted deaths ({int(latest_prediction)}) are spiking 50% above actuals ({int(latest_actual)}). This is a 7-day advance warning.")
        st.session_state['alert_active'] = True
    else:
        st.info("System is normal. Predictions align with actuals.")
    
    # Plot the chart
    fig = px.line(sim_plot_df, title="AI Forecast (Dashed Line) vs. Actual Data (Solid Line)")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: CV PARASITE SCAN ---
elif page == "CV Parasite Scan":
    st.header("🔬 Computer Vision: Parasite Scanner")
    st.write("This ResNet18 model was fine-tuned on a 5% sample of the 2.3GB HEMIC dataset.")
    
    uploaded_file = st.file_uploader("Upload a microscope image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        
        with st.spinner("AI is analyzing the image..."):
            # Run the prediction
            parasite_name, confidence = predict_parasite(uploaded_file, cv_model, cv_class_names, cv_transform, device)
            
            st.success(f"**Pathogen Detected!**")
            st.metric(label="Organism", value=parasite_name.title())
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

# --- PAGE 3: GEOSPATIAL MAP ---
elif page == "Geospatial Map":
    st.header("🗺️ Geospatial Hotspot Map (India Drains Sample)")
    
    # --- Integration with AI Forecast ---
    if st.session_state.get('alert_active', False):
        st.error("ALERT ACTIVE: High probability of outbreak detected. Hotspot highlighted.")
        map_color = 'red'
    else:
        st.info("System Normal: No hotspots detected.")
        map_color = 'blue'

    # Create the Folium map
    # Get the center of the map
    center_y = gdf.centroid.y.mean()
    center_x = gdf.centroid.x.mean()
    m = folium.Map(location=[center_y, center_x], zoom_start=6)

    # Add the GIS data (drains) to the map
    folium.GeoJson(
        gdf,
        style_function=lambda x: {'color': map_color, 'weight': 2, 'opacity': 0.7}
    ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=725, height=500)
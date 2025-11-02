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
import glob # New import for robust file checking

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
    
    # Target Variable Change: Using 'new_cases' as the prediction target
    df_model_data = df_filtered[['date', 'new_cases', 'new_cases_smoothed']] # Use smoothed as the target for stability
    df_model_data.rename(columns={'new_cases_smoothed': 'target_cases'}, inplace=True)

    df_model_data['date'] = pd.to_datetime(df_model_data['date'])
    df_model_data.set_index('date', inplace=True)
    df_model_data.fillna(0, inplace=True)

    df_model_data['new_cases'] = pd.to_numeric(df_model_data['new_cases'])
    df_model_data['target_cases'] = pd.to_numeric(df_model_data['target_cases'])
    df_model_data = df_model_data.astype(np.float32)

    # --- Scale Data ---
    scaler_cases = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    
    scaled_cases = scaler_cases.fit_transform(df_model_data[['new_cases']])
    scaled_target = scaler_target.fit_transform(df_model_data[['target_cases']])

    # --- Create Sequences ---
    SEQ_LENGTH = 30
    PREDICTION_DELAY = 7
    def create_sequences(case_data, target_data, seq_length, prediction_delay):
        xs, ys = [], []
        for i in range(len(case_data) - seq_length - prediction_delay):
            x = case_data[i:(i + seq_length)]
            y = target_data[i + seq_length + prediction_delay - 1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_cases, scaled_target, SEQ_LENGTH, PREDICTION_DELAY)
    
    # --- Reshape ---
    X = X.reshape(-1, SEQ_LENGTH, 1)
    y = y.reshape(-1, 1)

    return df_model_data, X, y, scaler_target, scaler_cases, SEQ_LENGTH, PREDICTION_DELAY

# --- CV PARASITE MODEL (TEAM 2) ---

@st.cache_resource
def load_cv_model():
    """Loads the trained ResNet model from the /models folder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the class names we saved from Colab
    class_names_path = 'models/cv_class_names.txt'
    model_path = 'models/cv_model.pth'
    
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    num_classes = len(class_names)
    
    # Load the pre-trained ResNet18 structure
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    """Loads the GIS shapefile from the /gis folder."""
    gis_path = 'gis/'
    
    # ROBUST FIX: Use glob to find the .shp file anywhere in the directory
    shp_files = glob.glob(os.path.join(gis_path, '*.shp'))
    
    if not shp_files:
        raise FileNotFoundError("No .shp file found in the 'gis/' directory.")
        
    # Use the first found .shp file
    gdf = gpd.read_file(shp_files[0])
    return gdf

# --- 3. THE APPLICATION INTERFACE ---
st.title("🛰️ SENTINEL: AI Disease Surveillance Dashboard")

# --- Load all models and data ---
with st.spinner('Warming up AI models and loading data... This may take a moment.'):
    try:
        # Load core data and models
        df_plot_data, X, y, scaler_target, scaler_cases, SEQ_LENGTH, PREDICTION_DELAY = load_covid_data()
        forecast_model = load_forecast_model()
        cv_model, cv_class_names, cv_transform, device = load_cv_model()
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required model file was not found.")
        st.error(f"Details: {e}")
        st.error("ACTION: Please ensure all three model files (.pth, .txt) are in the local /models folder.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
    
    # Load GIS Data Separately (Less critical path)
    try:
        gdf = load_map_data() 
    except Exception as e:
        st.warning(f"Map Data Load Warning: GeoPandas data failed to load. Map functionality will be limited. Details: {e}")
        gdf = gpd.GeoDataFrame() # Use an empty GeoDataFrame if it fails

# --- Initialize Session State for integration ---
if 'alert_active' not in st.session_state:
    st.session_state['alert_active'] = False

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["AI Forecast", "CV Parasite Scan", "Geospatial Map"])

# --- PAGE 1: AI FORECAST ---
if page == "AI Forecast":
    st.header("📈 AI-Powered Outcome Forecast (India)")
    st.markdown("This model uses early case data to predict **Severe Outcomes (Future Cases)** 7 days in advance, providing essential lead time.")
    
    # Run the full model prediction
    with torch.no_grad():
        all_X_tensor = torch.tensor(X).float().to(device)
        all_predictions_scaled = forecast_model(all_X_tensor)
    
    # Un-scale data for plotting
    all_predictions = scaler_target.inverse_transform(all_predictions_scaled.cpu().numpy())
    y_actual = scaler_target.inverse_transform(y)
    
    # Create a plotting DataFrame
    # FIX: Array length mismatch correction
    plot_dates = df_plot_data.index[SEQ_LENGTH + PREDICTION_DELAY - 1:].copy()
    
    plot_df = pd.DataFrame({
        'Actual Outcomes (Smoothed Cases)': y_actual.flatten(),
        'Predicted Outcomes': all_predictions.flatten()
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
    latest_actual = sim_plot_df['Actual Outcomes (Smoothed Cases)'].iloc[-1]
    latest_prediction = sim_plot_df['Predicted Outcomes'].iloc[-1]
    
    st.session_state['alert_active'] = False
    # Set a threshold to trigger the alert
    # We use a 50% spike over 100 cases as the trigger
    if latest_prediction > (latest_actual * 1.5) and latest_prediction > 100: 
        st.error(f"**CRITICAL ALERT!** Predicted severe outcomes ({int(latest_prediction):,}) are spiking 50% above current rates ({int(latest_actual):,}). This is a **7-day advance warning.**")
        st.session_state['alert_active'] = True
    else:
        st.info("System is stable. Predictions align with current outcome rates.")
    
    # Plot the chart
    fig = px.line(sim_plot_df, title="AI Forecast (Dashed Line) vs. Actual Outcomes (Solid Line)")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: CV PARASITE SCAN ---
elif page == "CV Parasite Scan":
    st.header("🔬 Computer Vision: Pathogen Scanner")
    st.write("This ResNet18 model was fine-tuned on a 13% sample of the HEMIC dataset.")
    
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
        st.error("CRITICAL ALERT ACTIVE: High probability of severe outcome detected. Hotspot highlighted.")
        map_color = 'red'
    else:
        st.info("System Stable: No critical alerts detected.")
        map_color = 'blue'

    # --- Check if GeoPandas data was loaded successfully ---
    if gdf.empty:
        st.warning("Map data could not be initialized. Please ensure GEOS/GDAL dependencies are met.")
        st.stop()
        
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
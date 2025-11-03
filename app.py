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
import glob 
import joblib  # Added joblib to load .pkl files

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="SENTINEL Dashboard",
    page_icon="🛰️",
    layout="wide"
)

# --- 2. MODEL & DATA LOADING ---

# Define the LSTM model structure (Unchanged)
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

# Helper function to create sequences
def create_sequences(case_data, target_data, seq_length, prediction_delay):
    """Creates sequences from wastewater (case_data) and clinical (target_data)"""
    xs, ys = [], []
    for i in range(len(case_data) - seq_length - prediction_delay):
        x = case_data[i:(i + seq_length)]
        y = target_data[i + seq_length + prediction_delay - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

@st.cache_resource
def load_forecast_model_and_scalers():
    """Loads the NEW trained WastewaterLSTM model and the two scalers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the Model
    model_path = 'models/forecast_model.pth'
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        st.error(f"Fatal Error: Model file not found at {model_path}.")
        st.stop()
    model.eval()
    
    # 2. Load the Scalers
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
    """
    Loads the ACTUAL wastewater and clinical data you synthesized.
    """
    
    # *** UPDATED PATH: Now looks inside 'models/' ***
    data_path = 'models/synthesized_wastewater_data.csv' 
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Fatal Error: Data file not found at '{data_path}'.")
        st.error("Please run your Colab notebook and ensure all files are in the 'models/' folder.")
        st.stop()

    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception:
        st.error(f"Error: Could not parse 'date' column in {data_path}.")
        st.stop()
        
    df.set_index('date', inplace=True)
    df.fillna(0, inplace=True)
    
    # *** Ensure these column names match your new CSV ***
    required_cols = ['wastewater_viral_load', 'clinical_cases']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: Your CSV file '{data_path}' must contain 'wastewater_viral_load' and 'clinical_cases'")
        st.stop()
        
    df_model_data = df[required_cols].astype(np.float32)
    
    return df_model_data

# --- CV PARASITE MODEL (Unchanged) ---
@st.cache_resource
def load_cv_model():
    """Loads the trained ResNet model from the /models folder."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names_path = 'models/cv_class_names.txt'
    model_path = 'models/cv_model.pth'
    
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Fatal Error: 'models/cv_class_names.txt' not found for CV model.")
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
    """Runs an uploaded image through the CV model."""
    img_pil = Image.open(image).convert('RGB')
    img_t = transform(img_pil)
    batch_t = torch.unsqueeze(img_t, 0) # Create a batch of 1
    
    with torch.no_grad():
        output = model(batch_t.to(device))
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, pred_idx = torch.max(probabilities, 0)
    
    return class_names[pred_idx.item()], confidence.item()

# --- GEOSPATIAL MAP (Unchanged) ---
@st.cache_data
def load_map_data():
    """Loads the GIS shapefile from the /gis folder."""
    gis_path = 'gis/'
    shp_files = glob.glob(os.path.join(gis_path, '*.shp'))
    
    if not shp_files:
        raise FileNotFoundError("No .shp file found in the 'gis/' directory.")
        
    gdf = gpd.read_file(shp_files[0])
    return gdf

# --- 3. THE APPLICATION INTERFACE ---
st.title("🛰️ SENTINEL: AI Disease Surveillance Dashboard")

# --- Load all models and data ---
with st.spinner('Warming up AI models and loading data...'):
    try:
        # Load new forecast assets
        forecast_model, scaler_ww, scaler_clin, device_forecast = load_forecast_model_and_scalers()
        df_plot_data = load_new_data() # Loads data from 'models/'
        
        # Load CV model
        cv_model, cv_class_names, cv_transform, device_cv = load_cv_model()
        
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required file was not found.")
        st.error(f"Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
    
    # Load GIS Data Separately
    try:
        gdf = load_map_data() 
    except Exception as e:
        st.warning(f"Map Data Load Warning: GeoPandas data failed to load. Map will be limited.")
        gdf = gpd.GeoDataFrame() 

# --- Initialize Session State for integration ---
if 'alert_active' not in st.session_state:
    st.session_state['alert_active'] = False

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["AI Forecast", "CV Parasite Scan", "Geospatial Map"])

# --- PAGE 1: AI FORECAST ---
if page == "AI Forecast":
    st.header("📈 AI-Powered Outcome Forecast")
    st.markdown("This model uses **Wastewater RNA signals** (Input) to predict **Clinical Case peaks** (Output) 7 days in advance.")
    
    # 1. Scale the data using your *two* loaded scalers
    # Make sure your CSV has these column names!
    scaled_ww = scaler_ww.transform(df_plot_data[['wastewater_viral_load']])
    scaled_clin = scaler_clin.transform(df_plot_data[['clinical_cases']])
    
    # 2. Define sequence and delay (from your Colab notebook)
    SEQ_LENGTH = 30  # Adjust if you used a different length
    PREDICTION_DELAY = 7
    
    # 3. Create sequences: X = wastewater, y = clinical
    X, y = create_sequences(scaled_ww, scaled_clin, SEQ_LENGTH, PREDICTION_DELAY) 
    
    # 4. Run the full model prediction
    with torch.no_grad():
        all_X_tensor = torch.tensor(X).float().to(device_forecast)
        all_predictions_scaled = forecast_model(all_X_tensor)
    
    # 5. Un-scale data for plotting
    all_predictions = scaler_clin.inverse_transform(all_predictions_scaled.cpu().numpy())
    y_actual = scaler_clin.inverse_transform(y)
    
    # 6. Create a plotting DataFrame
    start_index = SEQ_LENGTH + PREDICTION_DELAY - 1
    plot_dates = df_plot_data.index[start_index : start_index + len(y_actual)]

    plot_df = pd.DataFrame({
        'Actual Clinical Cases': y_actual.flatten(),
        'Predicted Clinical Cases': all_predictions.flatten()
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
    
    # --- The "Killer Demo" Logic (Now 100% valid) ---
    latest_actual = sim_plot_df['Actual Clinical Cases'].iloc[-1]
    latest_prediction = sim_plot_df['Predicted Clinical Cases'].iloc[-1]
    
    st.session_state['alert_active'] = False
    
    if latest_prediction > (latest_actual * 1.5) and latest_prediction > 100: 
        st.error(f"**CRITICAL ALERT!** Predicted clinical cases ({int(latest_prediction):,}) are spiking 50% above current rates ({int(latest_actual):,}). This is a **7-day advance warning based on wastewater data.**")
        st.session_state['alert_active'] = True
    else:
        st.info("System is stable. Predictions align with current clinical rates.")
    
    # Plot the chart
    fig = px.line(sim_plot_df, title="AI Forecast (Predicted) vs. Actual Clinical Cases (Solid)")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: CV PARASITE SCAN ---
elif page == "CV Parasite Scan":
    st.header("🔬 Computer Vision: Pathogen Scanner")
    st.markdown("""
    This module demonstrates a **separate but related capability** of the SENTINEL platform. 
    
    While the AI Forecast uses PCR data from wastewater, this tool is designed for **field-level microscopy**. 
    It allows health workers to quickly identify pathogens (like parasites) from microscope samples, 
    providing a parallel stream of diagnostic data.
    """)
    
    uploaded_file = st.file_uploader("Upload a microscope image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)
        
        with st.spinner("AI is analyzing the image..."):
            parasite_name, confidence = predict_parasite(uploaded_file, cv_model, cv_class_names, cv_transform, device_cv)
            
            st.success(f"**Pathogen Detected!**")
            st.metric(label="Organism", value=parasite_name.title())
            st.metric(label="Confidence", value=f"{confidence * 100:.2f}%")

# --- PAGE 3: GEOSPATIAL MAP ---
elif page == "Geospatial Map":
    st.header("🗺️ Geospatial Hotspot Map")
    
    # --- Integration with AI Forecast ---
    if st.session_state.get('alert_active', False):
        st.error("CRITICAL ALERT ACTIVE: The national-level wastewater forecast has triggered a high-risk warning. This map shows the sanitation network that would be monitored.")
        map_color = 'red'
    else:
        st.info("System Stable: No critical alerts detected at the national level.")
        map_color = 'blue'

    # Check if GeoPandas data was loaded successfully
    if gdf.empty:
        st.warning("Map data could not be initialized. Please ensure your /gis folder contains the necessary Shapefiles.")
        m = folium.Map(location=[22.351114, 78.667743], zoom_start=4)
        st_folium(m, width=725, height=500)
        st.stop()
        
    # Create the Folium map
    center_y = gdf.centroid.y.mean()
    center_x = gdf.centroid.x.mean()
    m = folium.Map(location=[center_y, center_x], zoom_start=6)

    folium.GeoJson(
        gdf,
        style_function=lambda x: {'color': map_color, 'weight': 2, 'opacity': 0.7}
    ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=725, height=500)
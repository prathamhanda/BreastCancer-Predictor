import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os


def get_clean_data():
    data_path = os.path.join(os.path.dirname(__file__), "../data/data.csv")
    data = pd.read_csv(data_path)
    drop_columns = ['Unnamed: 32', 'id']
    data = data.drop(columns=[col for col in drop_columns if col in data.columns], errors='ignore')

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def add_sidebar():
  st.sidebar.markdown("<h2 style='color:#f8f9fa;'> Cell Nuclei Measurements</h2>", unsafe_allow_html=True)
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value',
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error',
        
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value',
        
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True,
    template="plotly_dark"
  )
  
  return fig


def add_predictions(input_data):
  model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
  scaler_path = os.path.join(os.path.dirname(__file__), "../model/scaler.pkl")
  if not os.path.exists(model_path):
        st.error("⚠️ Model file is missing! Ensure `model.pkl` is in the `model/` directory.")
        return

  if not os.path.exists(scaler_path):
        st.error("⚠️ Scaler file is missing! Ensure `scaler.pkl` is in the `model/` directory.")
        return

  model = pickle.load(open(model_path, "rb"))
  scaler = pickle.load(open(scaler_path, "rb"))
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.success("✅ The tumor is likely **Benign**.")
  else:
    st.error("⚠️ The tumor is likely **Malignant**.")
  st.write("Probability of being benign: {:.2f}%".format(model.predict_proba(input_array_scaled)[0][0] * 100))
  st.write("Probability of being malignant: {:.2f}%".format(model.predict_proba(input_array_scaled)[0][1] * 100))
#   try:
#     import warnings
#     warnings.filterwarnings('ignore')
# >>>>>>> 9036baf7e5670aca32c40c4234637e243e01dff0
    
#     # Get the absolute path to the model directory
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     model_path = os.path.join(parent_dir, "model", "model.pkl")
#     scaler_path = os.path.join(parent_dir, "model", "scaler.pkl")
    
#     # Load model and scaler with error handling
#     try:
#       with open(model_path, "rb") as model_file:
#         model = pickle.load(model_file)
#       with open(scaler_path, "rb") as scaler_file:
#         scaler = pickle.load(scaler_file)
#     except FileNotFoundError as e:
#       st.error("Error: Model files not found. Please ensure model files are properly uploaded.")
#       return
#     except Exception as e:
#       # Try to handle version incompatibility
#       try:
#         import joblib
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#       except Exception as joblib_error:
#         st.error(f"Error loading model. Please ensure model compatibility: {str(e)}")
#         return
    
#     # Convert input data to float32 for better compatibility
#     input_array = np.array(list(input_data.values()), dtype=np.float32).reshape(1, -1)
    
#     try:
#       input_array_scaled = scaler.transform(input_array)
#     except Exception as scale_error:
#       st.error("Error scaling input data. Please check input values.")
#       return
    
#     try:
#       prediction = model.predict(input_array_scaled)
#       proba = model.predict_proba(input_array_scaled)[0]
#     except Exception as pred_error:
#       st.error("Error making prediction. Please check model compatibility.")
#       return
    
#     st.subheader("Cell cluster prediction")
#     st.markdown("<hr style='border: 1px solid #555;'>", unsafe_allow_html=True)
#     st.write("The cell cluster is:")
    
#     if prediction[0] == 0:
#       st.success("✅ The tumor is likely **Benign**.")
#     else:
#       st.error("⚠️ The tumor is likely **Malignant**.")
      
#     st.write("Probability of being benign: ", f"{proba[0]:.2%}")
#     st.write("Probability of being malicious: ", f"{proba[1]:.2%}")
    
#     st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")
  
# <<<<<<< HEAD
#   st.write("Probability of being benign: {:.2f}%".format(model.predict_proba(input_array_scaled)[0][0] * 100))
#   st.write("Probability of being malignant: {:.2f}%".format(model.predict_proba(input_array_scaled)[0][1] * 100))
  
#   st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

# =======
#   except Exception as e:
#     st.error(f"An unexpected error occurred: {str(e)}")
#     return
def main():
  st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  style_path = os.path.join(os.path.dirname(__file__), "../assets/style.css")
  if os.path.exists(style_path):
        with open(style_path) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("Breast Cancer Predictor")
    st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()
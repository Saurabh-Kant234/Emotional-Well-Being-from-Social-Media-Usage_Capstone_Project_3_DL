import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import keras

st.set_page_config(page_title="Social Media Emotion Predictor (DL)", page_icon="🧬", layout="wide")

st.title("Social Media Usage & Emotional Well-Being - Deep Learning Deployment")
st.write("Predict user emotional well-being using a regularized Artificial Neural Network (ANN).")

@st.cache_resource
def load_dl_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = keras.models.load_model(os.path.join(base_dir, 'best_dl_model.keras'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler_dl.pkl'))
    target_encoder = joblib.load(os.path.join(base_dir, 'target_encoder_dl.pkl'))
    return model, scaler, target_encoder

try:
    model, scaler, target_encoder = load_dl_artifacts()
except Exception as e:
    st.error(f"Error loading DL artifacts. Run dl_code.ipynb first! Details: {e}")
    st.stop()

st.sidebar.header("User Activity Parameters")
age = st.sidebar.slider("Age", min_value=10, max_value=80, value=25)
gender = st.sidebar.selectbox("Gender", ['Female', 'Male', 'Non-binary', 'Other'])
platform = st.sidebar.selectbox("Platform", ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'Snapchat', 'Whatsapp', 'Telegram'])
daily_usage = st.sidebar.number_input("Daily Usage Time (minutes)", min_value=0, max_value=1440, value=120)
posts = st.sidebar.number_input("Posts Per Day", min_value=0, max_value=100, value=3)
likes = st.sidebar.number_input("Likes Received Per Day", min_value=0, max_value=5000, value=45)
comments = st.sidebar.number_input("Comments Received Per Day", min_value=0, max_value=1000, value=10)
messages = st.sidebar.number_input("Messages Sent Per Day", min_value=0, max_value=2000, value=25)

if st.button('Predict Emotional State (DL)'):
    gender_map = {'Female': 0, 'Male': 1, 'Non-binary': 2, 'Other': 3}
    platform_map = {'Instagram': 0, 'Twitter': 1, 'Facebook': 2, 'LinkedIn': 3, 'Snapchat': 4, 'Whatsapp': 5, 'Telegram': 6}
    
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender_map.get(gender, 0),
        'Platform': platform_map.get(platform, 0),
        'Daily_Usage_Time (minutes)': daily_usage,
        'Posts_Per_Day': posts,
        'Likes_Received_Per_Day': likes,
        'Comments_Received_Per_Day': comments,
        'Messages_Sent_Per_Day': messages
    }])
    
    num_cols = ['Age', 'Daily_Usage_Time (minutes)', 'Posts_Per_Day', 
                'Likes_Received_Per_Day', 'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
    
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    
    probabilities = model.predict(input_data, verbose=0)[0]
    pred_encoded = np.argmax(probabilities)
    predicted_emotion = target_encoder.inverse_transform([pred_encoded])[0]
    confidence = float(probabilities[pred_encoded]) * 100
    
    st.subheader("Prediction Result")
    st.success(f"🧬 Neural Network Predicted Emotion: **{predicted_emotion}** (Confidence: {confidence:.2f}%)")
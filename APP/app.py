import streamlit as st
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torchaudio
from torchaudio.transforms import Resample
from tensorflow.keras.models import load_model

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/mms-1b")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/mms-1b").to(device)

saved_model_path = "CNN-MODEL"
try:
    cnn_model = load_model(saved_model_path)
except Exception as e:
    st.error(f"Error loading TensorFlow model: {e}")
    st.stop()

# Preprocessing Function
def preprocess_audio(audio_path):
    try:
        waveform, sampling_rate = torchaudio.load(audio_path)
        desired_sampling_rate = 16000
        if sampling_rate != desired_sampling_rate:
            resampler = Resample(sampling_rate, desired_sampling_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, desired_sampling_rate
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None

# Feature Extraction
def extract_features(audio_path, feature_extractor, wav2vec_model, device):
    waveform, fs = preprocess_audio(audio_path)
    if waveform is None:
        return None
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=fs, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        embeddings = outputs.last_hidden_state.cpu().numpy()
    avg_embeddings = np.mean(embeddings.squeeze(), axis=0)
    return avg_embeddings

# Prediction
def predict_with_cnn(audio_path, cnn_model, feature_extractor, wav2vec_model, device):
    features = extract_features(audio_path, feature_extractor, wav2vec_model, device)
    if features is None:
        return None, None, None
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    predictions = cnn_model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    class_names = ["bonafide", "spoof"]
    confidence = predictions[0][predicted_class[0]]  # Extract confidence for predicted class
    return class_names[predicted_class[0]], predictions[0], confidence

# Streamlit Application
st.set_page_config(page_title="🎵 Audio Spoof Detection", layout="wide")
st.title("🎵 Audio Spoof Detection")
st.markdown(
    """
    This application uses advanced machine learning models to detect whether an audio file is **bonafide** (real) or **spoofed** (fake).
    Upload a `.wav` file to get started!
    """
)

# File Upload
uploaded_file = st.file_uploader(
    "Upload your audio file (WAV format only):", type=["wav"]
)

if uploaded_file:
    # Save uploaded file to a temporary path
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display Audio Player
    st.audio(temp_file_path, format="audio/wav")

    # Processing Audio
    st.write("🎧 **Processing the audio...**")
    predicted_class, probabilities, confidence = predict_with_cnn(
        temp_file_path, cnn_model, feature_extractor, wav2vec_model, device
    )
    
    # Display Results
    if predicted_class:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                ## 🎉 **Prediction: `{predicted_class.upper()}`**
                """
            )
            st.markdown(f"### **Confidence**: `{confidence:.2f}`")

        with col2:
            st.write("### Class Probabilities")
            st.bar_chart(probabilities)

        # Display Detailed Probabilities
        st.markdown("### Class Details")
        st.write(f"**Bonafide Probability**: `{probabilities[0]:.2f}`")
        st.write(f"**Spoof Probability**: `{probabilities[1]:.2f}`")

    else:
        st.error("Failed to process the audio file. Please try again.")
else:
    st.info("Please upload a `.wav` audio file to analyze.")

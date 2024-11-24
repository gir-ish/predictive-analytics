
# 🎭 EmoFake Detection App

This is a Streamlit-based web application that uses advanced machine learning models to detect whether an audio file is **bonafide** (emotionally authentic) or **spoofed** (emotionally fake). The application leverages Facebook's Wav2Vec2 for feature extraction and a TensorFlow-based CNN model trained on PTMs features for classification.

---

## 📋 Features

- Upload `.wav` audio files for classification.
- Provides predictions (`bonafide` or `spoof`) with confidence scores.
- Displays class probabilities with a visually appealing bar chart.
- Easy-to-use interface built with Streamlit.
---
---
## UI

![Video Description](https://github.com/gir-ish/predictive-analytics/blob/main/EMO_FAKE-UI.gif)

---

## 🛠️ Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/gir-ish/Emo-FAKE.git
cd Emo-FAKE
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv env
source env/bin/activate  
```

### **3. Install Dependencies**
Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
Navigate to the `APP` folder and launch the Streamlit app:
```bash
cd APP
streamlit run app.py
```

### **5. If Model Doesn't Work, Download from Here**
If the model isn't available in your project, you can download it from the following link:

[**Download Saved CNN Model**](https://drive.google.com/file/d/1C70M9ooYKmIq00o0DnZpB88Lhg0vomMF/view?usp=sharing)

After downloading:
replace the model folder from new one.

---

---

## 🔗 How to Use

1. **Open the App**:
   After running the above command, Streamlit will provide a local URL (e.g., `http://localhost:----`). Open it in your browser.

2. **Upload an Audio File**:
   - Click on the "Browse files" button or drag and drop a `.wav` file.
   - Ensure the file is in WAV format and meets the required specifications (e.g., 16kHz sampling rate).

3. **View Results**:
   - The app will process the audio and display the following:
     - **Prediction**: Whether the file is "bonafide" or "spoof."
     - **Confidence Score**: How confident the model is in its prediction.
     - **Class Probabilities**: Probabilities for each class (visualized as a bar chart).

---

## 🖥️ Online Application

You can also use the application directly via the following link:

[**Access the EmoFake Detection App on Hugging Face Spaces**](https://huggingface.co/spaces/ggirishg/Emo-Fake-UI)

---

## 🛠️ Troubleshooting

### **Common Issues**

1. **Dependency Errors**:
   - Ensure you have the correct versions of `torch`, `transformers`, `tensorflow`, and `torchaudio`. Use the provided `requirements.txt` to avoid version mismatches.

2. **File Format Errors**:
   - Ensure the uploaded file is in `.wav` format and is not corrupted.

3. **GPU Not Detected**:
   - If using a GPU, ensure CUDA is installed and the `torch` version matches your CUDA version.

4. **Permission Issues (Windows)**:
   - Run Command Prompt as Administrator to resolve file permission issues.

---

## 📂 Project Structure

```plaintext
.
├── APP                  # Folder containing Streamlit application
│   ├── app.py           # Main Streamlit application
├── CNN-MODEL  # Trained TensorFlow CNN model directory
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

---


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<!-- ## 💬 Contact

For questions or feedback, feel free to contact:

- **GitHub**: [Your GitHub Profile](https://github.com/your-username)
 -->

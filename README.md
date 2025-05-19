# 🧠 Parkinson's Detection via Multimodal Analysis

A **Streamlit**-based web application for early detection of **Parkinson’s Disease** using **handwriting**, **voice**, and **typing test** inputs.

---

## 🚀 Features

This app provides a simple interface with three diagnostic tools:

1. **Handwriting Analysis**
   - Upload an image of a spiral you’ve drawn.
   - The system analyzes tremor patterns and deviations typically associated with Parkinson’s.

2. **Voice Analysis**
   - Upload an audio clip of yourself producing a sustained vowel sound (like "aaah").
   - The app extracts and evaluates features such as:
     - MFCCs (Mel-Frequency Cepstral Coefficients)
     - Jitter and shimmer
     - Vocal fold health indicators

3. **Typing Test**
   - Press and hold the **spacebar** as instructed.
   - Measures finger fatigue, motor control, and reaction consistency.

Each option redirects you to a dedicated page for input and real-time analysis.

---

## 🧪 Technical Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io)
- **Data Analysis & Preprocessing**:
  - `Pandas`, `NumPy`
- **Visualization**:
  - `Matplotlib`, `Seaborn`
- **Feature Engineering**:
  - Voice features: MFCCs, jitter, shimmer, vocal fold metrics
  - Typing and motor control metrics
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA)
- **Classification & Tracking**:
  - Statistical modeling to support disease classification and progression tracking

---

## 📊 Key Capabilities

- Upload-based interface for multimodal data input
- Feature extraction for biomedical signal analysis
- Comparative analysis of patient vs. control feature distributions
- Streamlined UI for non-technical users

---

## 📂 Getting Started

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/parkinsons-detector.git
cd parkinsons-detector
```

### 2. Install dependencies
It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the streamlit app
```bash
streamlit run mix.py
```
---

## 📎 Notes

- Make sure your microphone and image files meet the input format requirements.
- The models used are for educational and prototyping purposes. **This is not a medical-grade diagnostic tool.**

---

## 📌 TODOs / Enhancements

- [ ] Model refinement using deep learning (CNNs, RNNs)
- [ ] Live spiral drawing and voice recording via webcam/mic
- [ ] Secure and anonymized data storage
- [ ] Improved UI/UX


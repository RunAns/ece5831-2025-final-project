# Music Genre Classification

This project builds an **end-to-end Music Genre Classification (MGC)** system using **audio signal processing (MFCC features)** and a **Convolutional Neural Network (CNN)** trained on the **GTZAN** dataset. The trained model is deployed via a **Flask web application**, allowing users to upload audio files and receive **Top-3 genre predictions with confidence scores**.

---

## Project Links

- **Final Report (PDF):**  
  https://drive.google.com/file/d/1yc8d9bEVQ3yn1TJ9WzMl-ZrP3ppq6V0f/view?usp=sharing

- **Presentation Slides:**  
  https://docs.google.com/presentation/d/1tDpHmk3zNLnziO-aKrrh-R4PeRKuX8oQ/edit?usp=sharing&ouid=111650610762216042595&rtpof=true&sd=true

- **Dataset (GTZAN used in this project):**  
  https://drive.google.com/file/d/1dskBzo7LxMXuGnIir9f9zhCyadT9hAlM/view?usp=sharing

- **Project Demo Video:**  
  https://youtu.be/j7CRVgZhN4I

- **Pre-recorded Presentation Video:**  
  https://youtu.be/amlSIy4KrkE

---

## Project Overview

### Objective
Automatically classify music tracks into one of **10 genres**:
**Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock**

### Core Techniques
- Audio Signal Processing (MFCC)
- Convolutional Neural Networks (CNN)
- Segment-level prediction aggregation
- Flask-based deployment

---

## Model Pipeline

1. **Audio Input**
   - WAV or MP3 files
   - MP3 files are converted to WAV using FFmpeg

2. **Preprocessing**
   - Resampling to 22,050 Hz
   - Mono conversion
   - Track segmentation (10 segments per 30s track)

3. **Feature Extraction**
   - 13 Mel-Frequency Cepstral Coefficients (MFCCs)
   - FFT window size: 2048
   - Hop length: 512

4. **Deep Learning Model**
   - CNN trained on MFCC feature maps
   - Softmax output for 10 genres

5. **Inference**
   - Predict genre per segment
   - Average probabilities across segments
   - Return Top-3 genres with confidence scores

---

## Folder Structure

```
Music-Genre-Classification/
│
├── app.py                      # Main Flask application
├── app_flask.py                # Alternative Flask version
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Data/
│   └── genres_original/        # GTZAN dataset folders
│
├── templates/                  # HTML templates (Flask/Jinja2)
│   ├── base.html
│   ├── homepage.html
│   ├── prediction.html
│   ├── project.html
│   ├── About.html
│   └── contact.html
│
└── static/
    ├── css/
    ├── js/
    └── img/
```

---

## Installation & Setup

### 1. Create Virtual Environment

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Install FFmpeg (Required for MP3 support)

```
winget install Gyan.FFmpeg
```

Verify installation:
```
ffmpeg -version
```

---

## Running the Project

From the project root directory:
```
python app.py
```

Open in browser:
```
http://127.0.0.1:5000/
```

---

## Evaluation & Results

The final report includes:
- Confusion Matrix
- ROC Curves and AUC Scores
- Segment-wise Prediction Consistency
- Model Calibration Curves

These analyses demonstrate:
- Genre separability
- Common misclassification patterns
- Reliability of confidence scores

---

## Troubleshooting

- **FFmpeg not found** → Ensure FFmpeg is installed and added to PATH
- **Librosa audio errors** → Reinstall dependencies:
```
pip install --upgrade librosa soundfile
```
- **Templates not rendering** → Ensure `templates/` and `static/` folders exist at root level

---

## 👨‍💻 Authors

- **Sai Arunanshu Govindarajula**  
  ✉️ saiarun@umich.edu

- **Tejaswini**  
  ✉️ tejuu@umich.edu
---

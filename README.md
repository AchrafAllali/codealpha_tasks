# <img width="200" height="200" alt="alphacode" src="https://github.com/user-attachments/assets/0d689756-36f1-4c27-902b-6c78dd4bf4ea" />
 CodeAlpha — Machine Learning Internship
### March Batch 2026 · Achraf Allali

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**3 full-stack Machine Learning web applications built during the CodeAlpha ML Internship.**  
Each task is a complete Flask SPA with REST API, interactive visualizations, and live predictions.

[🔗 codealpha.tech](https://www.codealpha.tech)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tasks Summary](#tasks-summary)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Task 1 — Credit Scoring](#task-1--credit-scoring-model)
- [Task 2 — Emotion Recognition](#task-2--emotion-recognition-from-speech)
- [Task 3 — Handwritten Recognition](#task-3--handwritten-character-recognition)
- [Results](#results)
- [Author](#author)

---

## Overview

This repository contains the complete source code for **3 Machine Learning tasks** completed as part of the **CodeAlpha Machine Learning Internship (March Batch 2026)**. Each task has been implemented as a professional, full-stack web application using Flask for the backend and a custom-built Single Page Application (SPA) frontend with interactive charts and real-time ML predictions.

---

## Tasks Summary

| # | Task | Objective | Best Model | Best Accuracy |
|---|------|-----------|------------|---------------|
| 1 | [💳 Credit Scoring](#task-1--credit-scoring-model) | Predict financial creditworthiness | Logistic Regression | **98.99% ROC-AUC** |
| 2 | [🎙️ Emotion Recognition](#task-2--emotion-recognition-from-speech) | Classify 8 emotions from speech (MFCC) | Random Forest | **98.00%** |
| 3 | [✍️ Handwritten Recognition](#task-3--handwritten-character-recognition) | Recognize handwritten digits 0–9 | SVM (RBF) | **98.06%** |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.10+, Flask |
| **Machine Learning** | scikit-learn, numpy, pandas |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript (SPA) |
| **Charts** | Chart.js 4.4 |
| **Fonts** | Google Fonts (Playfair Display, Syne, Space Mono, JetBrains Mono) |
| **Audio Features** | MFCC simulation / librosa (optional) |

---

## Project Structure

```
codealpha_tasks/
│
├── 📁 Task1_Credit_Scoring/
│   ├── app.py                  # Flask backend — 5 REST endpoints
│   └── templates/
│       └── index.html          # Dark fintech SPA
│
├── 📁 Task2_Emotion_Recognition/
│   ├── app.py                  # Flask backend — 6 REST endpoints
│   └── templates/
│       └── index.html          # Dark futuristic SPA
│
├── 📁 Task3_Handwritten_Character_Recognition/
│   ├── app.py                  # Flask backend — 5 REST endpoints
│   └── templates/
│       └── index.html          # Dark terminal SPA + drawing canvas
│
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
Python 3.10+
pip
```

### Install dependencies

```bash
pip install flask scikit-learn pandas numpy
```

> For Task 2 with real RAVDESS audio data (optional):
> ```bash
> pip install librosa soundfile
> ```

### Run any task

```bash
# Task 1
cd Task1_Credit_Scoring
python app.py
# → http://127.0.0.1:5000

# Task 2
cd Task2_Emotion_Recognition
python app.py
# → http://127.0.0.1:5000

# Task 3
cd Task3_Handwritten_Character_Recognition
python app.py
# → http://127.0.0.1:5000
```

> Models are trained automatically at startup. Navigate to `http://127.0.0.1:5000` once the server is running.

---

## Task 1 — Credit Scoring Model

> **Predict the financial creditworthiness of an individual from their financial history.**

### Approach

A synthetic dataset of **5,000 individuals** is generated with realistic financial profiles. Four classification models are trained, compared, and exposed via a REST API.

### Models Used

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 97.20% | **98.99%** ⭐ |
| Gradient Boosting | 96.80% | 98.02% |
| Random Forest | 96.40% | 97.76% |
| Decision Tree | 88.30% | 89.41% |

### Key Features

- **12 raw features** : age, income, employment years, debt amount, credit utilization, savings, payment history, housing status, education, loan amount/duration
- **5 engineered features** : debt-to-income ratio, loan-to-income ratio, monthly repayment, repayment-to-income ratio, financial stability score
- **4 pages** : Dashboard · Data Analysis · Model Performance · Live Individual Prediction

### API Endpoints

```
GET  /api/overview            → KPIs, comparison table, sample data
GET  /api/analysis            → distributions, correlations, feature engineering
GET  /api/performance/<model> → confusion matrix, ROC curves, feature importance
POST /api/predict             → individual client prediction
GET  /api/models              → available models list
```

---

## Task 2 — Emotion Recognition from Speech

> **Recognize human emotions from speech audio using MFCC features.**

### Approach

A dataset of **2,000 audio samples** (250 per emotion) is generated using realistic MFCC profiles per emotion class. Four classifiers are trained on a **268-dimensional feature vector**.

### Emotions Recognized

`neutral` · `calm` · `happy` · `sad` · `angry` · `fearful` · `disgust` · `surprised`

### Models Used

| Model | Accuracy | F1-Score |
|-------|----------|---------|
| Random Forest | **98.00%** ⭐ | **98.00%** |
| Gradient Boosting | 96.25% | 96.24% |
| SVM (RBF) | 95.25% | 95.25% |
| MLP Neural Net | 91.25% | 91.17% |

### Feature Vector (268 dimensions)

```
MFCC mean (40) + MFCC std (40)
+ Delta mean (40) + Delta std (40)
+ Delta² mean (40) + Delta² std (40)
+ Chroma mean (12) + Chroma std (12)
+ ZCR mean + ZCR std + RMS mean + RMS std
```

### Key Features

- RAVDESS real dataset support via `librosa` (optional)
- PCA 2D scatter of 800 feature-space points
- Radar F1 chart across 8 emotion classes
- Per-emotion MFCC profile bar charts
- Live noise-controlled emotion simulation

### API Endpoints

```
GET  /api/overview            → KPIs, emotion distribution, comparison table
GET  /api/performance/<model> → confusion matrix, recall, F1, CV, report
GET  /api/pca                 → PCA 2D coordinates for scatter plot
GET  /api/mfcc_profiles       → MFCC coefficients per emotion
POST /api/predict             → live emotion prediction with noise control
GET  /api/models              → available models list
```

---

## Task 3 — Handwritten Character Recognition

> **Identify handwritten digits (0–9) from pixel features using the sklearn digits dataset.**

### Approach

The **sklearn digits dataset** (1,797 samples, 8×8 pixels, 10 classes) is used. Four classifiers are trained on the 64-pixel feature vector. The app includes a **live drawing canvas** where users can draw a digit and get an instant prediction.

### Models Used

| Model | Accuracy | F1-Score |
|-------|----------|---------|
| SVM (RBF) | **98.06%** ⭐ | **98.05%** |
| MLP Neural Net | 97.22% | 97.19% |
| Logistic Regression | 97.22% | 97.18% |
| Random Forest | 96.11% | 96.10% |

### Key Features

- **Live drawing canvas** (280×280) with adjustable brush + eraser
- Smart **bounding box crop** before 8×8 downscale (matches training distribution)
- **Real-time 8×8 preview** showing exactly what the model receives
- Digit gallery rendering real dataset pixel grids via HTML5 Canvas
- PCA 2D projection of all 1,797 samples colored by class

### Canvas Preprocessing Pipeline

```
User drawing (280×280)
    → Intermediate resize (56×56, high-quality)
    → Bounding box detection
    → Crop + square padding
    → Resize to 8×8
    → Contrast normalization (max pixel = 1.0)
    → Send 64 features to model
```

### API Endpoints

```
GET  /api/overview            → KPIs, digit distribution, gallery data
GET  /api/performance/<model> → confusion matrix 10×10, recall, F1, report
GET  /api/pca                 → PCA 2D for scatter plot
POST /api/predict             → predict from 64 canvas pixel values
GET  /api/models              → available models list
```

---

## Results

| Task | Dataset | Samples | Classes | Best Model | Best Score |
|------|---------|---------|---------|------------|------------|
| Credit Scoring | Synthetic financial data | 5,000 | 2 | Logistic Regression | 98.99% ROC-AUC |
| Emotion Recognition | MFCC simulated (RAVDESS-style) | 2,000 | 8 | Random Forest | 98.00% Accuracy |
| Handwritten Recognition | sklearn digits (MNIST-style) | 1,797 | 10 | SVM (RBF) | 98.06% Accuracy |

---

## Author

**Achraf Allali**  
CodeAlpha Machine Learning Internship — March Batch 2026

🔗 GitHub : [https://github.com/AchrafAllali/codealpha_tasks](https://github.com/AchrafAllali/codealpha_tasks)  
🌐 CodeAlpha : [codealpha.tech](https://www.codealpha.tech)

---

<div align="center">
  <sub>Built with ❤️ during the CodeAlpha ML Internship · March 2026</sub>
</div>

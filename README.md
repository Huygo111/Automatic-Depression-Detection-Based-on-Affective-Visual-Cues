# Automatic-Depression-Detection-Based-on-Affective-Visual-Cues
## 📌 Project Overview

This project implements a complete pipeline for **depression recognition** using **multimodal affective signals**, combining:

* Facial behavior (AUs, head pose, gaze) from **E-DAIC**
* Continuous **Valence–Arousal (VA)** predictions from a deep model pre-trained on **AffectNet**
* **Affective dynamics feature extraction**
* **Statistical analysis (ANOVA)**
* **Binary depression prediction (PHQ-based)**

## 📂 Repository Structure

```text
.
├── segment-edaic-aus-pose-gaze-30s.ipynb
├── train-pred-va-timeseries-affectnet-deep-model.ipynb
├── pred-phq-binary-edaic-anova-affective-dynamics.ipynb
└── README.md
```

---
## Pipeline Overview

```text
E-DAIC Behavioral Signals
        ↓
[1] Multimodal Segmentation (30s)
        ↓
[2] VA Prediction (Deep Model, AffectNet)
        ↓
[3] Affective Dynamics + ANOVA
        ↓
Binary Depression Prediction (PHQ)
```

---

## 📘 Notebook Descriptions

### 1️⃣ `segment-edaic-aus-pose-gaze-30s.ipynb`

**Multimodal segmentation of E-DAIC behavioral data**

**Purpose**

* Segment continuous behavioral signals into fixed **30-second windows**

**Modalities**

* Facial Action Units (AUs)
* Head pose (pitch, yaw, roll)
* Gaze features

**Key Steps**

* Load pre-extracted E-DAIC features
* Temporal alignment across modalities
* Segment signals into non-overlapping 30s windows

**Output**

* Structured multimodal segments per subject/session
* Ready for affective modeling or feature extraction

---

### 2️⃣ `train-pred-va-timeseries-affectnet-deep-model.ipynb`

**Train and apply deep model for Valence–Arousal prediction**

**Purpose**

* Predict continuous **Valence (V)** and **Arousal (A)** time series from behavioral inputs

**Dataset**

* Trained / validated on **AffectNet**
* Applied to E-DAIC segmented data

**Key Steps**

* Deep model training and loading pre-trained weights
* VA time-series prediction for each segment

**Output**

* Continuous VA predictions per time step / segment
* Input for affective dynamics analysis

---

### 3️⃣ `pred-phq-binary-edaic-anova-affective-dynamics.ipynb`

**Affective dynamics feature extraction, ANOVA, and depression prediction**

**Purpose**

* Analyze emotional dynamics and predict **binary depression labels (PHQ-based)**

**Key Steps**

* Extract affective dynamics features from VA time series:

  * Mean, variance
  * Temporal slope
  * Emotional inertia
  * Variability and instability
* Perform **ANOVA** to identify statistically significant features
* Train classical ML models for binary depression prediction

**Models**

* Logistic Regression
* Random Forest
* Gradient Boosting (e.g., XGBoost / LightGBM)

**Output**

* Significant affective features
* Binary depression prediction results
* Model performance metrics

---

# 🎯 Unified XAI Interface

**Multi-Modal Classification with Explainable AI**

A comprehensive web application integrating deepfake audio detection and lung cancer detection with explainable AI techniques (LIME, Grad-CAM, and SHAP). Built with Flask, PyTorch, TensorFlow and modern web technologies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Team](#-team)
- [Technologies Used](#-technologies-used)
- [Configuration](#-configuration)

---

## 🎯 Overview

This project implements a unified interface for two distinct classification tasks:

1. **Audio Classification**: Deepfake detection (Real vs Fake speech)

- **Models:** VGG16, MobileNet, ResNet
- **Dataset:** Fake-or-Real (FoR) Dataset
- **Input:** `.wav`, `.mp3` audio files

2. **Image Classification**: Chest X-ray analysis (Normal vs Malignant)

- **Models:** ConvNeXt, DenseNet
- **Dataset:** CheXpert chest X-rays
- **Input:** `.png`, `.jpg` image files

Both models are enhanced with  XAI techniques (**LIME**, **Grad-CAM** and **SHAP**) to provide interpretable insights into their predictions.

---

## ✨ Features

### 🎨 Web Interface (Flask)
- **Beautiful Modern UI** - Gradient design with smooth animations
- **Drag & Drop Upload** - Intuitive file upload
- **Real-time Processing** - Live progress updates
- **Responsive Design** - Works on all devices
- **Toast Notifications** - Clear user feedback

### 🧠 AI Predictions
- **Model Selection** - Easy switching between models:

   - **Audio Classification**:
      - **VGG16** - Classic CNN architecture
      - **MobileNet** - Lightweight, fast model
      - **ResNet** - Deep residual learning, strong feature extractor

   - **Image Classification**:
      - **ConvNeXt** - Modernized CNN
      - **DenseNet** - Dense connectivity, efficient feature reuse

- **Confidence Scores** - Probability distribution visualization

### 🔍 Explainable AI
- **LIME, Grad-CAM and SHAP Integration** - Visual feature importance
- **Automatic Filtering** - XAI method compatibility checking
- **Clear Visualizations** - Heatmaps and overlays
- **Interpretable Results** - Understand model decisions

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/lisacharuel/XAI_Final_Project.git
cd XAI_Final_Project
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

Download the pre-trained models [here](https://drive.google.com/file/d/15iRpPcClbGSxuxQ9ZDssZvuxWUFpniQ1/view?usp=sharing).
Unzip the `models.zip` archive and place the `models/` folder in the project root directory, replacing the existing one.

---

## 🎮 Usage

### Launch Flask Application

```bash
python app_flask.py
```

The application will start on **http://localhost:5000**

### Using the Interface

1. **Upload File**
   - Drag & drop or click to browse
   - Supports: `.wav`, `.mp3`, `.jpg`, `.png`
   - Max size: 10MB

2. **Select Model**
   - Click on the appropriate model button
   - MobileNet, Resnet50 or VGG16 for audio
   - ConvNeXt or DenseNet for images

3. **View Prediction**
   - See classification result
   - View confidence score
   - Examine probability distribution

4. **Generate Explanation**
   - Click "Explain with LIME/Grad-CAM/SHAP"
   - Wait 30-60 seconds for processing
   - View visual explanation

5. **Reset** (Optional)
   - Click reset button to start over

---

## 📁 Project Structure

```
XAI_Final_Project/
├── app_flask.py              # Flask web application (MAIN)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── models/                   # Neural network models
│   ├── audio/
│   │   ├── mobilenet.h5
│   │   ├── mobilenet.ipynb
│   │   ├── resnet50.h5
│   │   ├── resnet50.ipynb
│   │   ├── vgg16.h5
│   │   └── vgg16.ipynb
│   ├── image/
│   │   ├── convnext.json
│   │   ├── convnext.safetensors
│   │   ├── densenet.json
│   │   └── densenet.safetensors
│   └── model_loader.py
│
├── preprocessing/            # Data preprocessing
│   ├── audio_processor.py
│   └── image_processor.py
│
├── xai/                      # Explainable AI
│   ├── gradcam_explainer.py
│   ├── lime_explainer.py
│   └── shap_explainer.py
│
├── utils/                    # Utility functions
│   ├── file_handler.py
│   └── compatibility_checker.py
│
├── templates/                # Flask HTML templates
│   └── index.html
│
├── static/                   # CSS and JavaScript
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
│
├── docs/                     # Documentation
│   └── GENERATIVE_AI_USAGE.md
│
└── outputs/                  # Generated visualizations
    └── visualizations/
```

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[GENERATIVE_AI_USAGE.md](docs/GENERATIVE_AI_USAGE.md)** - AI tools disclosure

---

## 👥 Team

**Project:** Unified XAI Interface  
**Course:** Explainability AI 

**Team Members:**
- Lisa CHARUEL
- Aymeric MARTIN
- Julien DE VOS

---

## 🛠️ Technologies Used

### Backend
- **Python 3.8+** - Programming language
- **PyTorch 2.0+** - Deep learning framework
- **TensorFlow/Keras** - Model building
- **Flask 3.0+** - Web framework
- **LIME** - Explainable AI library
- **Grad-CAM** - Visual explanations
- **SHAP** - Model interpretability

### Audio Processing
- **librosa** - Audio analysis
- **soundfile** - Audio I/O
- **scipy** - Signal processing

### Image Processing
- **keras-preprocessing** - Image preprocessing
- **PIL** - Image handling
- **scikit-image** - Image segmentation

### Frontend
- **HTML5/CSS3** - Modern web standards
- **JavaScript (Vanilla)** - Interactive UI
- **Inter Font** - Typography

---

## ⚙️ Configuration

### Change Port

Edit `app_flask.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

### Adjust File Size Limit

Edit `app_flask.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB
```
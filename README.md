# 🎯 Unified XAI Interface

**Multi-Modal Classification with Explainable AI**

A comprehensive web application integrating deepfake audio detection and lung cancer detection with explainable AI techniques (LIME). Built with Flask, PyTorch, and modern web technologies.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models & XAI](#models--xai)
- [Documentation](#documentation)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements a unified interface for two distinct classification tasks:

1. **Audio Classification**: Deepfake detection (Real vs Fake speech)

- **Models:** VGG16, MobileNet, ResNet, Custom CNN
- **Dataset:** Fake-or-Real (FoR) Dataset
- **Input:** `.wav` audio files
- **XAI Methods:** LIME, SHAP

2. **Image Classification**: Chest X-ray analysis (Normal vs Malignant)

- **Models:** AlexNet, DenseNet
- **Dataset:** CheXpert chest X-rays
- **Input:** `.png`, `.jpg` image files
- **XAI Methods:** Grad-CAM, LIME, SHAP

Both models are enhanced with **LIME** (Local Interpretable Model-agnostic Explanations) to provide visual explanations of predictions, making the AI decision-making process transparent and interpretable.

### Key Objectives

- ✅ Unified interface for multi-modal inputs
- ✅ Explainable AI integration
- ✅ Automatic compatibility filtering
- ✅ Professional web interface
- ✅ Modular, extensible architecture

---

## ✨ Features

### 🎨 Web Interface (Flask)
- **Beautiful Modern UI** - Gradient design with smooth animations
- **Drag & Drop Upload** - Intuitive file upload
- **Real-time Processing** - Live progress updates
- **Responsive Design** - Works on all devices
- **Toast Notifications** - Clear user feedback

### 🧠 Machine Learning
- **CustomCNN** - Lightweight audio classification model
- **AlexNet** - Transfer learning for medical imaging
- **Dual Modality** - Handles both audio and images
- **CPU Support** - No GPU required (Colab optional)

### 🔍 Explainable AI
- **LIME Integration** - Visual feature importance
- **Automatic Filtering** - XAI method compatibility checking
- **Clear Visualizations** - Heatmaps and overlays
- **Interpretable Results** - Understand model decisions

### 📊 Additional Features
- Session management
- File validation
- Automatic preprocessing
- Confidence scores
- Probability visualization

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

### Step 4: Add Sample Data

Download the sample data and place files in:
```
data/
├── sample_audio/
│   ├── real_speech_1.wav
│   ├── real_speech_2.wav
│   ├── fake_speech_1.wav
│   └── fake_speech_2.wav
└── sample_images/
    ├── normal_xray_1.png
    ├── normal_xray_2.png
    ├── malignant_xray_1.png
    └── malignant_xray_2.png
```

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
   - CustomCNN for audio
   - AlexNet for images

3. **View Prediction**
   - See classification result
   - View confidence score
   - Examine probability distribution

4. **Generate Explanation**
   - Click "Explain with LIME"
   - Wait 30-60 seconds for processing
   - View visual explanation

5. **Reset** (Optional)
   - Click reset button to start over

### Alternative: Chainlit Interface

A chat-based interface is also available:

```bash
chainlit run app.py -w --port 8080
```

Access at **http://localhost:8080**

---

## 📁 Project Structure

```
XAI_Final_Project/
├── app_flask.py              # Flask web application (MAIN)
├── app.py                    # Chainlit interface (alternative)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── models/                   # Neural network models
│   ├── audio/
│   │   └── custom_cnn_audio.py
│   ├── image/
│   │   └── alexnet_image.py
│   └── model_loader.py
│
├── preprocessing/            # Data preprocessing
│   ├── audio_processor.py
│   └── image_processor.py
│
├── xai/                      # Explainable AI
│   └── lime_explainer.py
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
├── data/                     # Sample data (not in git)
│   ├── sample_audio/
│   └── sample_images/
│
├── docs/                     # Documentation
│   ├── TECHNICAL_REPORT.md
│   └── GENERATIVE_AI_USAGE.md
│
└── outputs/                  # Generated visualizations
    └── visualizations/
```

---

## 🧠 Models & XAI

### Audio Model: CustomCNN

**Architecture:**
- 3 Convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool)
- Global Average Pooling
- 2 Fully Connected layers
- Dropout regularization

**Input:** Mel-spectrogram (3, 128, 128)  
**Output:** Binary classification (Real/Fake)  
**Parameters:** ~200K

### Image Model: AlexNet

**Architecture:**
- 5 Convolutional layers
- 3 Fully Connected layers
- Transfer learning from ImageNet
- Modified final layer for binary classification

**Input:** RGB image (3, 224, 224)  
**Output:** Binary classification (Normal/Malignant)  
**Parameters:** ~60M

### XAI Method: LIME

**Local Interpretable Model-agnostic Explanations**

- Generates 1000 perturbed samples
- Fits local linear model
- Identifies important features
- Creates visual heatmaps
- Works for both audio and images

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)** - Detailed technical documentation
- **[GENERATIVE_AI_USAGE.md](docs/GENERATIVE_AI_USAGE.md)** - AI tools disclosure
- **[DEMO_GUIDE.md](docs/DEMO_GUIDE.md)** - Presentation guide

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
- **Flask 3.0+** - Web framework
- **LIME** - Explainable AI library

### Audio Processing
- **librosa** - Audio analysis
- **soundfile** - Audio I/O
- **scipy** - Signal processing

### Image Processing
- **torchvision** - Image preprocessing
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

### Modify LIME Parameters

Edit `xai/lime_explainer.py`:
```python
self.num_samples = 1000  # Number of samples
self.num_features = 10   # Top features to show
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use different port
# Edit app_flask.py or:
python app_flask.py  # Then change port in file
```

### Module Not Found
```bash
pip install -r requirements.txt
```

### LIME Takes Too Long
Reduce samples in `xai/lime_explainer.py`:
```python
self.num_samples = 500  # Faster but less accurate
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Anthropic** - Claude AI and Gemini for development assistance
- **PyTorch Team** - Deep learning framework
- **LIME Authors** - Explainable AI methodology
- **Flask Community** - Web framework
- **Course Instructors** - Project guidance

---


## 🎯 Project Status

- ✅ Phase 1: Project Foundation
- ✅ Phase 2: Model Implementation
- ✅ Phase 3: XAI Integration
- ✅ Phase 4: Web Interface (Flask)
- ✅ Phase 5: Documentation & Testing


---


# Unified Explainable AI Interface

**Multi-Modal Classification System with Explainable AI**

This project integrates deepfake audio detection and lung cancer detection into a single unified interface, providing explainable AI visualizations for both modalities.

---

## 👥 Team Information

**TD Group:** CDOF1

**Team Members:**
- Lisa Charuel
- Aymeric Martin
- Julien De Vos

---

## 🎯 Project Overview

This unified platform combines two XAI systems:

### 1. **Deepfake Audio Detection**
- **Models:** VGG16, MobileNet, ResNet, Custom CNN
- **Dataset:** Fake-or-Real (FoR) Dataset
- **Input:** `.wav` audio files
- **XAI Methods:** LIME, SHAP

### 2. **Lung Cancer Detection**
- **Models:** AlexNet, DenseNet
- **Dataset:** CheXpert chest X-rays
- **Input:** `.png`, `.jpg` image files
- **XAI Methods:** Grad-CAM, LIME, SHAP

### Key Features
✅ Multi-modal input support (audio + images)  
✅ Multiple pre-trained classification models  
✅ Automatic XAI method filtering based on input type  
✅ Side-by-side comparison of explainability techniques  
✅ Interactive web interface built with Chainlit  

---

## 🏗️ Project Structure

```
unified-xai-interface/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── app.py                            # Main Chainlit application
├── config.py                         # Configuration settings
│
├── models/                           # Model architectures & weights
│   ├── audio/
│   │   ├── vgg16_audio.py
│   │   ├── mobilenet_audio.py
│   │   ├── resnet_audio.py
│   │   └── custom_cnn_audio.py
│   ├── image/
│   │   ├── alexnet_image.py
│   │   └── densenet_image.py
│   └── model_loader.py               # Unified model loading
│
├── xai/                              # Explainability implementations
│   ├── lime_explainer.py
│   ├── shap_explainer.py
│   ├── gradcam_explainer.py
│   └── xai_manager.py                # XAI method router
│
├── preprocessing/                     # Data preprocessing
│   ├── audio_processor.py
│   └── image_processor.py
│
├── utils/                            # Utility functions
│   ├── visualizations.py
│   ├── file_handler.py
│   └── compatibility_checker.py
│
├── assets/                           # Static files
│   └── styles.css
│
├── data/                             # Sample data for testing
│   ├── sample_audio/
│   └── sample_images/
│
├── weights/                          # Pre-trained model weights
│   ├── audio_models/
│   └── image_models/
│
├── notebooks/                        # Jupyter notebooks for Colab
│   ├── train_audio_models.ipynb
│   └── train_image_models.ipynb
│
└── docs/                            # Documentation
    ├── TECHNICAL_REPORT.md
    ├── GENERATIVE_AI_USAGE.md
    └── DEMO_GUIDE.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU for faster inference - can use Google Colab

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

### Step 4: Download Pre-trained Weights
```bash
# Run the setup script to download model weights
python setup_models.py
```

### Step 5: Prepare Sample Data
Place your test files in:
- Audio files: `data/sample_audio/`
- Image files: `data/sample_images/`

---

## 🎮 Running the Application

### Local Deployment (Chainlit)
```bash
chainlit run app.py -w
```

Then open your browser to: `http://localhost:8000`

### Google Colab Deployment
If you want to run on Colab with GPU:
1. Open `notebooks/deploy_colab.ipynb`
2. Run all cells
3. Use the generated ngrok URL

---

## 📖 User Guide

### Basic Workflow

1. **Upload Data**
   - Click "Upload File" button
   - Select audio (`.wav`) or image (`.jpg`, `.png`)

2. **Select Model**
   - Choose from available models for your data type
   - Audio: VGG16, MobileNet, ResNet, Custom CNN
   - Image: AlexNet, DenseNet

3. **Choose XAI Method**
   - Methods automatically filter based on input type
   - Audio compatible: LIME, SHAP
   - Image compatible: LIME, SHAP, Grad-CAM

4. **View Results**
   - See classification prediction
   - Explore explainability visualizations

5. **Compare (Optional)**
   - Navigate to "Comparison" tab
   - Select multiple XAI methods
   - View side-by-side analysis

---

## 🔬 XAI Methods Explained

### LIME (Local Interpretable Model-agnostic Explanations)
- **Works on:** Audio & Images
- **Explanation:** Shows which regions/features influence the prediction by perturbing input and observing changes

### SHAP (SHapley Additive exPlanations)
- **Works on:** Audio & Images
- **Explanation:** Assigns importance values to each feature based on game theory principles

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Works on:** Images only
- **Explanation:** Highlights regions in the image that are important for the prediction using gradient information

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Test with Sample Data
Sample audio and images are provided in the `data/` folder for quick testing.

---

## 📊 Technical Details

### Models

**Audio Classification:**
- Input: Mel-spectrogram (128x128) from audio waveform
- Preprocessing: Librosa for audio feature extraction
- Models trained on FoR dataset (real vs. fake audio)

**Image Classification:**
- Input: Chest X-ray images (224x224)
- Preprocessing: Standard ImageNet normalization
- Models trained on CheXpert dataset (normal vs. malignant)

### Automatic Compatibility Filtering

The system automatically determines which XAI methods are compatible with the uploaded file:

```python
# Example logic
if input_type == "audio":
    available_xai = ["LIME", "SHAP"]
elif input_type == "image":
    available_xai = ["LIME", "SHAP", "Grad-CAM"]
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- Multi-modal machine learning integration
- Explainable AI implementation across different data types
- Building user-friendly interfaces for ML models
- Model comparison and evaluation
- Software engineering best practices for ML projects

---

## 🤖 Generative AI Usage Statement

### Declaration
This project utilized Generative AI tools during development.

**Tools Used:**
- Claude (Anthropic) - AI assistant

**Purposes:**
- Code architecture design and refactoring
- Documentation writing and README structure
- Debugging assistance
- XAI implementation guidance
- Best practices recommendations

**Human Contributions:**
- Model training and evaluation
- Dataset preparation and analysis
- Interface design decisions
- Testing and validation
- Final code review and modifications

All AI-generated code was reviewed, tested, and modified by team members to ensure correctness and project requirements compliance.

---

## 📝 License

This project is for educational purposes as part of the XAI course curriculum.

---

## 🐛 Known Issues & Future Improvements

### Current Limitations
- Models require significant memory for inference
- Large audio files may take time to process
- Limited to binary classification tasks

### Planned Enhancements
- [ ] Add support for CSV data
- [ ] Implement additional XAI techniques (Integrated Gradients, XRAI)
- [ ] Add interactive zoom/pan features for visualizations
- [ ] Support batch processing
- [ ] Add model performance metrics display

---



## 🙏 Acknowledgments

- Original Deepfake Audio Detection repository by Guri10
- Lung Cancer Detection implementation by schaudhuri16
- FoR Dataset creators
- CheXpert dataset (Stanford ML Group)

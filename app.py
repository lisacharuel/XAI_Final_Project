"""
Unified XAI Interface - Chainlit Application
Multi-modal classification with explainable AI

Features:
- File upload (audio and images)
- Auto-detection of input type
- Model selection (filtered by input type)
- XAI method selection (auto-filtered)
- Real-time predictions
- LIME explanations
- Comparison mode
"""

import chainlit as cl
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io

# Import our modules
from preprocessing.audio_processor import audio_processor
from preprocessing.image_processor import image_processor
from models.model_loader import model_loader, quick_predict
from xai.lime_explainer import explain_with_lime
from utils.file_handler import file_handler
from utils.compatibility_checker import compatibility_checker
from config import AUDIO_CONFIG, IMAGE_CONFIG, DEVICE


# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class SessionState:
    """Manages state for each user session"""
    def __init__(self):
        self.uploaded_file = None
        self.file_type = None
        self.processed_tensor = None
        self.original_data = None
        self.selected_model = None
        self.loaded_model = None
        self.prediction_result = None
        self.selected_xai = None

# Store session states
session_states = {}


def get_session_state(session_id: str) -> SessionState:
    """Get or create session state"""
    if session_id not in session_states:
        session_states[session_id] = SessionState()
    return session_states[session_id]


# ============================================================================
# STARTUP
# ============================================================================

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    # Welcome message
    welcome_message = """
# 🎉 Welcome to Unified XAI Interface!

**Multi-Modal Classification with Explainable AI**

This interface supports:
- 🎵 **Audio**: Deepfake detection (Real vs Fake speech)
- 🏥 **Images**: Chest X-ray classification (Normal vs Malignant)

## 🚀 How to Use:

1. **Upload a file** using the button below
2. **Select a model** (automatically filtered for your file type)
3. **Choose XAI method** (LIME is recommended)
4. **View results** with explanations!

---

**Supported Formats:**
- Audio: `.wav` files
- Images: `.jpg`, `.png` files

**Available Models:**
- CustomCNN (Audio)
- AlexNet (Images)

**XAI Methods:**
- LIME (Both audio & images)
- SHAP (Coming soon)
- Grad-CAM (Images only - Coming soon)

---

**Current Device:** `{device}`

Ready to start? Upload a file! 📁
    """.format(device=DEVICE)
    
    await cl.Message(content=welcome_message).send()
    
    # Initialize session state
    session_id = cl.user_session.get("id")
    state = get_session_state(session_id)
    cl.user_session.set("state", state)


# ============================================================================
# FILE UPLOAD HANDLING
# ============================================================================

@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages and file uploads"""
    
    state = cl.user_session.get("state")
    
    # Check if files were uploaded
    if message.elements:
        await handle_file_upload(message.elements[0], state)
    else:
        # Handle text commands
        text = message.content.lower()
        
        if "help" in text:
            await show_help()
        elif "reset" in text or "clear" in text:
            await reset_session(state)
        else:
            await cl.Message(
                content="Please upload a file to get started! Use /help for instructions."
            ).send()


async def handle_file_upload(file_element, state):
    """Process uploaded file"""
    
    # Show processing message
    msg = cl.Message(content="🔄 Processing your file...")
    await msg.send()
    
    try:
        # Get file content
        file_content = file_element.content
        file_name = file_element.name
        file_path = file_element.path
        
        # Validate file
        temp_path = Path(file_path)
        is_valid, file_type, error_msg = file_handler.validate_file(temp_path)
        
        if not is_valid:
            await msg.update(content=f"❌ Error: {error_msg}")
            return
        
        # Store in state
        state.uploaded_file = temp_path
        state.file_type = file_type
        
        # Process the file
        if file_type == "audio":
            await process_audio_file(state, file_name, msg)
        elif file_type == "image":
            await process_image_file(state, file_name, msg)
        
    except Exception as e:
        await msg.update(content=f"❌ Error processing file: {str(e)}")
        import traceback
        print(traceback.format_exc())


async def process_audio_file(state, file_name, msg):
    """Process uploaded audio file"""
    
    await msg.update(content=f"🎵 Processing audio file: **{file_name}**...")
    
    try:
        # Preprocess
        tensor, spectrogram = audio_processor.preprocess(state.uploaded_file)
        
        state.processed_tensor = tensor
        state.original_data = spectrogram
        
        # Show file info
        info = audio_processor.get_audio_info(state.uploaded_file)
        
        info_text = f"""
✅ **Audio file processed successfully!**

📊 **File Information:**
- **Filename:** {file_name}
- **Duration:** {info.get('duration_seconds', 'N/A')}s
- **Sample Rate:** {info.get('sample_rate', 'N/A')} Hz
- **Processed Shape:** {tensor.shape}

🎯 **Next Steps:**
1. Select a model (use the commands below)
2. Get prediction and explanation

**Available Models for Audio:**
"""
        
        # Get compatible models
        models = compatibility_checker.get_compatible_models("audio")
        for i, model_info in enumerate(models, 1):
            info_text += f"\n{i}. **{model_info['name']}** - {model_info['description']}"
        
        info_text += "\n\n💡 Type: `/model custom_cnn` to select CustomCNN"
        
        await msg.update(content=info_text)
        
        # Create action buttons
        actions = [
            cl.Action(name="model_custom_cnn", value="custom_cnn", label="🎵 Use CustomCNN"),
            cl.Action(name="show_spectrogram", value="show_spec", label="👁️ View Spectrogram"),
        ]
        
        await cl.Message(
            content="**Quick Actions:**",
            actions=actions
        ).send()
        
    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")


async def process_image_file(state, file_name, msg):
    """Process uploaded image file"""
    
    await msg.update(content=f"🏥 Processing image file: **{file_name}**...")
    
    try:
        # Preprocess
        normalized, unnormalized, original = image_processor.preprocess(state.uploaded_file)
        
        state.processed_tensor = normalized
        state.original_data = np.array(original)
        
        # Show file info
        info = image_processor.get_image_info(state.uploaded_file)
        
        info_text = f"""
✅ **Image file processed successfully!**

📊 **File Information:**
- **Filename:** {file_name}
- **Original Size:** {info.get('width', 'N/A')}x{info.get('height', 'N/A')}
- **Resized To:** {info.get('resized_to', 'N/A')}
- **Processed Shape:** {normalized.shape}

🎯 **Next Steps:**
1. Select a model
2. Get prediction and explanation

**Available Models for Images:**
"""
        
        # Get compatible models
        models = compatibility_checker.get_compatible_models("image")
        for i, model_info in enumerate(models, 1):
            info_text += f"\n{i}. **{model_info['name']}** - {model_info['description']}"
        
        info_text += "\n\n💡 Type: `/model alexnet` to select AlexNet"
        
        await msg.update(content=info_text)
        
        # Show the uploaded image
        image_element = cl.Image(
            name="uploaded_image",
            path=str(state.uploaded_file),
            display="inline",
            size="medium"
        )
        
        await cl.Message(
            content="**Your uploaded image:**",
            elements=[image_element]
        ).send()
        
        # Create action buttons
        actions = [
            cl.Action(name="model_alexnet", value="alexnet", label="🏥 Use AlexNet"),
        ]
        
        await cl.Message(
            content="**Quick Actions:**",
            actions=actions
        ).send()
        
    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")


# ============================================================================
# ACTION HANDLERS
# ============================================================================

@cl.action_callback("model_custom_cnn")
async def on_model_custom_cnn(action: cl.Action):
    """Handle CustomCNN selection"""
    state = cl.user_session.get("state")
    await select_model(state, "custom_cnn")


@cl.action_callback("model_alexnet")
async def on_model_alexnet(action: cl.Action):
    """Handle AlexNet selection"""
    state = cl.user_session.get("state")
    await select_model(state, "alexnet")


@cl.action_callback("show_spectrogram")
async def on_show_spectrogram(action: cl.Action):
    """Show audio spectrogram"""
    state = cl.user_session.get("state")
    
    if state.original_data is None:
        await cl.Message(content="❌ No audio data available").send()
        return
    
    # Create visualization
    import matplotlib.pyplot as plt
    fig = audio_processor.visualize_spectrogram(state.original_data, title="Mel-Spectrogram")
    
    # Save to temp file
    temp_path = Path("outputs/visualizations/temp_spectrogram.png")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(temp_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Send image
    image_element = cl.Image(
        name="spectrogram",
        path=str(temp_path),
        display="inline",
        size="large"
    )
    
    await cl.Message(
        content="**Audio Spectrogram:**",
        elements=[image_element]
    ).send()


async def select_model(state, model_key: str):
    """Load and use selected model"""
    
    msg = cl.Message(content=f"🔄 Loading model...")
    await msg.send()
    
    try:
        # Check if file is uploaded
        if state.processed_tensor is None:
            await msg.update(content="❌ Please upload a file first!")
            return
        
        # Load model
        if state.file_type == "audio":
            model = model_loader.load_audio_model(model_key, use_pretrained=False)
            class_names = AUDIO_CONFIG["classes"]
        else:
            model = model_loader.load_image_model(model_key, use_pretrained=False)
            class_names = IMAGE_CONFIG["classes"]
        
        state.loaded_model = model
        state.selected_model = model_key
        
        await msg.update(content="✅ Model loaded! Making prediction...")
        
        # Make prediction
        result = quick_predict(model, state.processed_tensor, state.file_type)
        state.prediction_result = result
        
        # Show results
        result_text = f"""
## 🎯 Prediction Results

**Model:** {model_key.upper()}

**Prediction:** `{result['predicted_class']}`  
**Confidence:** `{result['confidence']:.1%}`

**All Probabilities:**
"""
        
        for class_name, prob in result['all_probabilities'].items():
            bar = "█" * int(prob * 20)
            result_text += f"\n- {class_name}: `{prob:.1%}` {bar}"
        
        result_text += "\n\n💡 **Want an explanation?** Type `/explain lime` to see which features influenced this prediction!"
        
        await msg.update(content=result_text)
        
        # Add XAI action button
        actions = [
            cl.Action(name="explain_lime", value="lime", label="🔍 Explain with LIME"),
        ]
        
        await cl.Message(
            content="**Get Explanation:**",
            actions=actions
        ).send()
        
    except Exception as e:
        await msg.update(content=f"❌ Error: {str(e)}")
        import traceback
        print(traceback.format_exc())


@cl.action_callback("explain_lime")
async def on_explain_lime(action: cl.Action):
    """Generate LIME explanation"""
    state = cl.user_session.get("state")
    await generate_explanation(state, "lime")


async def generate_explanation(state, xai_method: str):
    """Generate XAI explanation"""
    
    msg = cl.Message(content="🔍 Generating explanation...")
    await msg.send()
    
    try:
        # Check prerequisites
        if state.loaded_model is None:
            await msg.update(content="❌ Please select a model first!")
            return
        
        if state.prediction_result is None:
            await msg.update(content="❌ Please make a prediction first!")
            return
        
        # Check compatibility
        if not compatibility_checker.is_xai_compatible(xai_method, state.file_type):
            await msg.update(
                content=f"❌ {xai_method.upper()} is not compatible with {state.file_type} files!"
            )
            return
        
        await msg.update(content=f"🔍 Generating {xai_method.upper()} explanation... (this may take 30-60 seconds)")
        
        # Generate explanation
        if xai_method == "lime":
            class_names = AUDIO_CONFIG["classes"] if state.file_type == "audio" else IMAGE_CONFIG["classes"]
            
            save_path = Path(f"outputs/visualizations/{state.file_type}_lime_{state.selected_model}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            explanation_img, scores, fig = explain_with_lime(
                model=state.loaded_model,
                input_tensor=state.processed_tensor,
                original_data=state.original_data,
                input_type=state.file_type,
                class_names=class_names,
                prediction_result=state.prediction_result,
                save_path=save_path
            )
            
            plt.close(fig)
            
            # Send explanation
            image_element = cl.Image(
                name="lime_explanation",
                path=str(save_path),
                display="inline",
                size="large"
            )
            
            explanation_text = f"""
## 🔍 LIME Explanation

**Method:** Local Interpretable Model-agnostic Explanations

**What LIME shows:**
- Which regions of the input were most important
- Positive features (green) support the prediction
- Negative features (red) argue against it

**Top Features Analyzed:** {len(scores)}

**Interpretation:**
LIME perturbed {1000} versions of your input and observed how predictions changed. 
The highlighted regions had the strongest influence on the model's decision.

---

**Predicted:** `{state.prediction_result['predicted_class']}`  
**Confidence:** `{state.prediction_result['confidence']:.1%}`
"""
            
            await msg.update(content=explanation_text, elements=[image_element])
            
        else:
            await msg.update(content=f"❌ {xai_method.upper()} not yet implemented!")
        
    except Exception as e:
        await msg.update(content=f"❌ Error generating explanation: {str(e)}")
        import traceback
        print(traceback.format_exc())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def show_help():
    """Show help message"""
    help_text = """
# 📚 Help & Commands

## Basic Workflow:
1. Upload a file (audio or image)
2. Select a model
3. Get prediction
4. Request explanation

## Available Commands:
- `/help` - Show this help
- `/reset` - Clear session and start over
- `/model <name>` - Select model (custom_cnn, alexnet)
- `/explain <method>` - Generate explanation (lime)

## File Support:
- **Audio:** .wav files (3 seconds, 16kHz recommended)
- **Images:** .jpg, .png files (will be resized to 224x224)

## Models:
- **CustomCNN** - For audio deepfake detection
- **AlexNet** - For chest X-ray classification

## XAI Methods:
- **LIME** - Available for both audio and images
- **SHAP** - Coming soon
- **Grad-CAM** - Coming soon (images only)

## Tips:
- Use the action buttons for quick access
- LIME explanations take 30-60 seconds to generate
- Models use random weights (train for better results)

Need more help? Check the README.md file!
"""
    await cl.Message(content=help_text).send()


async def reset_session(state):
    """Reset the current session"""
    state.uploaded_file = None
    state.file_type = None
    state.processed_tensor = None
    state.original_data = None
    state.selected_model = None
    state.loaded_model = None
    state.prediction_result = None
    state.selected_xai = None
    
    await cl.Message(content="✅ Session reset! Upload a new file to start.").send()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # This is handled by chainlit CLI
    pass
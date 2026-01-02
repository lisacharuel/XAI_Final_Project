"""
Unified XAI Interface - Flask Application
Multi-modal classification with explainable AI

Features:
- Beautiful web interface
- File upload (audio and images)
- Model selection
- Real-time predictions
- LIME explanations
- Comparison view
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io
import uuid
import json

# Import our modules
from preprocessing.audio_processor import audio_processor
from preprocessing.image_processor import image_processor
from models.model_loader import model_loader, quick_predict
from xai.lime_explainer import explain_with_lime
from utils.file_handler import file_handler
from utils.compatibility_checker import compatibility_checker
from config import AUDIO_CONFIG, IMAGE_CONFIG, DEVICE

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
app.secret_key = 'unified-xai-interface-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = Path('temp_uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Store session data in memory (for demo purposes)
# In production, use Redis or database
sessions_data = {}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def get_session_data():
    """Get session data"""
    session_id = get_session_id()
    if session_id not in sessions_data:
        sessions_data[session_id] = {
            'uploaded_file': None,
            'file_type': None,
            'processed_tensor': None,
            'original_data': None,
            'selected_model': None,
            'loaded_model': None,
            'prediction_result': None
        }
    return sessions_data[session_id]


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', device=DEVICE)


@app.route('/api/info')
def api_info():
    """Get system information"""
    return jsonify({
        'device': str(DEVICE),
        'audio_models': [m['name'] for m in compatibility_checker.get_compatible_models('audio')],
        'image_models': [m['name'] for m in compatibility_checker.get_compatible_models('image')],
        'xai_methods': list(compatibility_checker.xai_methods.keys())
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(filepath)
        
        # Validate file
        is_valid, file_type, error_msg = file_handler.validate_file(filepath)
        
        if not is_valid:
            filepath.unlink()  # Delete invalid file
            return jsonify({'error': error_msg}), 400
        
        # Process based on type
        if file_type == 'audio':
            result = process_audio(filepath)
        else:
            result = process_image(filepath)
        
        # Store in session
        session_data = get_session_data()
        session_data['uploaded_file'] = str(filepath)
        session_data['file_type'] = file_type
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def process_audio(filepath):
    """Process audio file"""
    # Preprocess
    tensor, spectrogram = audio_processor.preprocess(filepath)
    
    # Store in session
    session_data = get_session_data()
    session_data['processed_tensor'] = tensor
    session_data['original_data'] = spectrogram
    
    # Get info
    info = audio_processor.get_audio_info(filepath)
    
    # Get compatible models
    models = compatibility_checker.get_compatible_models('audio')
    
    return {
        'success': True,
        'file_type': 'audio',
        'info': {
            'duration': info.get('duration_seconds', 'N/A'),
            'sample_rate': info.get('sample_rate', 'N/A'),
            'shape': str(tensor.shape)
        },
        'compatible_models': models,
        'compatible_xai': compatibility_checker.get_compatible_xai_methods('audio')
    }


def process_image(filepath):
    """Process image file"""
    # Preprocess
    normalized, unnormalized, original = image_processor.preprocess(filepath)
    
    # Store in session
    session_data = get_session_data()
    session_data['processed_tensor'] = normalized
    session_data['original_data'] = np.array(original)
    
    # Get info
    info = image_processor.get_image_info(filepath)
    
    # Get compatible models
    models = compatibility_checker.get_compatible_models('image')
    
    return {
        'success': True,
        'file_type': 'image',
        'info': {
            'width': info.get('width', 'N/A'),
            'height': info.get('height', 'N/A'),
            'shape': str(normalized.shape)
        },
        'compatible_models': models,
        'compatible_xai': compatibility_checker.get_compatible_xai_methods('image'),
        'image_url': f'/api/uploaded-image'
    }


@app.route('/api/uploaded-image')
def get_uploaded_image():
    """Return uploaded image"""
    session_data = get_session_data()
    
    if session_data['uploaded_file'] is None:
        return jsonify({'error': 'No image uploaded'}), 404
    
    return send_file(session_data['uploaded_file'], mimetype='image/png')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction with selected model"""
    
    data = request.get_json()
    model_key = data.get('model')
    
    if not model_key:
        return jsonify({'error': 'No model specified'}), 400
    
    session_data = get_session_data()
    
    if session_data['processed_tensor'] is None:
        return jsonify({'error': 'No file uploaded'}), 400
    
    try:
        # Load model
        if session_data['file_type'] == 'audio':
            model = model_loader.load_audio_model(model_key, use_pretrained=False)
            class_names = AUDIO_CONFIG['classes']
        else:
            model = model_loader.load_image_model(model_key, use_pretrained=False)
            class_names = IMAGE_CONFIG['classes']
        
        # Store model
        session_data['loaded_model'] = model
        session_data['selected_model'] = model_key
        
        # Make prediction
        result = quick_predict(model, session_data['processed_tensor'], session_data['file_type'])
        session_data['prediction_result'] = result
        
        return jsonify({
            'success': True,
            'model': model_key,
            'prediction': result['predicted_class'],
            'confidence': float(result['confidence']),
            'probabilities': {k: float(v) for k, v in result['all_probabilities'].items()}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain():
    """Generate XAI explanation"""
    
    data = request.get_json()
    xai_method = data.get('method', 'lime')
    
    session_data = get_session_data()
    
    if session_data['loaded_model'] is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    if session_data['prediction_result'] is None:
        return jsonify({'error': 'No prediction made'}), 400
    
    # Check compatibility
    if not compatibility_checker.is_xai_compatible(xai_method, session_data['file_type']):
        return jsonify({'error': f'{xai_method.upper()} not compatible with {session_data["file_type"]}'}), 400
    
    try:
        if xai_method == 'lime':
            class_names = AUDIO_CONFIG['classes'] if session_data['file_type'] == 'audio' else IMAGE_CONFIG['classes']
            
            # Generate explanation
            save_path = Path(f"outputs/visualizations/flask_{session_data['file_type']}_lime_{session_data['selected_model']}.png")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            explanation_img, scores, fig = explain_with_lime(
                model=session_data['loaded_model'],
                input_tensor=session_data['processed_tensor'],
                original_data=session_data['original_data'],
                input_type=session_data['file_type'],
                class_names=class_names,
                prediction_result=session_data['prediction_result'],
                save_path=save_path
            )
            
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            return jsonify({
                'success': True,
                'method': xai_method,
                'explanation_url': f'/api/explanation-image?path={save_path.name}',
                'num_features': len(scores)
            })
        else:
            return jsonify({'error': f'{xai_method.upper()} not yet implemented'}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/explanation-image')
def get_explanation_image():
    """Return explanation image"""
    filename = request.args.get('path')
    filepath = Path('outputs/visualizations') / filename
    
    if not filepath.exists():
        return jsonify({'error': 'Explanation not found'}), 404
    
    return send_file(filepath, mimetype='image/png')


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session data"""
    session_id = get_session_id()
    
    # Clean up uploaded file
    if session_id in sessions_data:
        if sessions_data[session_id]['uploaded_file']:
            try:
                Path(sessions_data[session_id]['uploaded_file']).unlink()
            except:
                pass
        
        # Reset session data
        sessions_data[session_id] = {
            'uploaded_file': None,
            'file_type': None,
            'processed_tensor': None,
            'original_data': None,
            'selected_model': None,
            'loaded_model': None,
            'prediction_result': None
        }
    
    return jsonify({'success': True})


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 10MB)'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    Path('templates').mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("🚀 Unified XAI Interface - Flask Version")
    print("="*70)
    print(f"\n📊 Device: {DEVICE}")
    print(f"🎵 Audio Models: {len(compatibility_checker.get_compatible_models('audio'))}")
    print(f"🖼️  Image Models: {len(compatibility_checker.get_compatible_models('image'))}")
    print(f"🔍 XAI Methods: {len(compatibility_checker.xai_methods)}")
    print("\n" + "="*70)
    print("🌐 Starting server...")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
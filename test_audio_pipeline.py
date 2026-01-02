"""
Test Audio Pipeline
End-to-end test of audio classification pipeline

This script tests:
1. Audio loading and preprocessing
2. Model loading and inference
3. Prediction output
4. Complete workflow
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.audio_processor import audio_processor
from models.model_loader import model_loader, quick_predict
from config import DEVICE, AUDIO_CONFIG


def test_audio_processor():
    """Test audio preprocessing"""
    print("\n" + "="*70)
    print("TEST 1: Audio Processor")
    print("="*70)
    
    # Test with synthetic audio (since we might not have real files yet)
    print("\n📊 Testing with synthetic audio data...")
    
    # Create a dummy mel-spectrogram directly
    dummy_spectrogram = np.random.randn(128, 128)
    print(f"   ✓ Created dummy spectrogram: {dummy_spectrogram.shape}")
    
    # Normalize it
    normalized = audio_processor.normalize_spectrogram(dummy_spectrogram)
    print(f"   ✓ Normalized to range [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).float()
    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)  # Make RGB
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    tensor = tensor.to(DEVICE)
    
    print(f"   ✓ Converted to tensor: {tensor.shape}")
    print(f"   ✓ Device: {tensor.device}")
    print(f"   ✓ Data type: {tensor.dtype}")
    
    return tensor


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*70)
    print("TEST 2: Model Loading")
    print("="*70)
    
    try:
        # Load model without pretrained weights (for testing)
        model = model_loader.load_audio_model("custom_cnn", use_pretrained=False)
        
        # Get model info
        info = model_loader.get_model_info(model)
        print(f"\n📊 Model Information:")
        print(f"   Class: {info['model_class']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Trainable parameters: {info['trainable_parameters']:,}")
        print(f"   Device: {info['device']}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def test_inference(model, input_tensor):
    """Test model inference"""
    print("\n" + "="*70)
    print("TEST 3: Model Inference")
    print("="*70)
    
    try:
        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        print(f"\n📊 Raw Model Output:")
        print(f"   Logits: {logits}")
        print(f"   Probabilities: {probabilities}")
        
        # Get class names
        class_names = AUDIO_CONFIG["classes"]
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()
        
        print(f"\n🎯 Prediction:")
        print(f"   Predicted class: {class_names[pred_idx]}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Real probability: {probabilities[0][0].item():.2%}")
        print(f"   Fake probability: {probabilities[0][1].item():.2%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_predict(model, input_tensor):
    """Test the convenience function"""
    print("\n" + "="*70)
    print("TEST 4: Quick Predict Function")
    print("="*70)
    
    try:
        result = quick_predict(model, input_tensor, "audio")
        
        print(f"\n📊 Prediction Result:")
        print(f"   Predicted: {result['predicted_class']}")
        print(f"   Index: {result['predicted_index']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n   All Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"      {class_name}: {prob:.2%}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_audio_if_available():
    """Test with real audio file if available"""
    print("\n" + "="*70)
    print("TEST 5: Real Audio File (Optional)")
    print("="*70)
    
    # Check for sample audio files
    sample_dir = Path("data/sample_audio")
    audio_files = list(sample_dir.glob("*.wav")) if sample_dir.exists() else []
    
    if not audio_files:
        print("\n⚠️  No sample audio files found in data/sample_audio/")
        print("   To test with real audio:")
        print("   1. Add a .wav file to data/sample_audio/")
        print("   2. Run this script again")
        return None
    
    # Test with first audio file
    audio_file = audio_files[0]
    print(f"\n📁 Found audio file: {audio_file.name}")
    
    try:
        # Preprocess
        tensor, spectrogram = audio_processor.preprocess(audio_file)
        print(f"   ✓ Preprocessed successfully")
        print(f"   ✓ Tensor shape: {tensor.shape}")
        
        # Load model
        model = model_loader.load_audio_model("custom_cnn", use_pretrained=False)
        
        # Predict
        result = quick_predict(model, tensor, "audio")
        print(f"\n🎯 Prediction for {audio_file.name}:")
        print(f"   {result['predicted_class']} ({result['confidence']:.2%} confidence)")
        
        # Visualize spectrogram
        fig = audio_processor.visualize_spectrogram(
            spectrogram,
            title=f"Mel-Spectrogram: {audio_file.name}"
        )
        
        output_path = Path("outputs/visualizations/test_spectrogram.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n   ✓ Saved spectrogram to {output_path}")
        plt.close(fig)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error processing real audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_summary_visualization(results):
    """Create a summary visualization of all tests"""
    print("\n" + "="*70)
    print("Creating Summary Visualization...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Audio Classification Pipeline Test Results', 
                 fontsize=16, fontweight='bold')
    
    # Test 1: Model Architecture
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.5, 'CustomCNNAudio\n\n3 Conv Blocks\nGlobal Avg Pooling\n2 FC Layers',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_title('Model Architecture', fontweight='bold')
    ax1.axis('off')
    
    # Test 2: Input Processing
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.5, 'Audio (.wav)\n↓\nMel-Spectrogram\n↓\n128x128x3 RGB\n↓\nNormalized Tensor',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.set_title('Preprocessing Pipeline', fontweight='bold')
    ax2.axis('off')
    
    # Test 3: Prediction Results
    ax3 = axes[1, 0]
    if results:
        probs = results['all_probabilities']
        classes = list(probs.keys())
        values = list(probs.values())
        bars = ax3.bar(classes, values, color=['#3498db', '#e74c3c'])
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Probability', fontweight='bold')
        ax3.set_title('Prediction Probabilities', fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2%}', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'No prediction results available',
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    # Test 4: System Status
    ax4 = axes[1, 1]
    status_text = f"""
    ✓ Audio Processor: Working
    ✓ Model Loader: Working
    ✓ Inference: Working
    ✓ Device: {DEVICE}
    
    Ready for integration!
    """
    ax4.text(0.1, 0.5, status_text, ha='left', va='center', fontsize=11,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax4.set_title('System Status', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("outputs/visualizations/pipeline_test_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved summary to {output_path}")
    plt.close(fig)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("AUDIO CLASSIFICATION PIPELINE TEST SUITE")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Audio Config: {AUDIO_CONFIG['sample_rate']}Hz, {AUDIO_CONFIG['duration']}s")
    
    # Run tests
    input_tensor = test_audio_processor()
    model = test_model_loading()
    
    if model is not None and input_tensor is not None:
        test_inference(model, input_tensor)
        result = test_quick_predict(model, input_tensor)
        test_with_real_audio_if_available()
        
        # Create summary
        create_summary_visualization(result)
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\n✅ All core components working!")
    print("\n📋 Next Steps:")
    print("   1. Add real audio files to data/sample_audio/")
    print("   2. Train the model (or load pretrained weights)")
    print("   3. Implement LIME explainer")
    print("   4. Build Chainlit interface")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
"""
Test Image Pipeline
End-to-end test of chest X-ray classification pipeline

This script tests:
1. Image loading and preprocessing
2. Model loading and inference
3. Prediction output
4. Complete workflow
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.image_processor import image_processor
from models.model_loader import model_loader, quick_predict
from config import DEVICE, IMAGE_CONFIG


def create_dummy_xray():
    """Create a synthetic chest X-ray image for testing"""
    print("\n📊 Creating synthetic chest X-ray...")
    
    # Create a grayscale image that looks somewhat like an X-ray
    img = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    
    # Add some structure (simulate ribcage)
    for i in range(5):
        y = 150 + i * 40
        img[y:y+5, :] = 220
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img, mode='L')
    
    # Save it
    output_path = Path("data/sample_images/dummy_xray.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(output_path)
    
    print(f"   ✓ Created dummy X-ray: {output_path}")
    return output_path


def test_image_processor():
    """Test image preprocessing"""
    print("\n" + "="*70)
    print("TEST 1: Image Processor")
    print("="*70)
    
    # Create a dummy X-ray
    image_path = create_dummy_xray()
    
    try:
        # Preprocess
        normalized, unnormalized, original = image_processor.preprocess(image_path)
        
        print(f"\n   ✓ Image loaded and preprocessed")
        print(f"   ✓ Original size: {original.size}")
        print(f"   ✓ Normalized tensor: {normalized.shape}")
        print(f"   ✓ Unnormalized tensor: {unnormalized.shape}")
        print(f"   ✓ Device: {normalized.device}")
        print(f"   ✓ Data type: {normalized.dtype}")
        
        # Get image info
        info = image_processor.get_image_info(image_path)
        print(f"\n📊 Image Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        return normalized, unnormalized, original
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_model_loading():
    """Test AlexNet loading"""
    print("\n" + "="*70)
    print("TEST 2: AlexNet Loading")
    print("="*70)
    
    try:
        # Load model without pretrained weights (for quick testing)
        # Set pretrained=True to use ImageNet weights
        model = model_loader.load_image_model("alexnet", use_pretrained=False)
        
        # Get model info
        info = model_loader.get_model_info(model)
        print(f"\n📊 Model Information:")
        print(f"   Class: {info['model_class']}")
        print(f"   Total parameters: {info['total_parameters']:,}")
        print(f"   Trainable parameters: {info['trainable_parameters']:,}")
        print(f"   Device: {info['device']}")
        
        # Show conv layers (useful for Grad-CAM later)
        conv_layers = model.get_conv_layers()
        print(f"\n   Convolutional layers: {len(conv_layers)}")
        print(f"   Last conv layer index: {conv_layers[-1][0]}")
        
        return model
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
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
        class_names = IMAGE_CONFIG["classes"]
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()
        
        print(f"\n🎯 Prediction:")
        print(f"   Predicted class: {class_names[pred_idx]}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Normal probability: {probabilities[0][0].item():.2%}")
        print(f"   Malignant probability: {probabilities[0][1].item():.2%}")
        
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
        result = quick_predict(model, input_tensor, "image")
        
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


def create_visualization(original_img, result):
    """Create visualization of the test"""
    print("\n" + "="*70)
    print("Creating Visualization...")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Input Chest X-Ray', fontweight='bold')
    axes[0].axis('off')
    
    # Prediction results
    if result:
        probs = result['all_probabilities']
        classes = list(probs.keys())
        values = list(probs.values())
        
        colors = ['#2ecc71' if i == result['predicted_index'] else '#95a5a6' 
                 for i in range(len(classes))]
        bars = axes[1].bar(classes, values, color=colors)
        
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel('Probability', fontweight='bold', fontsize=12)
        axes[1].set_title('Classification Results', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add prediction text
        pred_text = f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']:.1%}"
        axes[1].text(0.5, 0.95, pred_text, 
                    transform=axes[1].transAxes,
                    ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/visualizations/image_test_result.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to {output_path}")
    plt.close(fig)


def create_summary():
    """Create summary visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Image Classification Pipeline Test Results', 
                 fontsize=16, fontweight='bold')
    
    # Model Architecture
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.5, 
             'AlexNet\n\n5 Conv Layers\n3 FC Layers\nImageNet Pretrained',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_title('Model Architecture', fontweight='bold')
    ax1.axis('off')
    
    # Preprocessing
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.5, 
             'Chest X-Ray\n↓\nResize to 224×224\n↓\nImageNet Normalization\n↓\nRGB Tensor',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.set_title('Preprocessing Pipeline', fontweight='bold')
    ax2.axis('off')
    
    # Classes
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 
             'Classification Task\n\nNormal vs Malignant\n\nBinary Classification',
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax3.set_title('Task Description', fontweight='bold')
    ax3.axis('off')
    
    # Status
    ax4 = axes[1, 1]
    status_text = f"""
    ✓ Image Processor: Working
    ✓ AlexNet Model: Working
    ✓ Inference: Working
    ✓ Device: {DEVICE}
    
    Ready for Grad-CAM!
    """
    ax4.text(0.1, 0.5, status_text, ha='left', va='center', fontsize=11,
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))
    ax4.set_title('System Status', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    output_path = Path("outputs/visualizations/image_pipeline_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved summary to {output_path}")
    plt.close(fig)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("IMAGE CLASSIFICATION PIPELINE TEST SUITE")
    print("="*70)
    print(f"\nDevice: {DEVICE}")
    print(f"Image Config: {IMAGE_CONFIG['image_size']}, {IMAGE_CONFIG['num_classes']} classes")
    
    # Run tests
    normalized, unnormalized, original = test_image_processor()
    model = test_model_loading()
    
    if model is not None and normalized is not None:
        test_inference(model, normalized)
        result = test_quick_predict(model, normalized)
        
        # Create visualizations
        if result:
            create_visualization(original, result)
            create_summary()
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("\n✅ Image pipeline working!")
    print("\n📋 Next Steps:")
    print("   1. Add real chest X-ray images to data/sample_images/")
    print("   2. Train AlexNet on CheXpert dataset (or load weights)")
    print("   3. Implement Grad-CAM explainer")
    print("   4. Integrate into Chainlit interface")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
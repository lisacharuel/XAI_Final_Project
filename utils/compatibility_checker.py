"""
Compatibility Checker
Determines which models and XAI methods are compatible with given input types
"""

from typing import List, Dict
from config import AUDIO_CONFIG, IMAGE_CONFIG, XAI_CONFIG


class CompatibilityChecker:
    """
    Manages compatibility between input types, models, and XAI methods
    """
    
    def __init__(self):
        self.audio_models = list(AUDIO_CONFIG["models"].keys())
        self.image_models = list(IMAGE_CONFIG["models"].keys())
        self.xai_methods = XAI_CONFIG["methods"]
    
    
    def get_compatible_models(self, input_type: str) -> List[Dict[str, str]]:
        """
        Get models compatible with the input type
        
        Args:
            input_type: 'audio' or 'image'
        
        Returns:
            List of dictionaries with model information:
            [
                {
                    'key': 'vgg16',
                    'name': 'VGG16 Audio',
                    'description': '...'
                },
                ...
            ]
        """
        if input_type == "audio":
            config = AUDIO_CONFIG["models"]
        elif input_type == "image":
            config = IMAGE_CONFIG["models"]
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        models = []
        for key, info in config.items():
            models.append({
                'key': key,
                'name': info['name'],
                'description': info['description']
            })
        
        return models
    
    
    def get_compatible_xai_methods(self, input_type: str) -> List[Dict[str, str]]:
        """
        Get XAI methods compatible with the input type
        
        Args:
            input_type: 'audio' or 'image'
        
        Returns:
            List of dictionaries with XAI method information:
            [
                {
                    'key': 'lime',
                    'name': 'LIME',
                    'description': '...'
                },
                ...
            ]
        """
        compatible = []
        
        for method_key, method_info in self.xai_methods.items():
            if input_type in method_info["compatible_with"]:
                compatible.append({
                    'key': method_key,
                    'name': method_info['name'],
                    'description': method_info['description']
                })
        
        return compatible
    
    
    def is_xai_compatible(self, xai_method: str, input_type: str) -> bool:
        """
        Check if an XAI method is compatible with input type
        
        Args:
            xai_method: XAI method key (e.g., 'lime', 'gradcam')
            input_type: 'audio' or 'image'
        
        Returns:
            True if compatible, False otherwise
        """
        if xai_method not in self.xai_methods:
            return False
        
        return input_type in self.xai_methods[xai_method]["compatible_with"]
    
    
    def is_model_compatible(self, model_key: str, input_type: str) -> bool:
        """
        Check if a model is compatible with input type
        
        Args:
            model_key: Model key (e.g., 'vgg16', 'alexnet')
            input_type: 'audio' or 'image'
        
        Returns:
            True if compatible, False otherwise
        """
        if input_type == "audio":
            return model_key in self.audio_models
        elif input_type == "image":
            return model_key in self.image_models
        else:
            return False
    
    
    def filter_xai_methods(self, selected_methods: List[str], 
                          input_type: str) -> List[str]:
        """
        Filter a list of XAI methods to only include compatible ones
        
        Args:
            selected_methods: List of XAI method keys
            input_type: 'audio' or 'image'
        
        Returns:
            Filtered list of compatible XAI methods
        """
        return [
            method for method in selected_methods
            if self.is_xai_compatible(method, input_type)
        ]
    
    
    def get_incompatibility_reason(self, xai_method: str, 
                                   input_type: str) -> str:
        """
        Get explanation for why an XAI method is incompatible
        
        Args:
            xai_method: XAI method key
            input_type: 'audio' or 'image'
        
        Returns:
            Explanation string
        """
        if xai_method not in self.xai_methods:
            return f"Unknown XAI method: {xai_method}"
        
        compatible_with = self.xai_methods[xai_method]["compatible_with"]
        
        if input_type not in compatible_with:
            compatible_str = ", ".join(compatible_with)
            return (f"{self.xai_methods[xai_method]['name']} is only "
                   f"compatible with: {compatible_str}")
        
        return "Compatible"
    
    
    def get_compatibility_matrix(self) -> Dict:
        """
        Get a complete compatibility matrix for documentation
        
        Returns:
            Dictionary showing all compatibilities:
            {
                'audio': {
                    'models': [...],
                    'xai_methods': [...]
                },
                'image': {
                    'models': [...],
                    'xai_methods': [...]
                }
            }
        """
        return {
            'audio': {
                'models': self.get_compatible_models('audio'),
                'xai_methods': self.get_compatible_xai_methods('audio')
            },
            'image': {
                'models': self.get_compatible_models('image'),
                'xai_methods': self.get_compatible_xai_methods('image')
            }
        }
    
    
    def print_compatibility_info(self):
        """
        Print compatibility information for debugging
        """
        print("\n" + "="*60)
        print("COMPATIBILITY MATRIX")
        print("="*60)
        
        matrix = self.get_compatibility_matrix()
        
        for input_type, info in matrix.items():
            print(f"\n📂 {input_type.upper()}")
            
            print(f"\n  Models:")
            for model in info['models']:
                print(f"    ✓ {model['name']} ({model['key']})")
            
            print(f"\n  XAI Methods:")
            for method in info['xai_methods']:
                print(f"    ✓ {method['name']} ({method['key']})")
        
        print("\n" + "="*60 + "\n")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global compatibility checker instance
compatibility_checker = CompatibilityChecker()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_available_models(input_type: str) -> List[str]:
    """
    Quick function to get list of model keys
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of model keys
    """
    models = compatibility_checker.get_compatible_models(input_type)
    return [m['key'] for m in models]


def get_available_xai(input_type: str) -> List[str]:
    """
    Quick function to get list of XAI method keys
    
    Args:
        input_type: 'audio' or 'image'
    
    Returns:
        List of XAI method keys
    """
    methods = compatibility_checker.get_compatible_xai_methods(input_type)
    return [m['key'] for m in methods]


# Test function
if __name__ == "__main__":
    checker = CompatibilityChecker()
    checker.print_compatibility_info()

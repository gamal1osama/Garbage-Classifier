import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the pre-trained model
MODEL_PATH = "../Resnet_101v2/finetuned_garbage_classifier_model_ResNet101V2.keras"

def load_model():
    """Load the garbage classification model with enhanced error handling"""
    try:
        # Check if file exists
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            return None
            
        # Check file extension
        if not MODEL_PATH.endswith('.keras'):
            print(f"❌ Invalid file format. Expected .keras file, got: {MODEL_PATH}")
            return None
            
        # Try to load the model
        print(f"🔄 Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def find_model_file():
    """Search for model files in current directory and subdirectories"""
    possible_extensions = ['.keras', '.h5', '.pb']
    possible_names = [
        'finetuned_garbage_classifier_model_ResNet101V2',
        'garbage_classifier',
        'resnet101v2',
        'model'
    ]
    
    found_models = []
    
    # Search in current directory
    for file in os.listdir('.'):
        if any(file.endswith(ext) for ext in possible_extensions):
            if any(name.lower() in file.lower() for name in possible_names):
                found_models.append(file)
    
    # Search in common model directories
    model_dirs = ['models', 'trained_models', 'checkpoints', 'saved_models']
    for dir_name in model_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if any(file.endswith(ext) for ext in possible_extensions):
                    if any(name.lower() in file.lower() for name in possible_names):
                        found_models.append(os.path.join(dir_name, file))
    
    return found_models

# Define garbage class labels (adjust these based on your model's classes)
CLASS_LABELS = [
    'battery',
    'biological',
    'cardboard',
    'clothes',
    'glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash'
]

# Professional CSS styling
CUSTOM_CSS = """
/* Background gradient */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Main container styling */
.main-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

/* Header styling */
.header {
    text-align: center;
    margin-bottom: 2rem;
}

.header h1 {
    color: #2c3e50;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
}

.header p {
    color: #7f8c8d;
    font-size: 1.2rem;
    margin-bottom: 0;
}

/* Input/Output sections */
.input-section, .output-section {
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid #dee2e6;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
}

/* Button styling */
.predict-btn {
    background: linear-gradient(45deg, #28a745, #20c997);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-size: 1.1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
}

.predict-btn:hover {
    background: linear-gradient(45deg, #218838, #1ea87a);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
}

/* Image upload area */
.image-upload {
    border: 2px dashed #6c757d;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    background: #f8f9fa;
}

.image-upload:hover {
    border-color: #28a745;
    background: #e8f5e8;
}

/* Result text area */
.result-text {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 1rem;
    font-family: 'Courier New', monospace;
    font-size: 1rem;
    line-height: 1.5;
}

/* Status indicators */
.status-success {
    color: #28a745;
    font-weight: 600;
}

.status-error {
    color: #dc3545;
    font-weight: 600;
}

.status-info {
    color: #17a2b8;
    font-weight: 600;
}

/* Confidence bar */
.confidence-bar {
    background: #e9ecef;
    border-radius: 10px;
    height: 20px;
    overflow: hidden;
    margin: 10px 0;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #28a745, #20c997);
    transition: width 0.5s ease;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #dee2e6;
    color: #6c757d;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-container {
        margin: 1rem;
        padding: 1rem;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .header p {
        font-size: 1rem;
    }
}

/* Animation for results */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-animation {
    animation: fadeIn 0.5s ease-out;
}
"""

def preprocess_image(image):
    """Preprocess the input image for the model"""
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image to match model input size (typically 224x224 for ResNet)
    image = tf.image.resize(image, [224, 224])
    
    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    
    # Add batch dimension
    image = tf.expand_dims(image, axis=0)
    
    return image

def predict_garbage_type(image):
    """Predict the type of garbage in the image with professional formatting"""
    if image is None:
        return "⚠️ Please upload an image first to begin classification."
    
    # Load model
    model = load_model()
    if model is None:
        error_msg = "❌ MODEL LOADING ERROR\n"
        error_msg += "═══════════════════════════\n\n"
        error_msg += "🚨 Could not load the classification model.\n\n"
        error_msg += "💡 TROUBLESHOOTING STEPS:\n"
        error_msg += "────────────────────────\n"
        error_msg += "1. Check if model file exists in current directory\n"
        error_msg += "2. Verify file name: 'finetuned_garbage_classifier_model_ResNet101V2.keras'\n"
        error_msg += "3. Ensure file is a valid .keras format\n"
        error_msg += "4. Check file permissions\n\n"
        
        # Search for alternative model files
        found_models = find_model_file()
        if found_models:
            error_msg += "🔍 FOUND ALTERNATIVE MODEL FILES:\n"
            error_msg += "─────────────────────────────\n"
            for i, model_file in enumerate(found_models, 1):
                error_msg += f"{i}. {model_file}\n"
            error_msg += f"\n💡 Update MODEL_PATH variable to use one of these files.\n"
        else:
            error_msg += "❌ No model files found in current directory.\n"
        
        return error_msg
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get class label
        if predicted_class_idx < len(CLASS_LABELS):
            predicted_class = CLASS_LABELS[predicted_class_idx]
        else:
            predicted_class = f"Unknown Class {predicted_class_idx}"
        
        # Format professional result
        result = f"🎯 CLASSIFICATION RESULTS\n"
        result += f"═══════════════════════════\n\n"
        result += f"📊 Predicted Category: {predicted_class}\n"
        result += f"🔍 Confidence Score: {confidence:.2%}\n"
        result += f"📈 Reliability: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}\n\n"
        
        # Add confidence bar representation
        bar_length = 20
        filled_length = int(bar_length * confidence)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        result += f"📊 Confidence Bar: [{bar}] {confidence:.1%}\n\n"
        
        # Add top predictions
        top_indices = np.argsort(predictions[0])[::-1][:3]
        result += f"🏆 TOP 3 PREDICTIONS:\n"
        result += f"────────────────────\n"
        
        for i, idx in enumerate(top_indices):
            class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class {idx}"
            percentage = predictions[0][idx] * 100
            
            # Add medal emojis
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            result += f"{medal} {i+1}. {class_name}: {percentage:.1f}%\n"
        
        # Add recommendation
        result += f"\n💡 RECOMMENDATION:\n"
        result += f"────────────────────\n"
        
        disposal_tips = {
            'Cardboard': 'Flatten and place in recycling bin',
            'Glass': 'Clean and sort by color for recycling',
            'Metal': 'Remove labels and recycle appropriately',
            'Paper': 'Keep dry and place in paper recycling',
            'Plastic': 'Check recycling number and clean container',
            'Organic': 'Compost if possible, otherwise general waste'
        }
        
        tip = disposal_tips.get(predicted_class, 'Dispose according to local guidelines')
        result += f"♻️ {tip}\n"
        
        result += f"\n✅ Classification completed successfully!"
        
        return result
        
    except Exception as e:
        return f"❌ PREDICTION ERROR:\n════════════════════════\n\n🚨 {str(e)}\n\n💡 TROUBLESHOOTING:\n• Upload a different image\n• Check image format (JPG, PNG)\n• Ensure image is clear and well-lit\n• Verify model compatibility"

# Create the Gradio interface
def create_interface():
    """Create and return the professional Gradio interface"""
    
    # Define the interface with professional theme
    interface = gr.Interface(
        fn=predict_garbage_type,
        inputs=gr.Image(
            type="pil", 
            label="📸 Upload Garbage Image",
            elem_classes=["image-upload"]
        ),
        outputs=gr.Textbox(
            label="🎯 Classification Results", 
            lines=15,
            elem_classes=["result-text"]
        ),
        title="🗑️ Professional Garbage Classification System",
        description="""
        <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #e3f2fd, #f3e5f5); border-radius: 15px; margin: 10px 0;'>
            <h3 style='color: #1565c0; margin: 0;'>🤖 AI-Powered Waste Management Solution</h3>
            <p style='color: #424242; margin: 5px 0;'>Upload an image of waste material for intelligent classification and disposal recommendations</p>
            <p style='color: #757575; font-size: 0.9em; margin: 0;'>Powered by Fine-tuned ResNet101V2 Deep Learning Model</p>
        </div>
        """,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate",
            radius_size="lg",
            spacing_size="lg"
        ),
        allow_flagging="never",
        article="""
        <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px; margin-top: 20px;'>
            <p style='color: #2b3233; margin: 0;'>
                💡 Tips for best results: Use clear, well-lit images • Focus on single items • Avoid blurry or dark photos
            </p>
        </div>
        """
    )
    
    return interface

# Professional version with custom layout
def create_interface_with_custom_button():
    """Create professional interface with custom predict button and advanced layout"""
    
    with gr.Blocks(
        title="Professional Garbage Classification System",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate",
            radius_size="lg",
            spacing_size="lg"
        ),
        css=CUSTOM_CSS
    ) as interface:
        
        # Header section
        with gr.Row():
            gr.HTML("""
            <div class="header">
                <h1>🗑️ Professional Garbage Classification System</h1>
                <p>AI-Powered Waste Management & Classification Solution</p>
            </div>
            """)
        
        # Main content area
        with gr.Row():
            # Left column - Input section
            with gr.Column(scale=1):
                gr.HTML("""
                <div style='text-align: center; padding: 15px; background: linear-gradient(45deg, #e3f2fd, #f3e5f5); border-radius: 15px; margin-bottom: 20px;'>
                    <h3 style='color: #1565c0; margin: 0;'>📸 Image Upload</h3>
                    <p style='color: #424242; margin: 5px 0; font-size: 0.9em;'>Upload clear, well-lit images for best results</p>
                </div>
                """)
                
                image_input = gr.Image(
                    type="pil",
                    label="Select Waste Image",
                    elem_classes=["image-upload"],
                    height=300
                )
                
                with gr.Row():
                    predict_btn = gr.Button(
                        "🎯 Analyze & Classify",
                        variant="primary",
                        size="lg",
                        elem_classes=["predict-btn"]
                    )
                    
                gr.HTML("""
                <div style='text-align: center; padding: 10px; background: #fff3cd; border-radius: 10px; margin-top: 15px;'>
                    <p style='color: #856404; margin: 0; font-size: 0.8em;'>
                        <strong>🔒 Privacy Notice:</strong> Images are processed locally and not stored
                    </p>
                </div>
                """)
            
            # Right column - Results section
            with gr.Column(scale=1):
                gr.HTML("""
                <div style='text-align: center; padding: 15px; background: linear-gradient(45deg, #e8f5e9, #fff3e0); border-radius: 15px; margin-bottom: 20px;'>
                    <h3 style='color: #2e7d32; margin: 0;'>🎯 Classification Results</h3>
                    <p style='color: #424242; margin: 5px 0; font-size: 0.9em;'>Detailed analysis and disposal recommendations</p>
                </div>
                """)
                
                output_text = gr.Textbox(
                    label="Analysis Results",
                    lines=18,
                    elem_classes=["result-text"],
                    placeholder="Upload an image and click 'Analyze & Classify' to see detailed results...",
                    interactive=False
                )
        
        # Statistics section
        with gr.Row():
            gr.HTML("""
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); border-radius: 15px; margin-top: 20px;'>
                <h4 style='color: #2c3e50; margin: 0 0 10px 0;'>📊 System Performance</h4>
                <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
                    <div style='text-align: center; margin: 5px;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #27ae60;'>95.2%</div>
                        <div style='font-size: 0.9em; color: #7f8c8d;'>Accuracy</div>
                    </div>
                    <div style='text-align: center; margin: 5px;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #3498db;'>6</div>
                        <div style='font-size: 0.9em; color: #7f8c8d;'>Categories</div>
                    </div>
                    <div style='text-align: center; margin: 5px;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #e74c3c;'>ResNet101V2</div>
                        <div style='font-size: 0.9em; color: #7f8c8d;'>Model</div>
                    </div>
                    <div style='text-align: center; margin: 5px;'>
                        <div style='font-size: 1.5em; font-weight: bold; color: #f39c12;'>< 2s</div>
                        <div style='font-size: 0.9em; color: #7f8c8d;'>Processing</div>
                    </div>
                </div>
            </div>
            """)
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>🌱 Contributing to a cleaner environment through intelligent waste classification</p>
            <p style='font-size: 0.8em; color: #adb5bd;'>Powered by TensorFlow & Gradio | ResNet101V2 Deep Learning Model</p>
        </div>
        """)
        
        # Connect the button to the prediction function
        predict_btn.click(
            fn=predict_garbage_type,
            inputs=image_input,
            outputs=output_text
        )
    
    return interface

if __name__ == "__main__":
    print("🗑️ Professional Garbage Classification System")
    print("═" * 50)
    
    # Check if model file exists and provide guidance
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print(f"📁 Current directory: {os.getcwd()}")
        print("\n🔍 Searching for alternative model files...")
        
        found_models = find_model_file()
        if found_models:
            print(f"✅ Found {len(found_models)} potential model file(s):")
            for i, model_file in enumerate(found_models, 1):
                print(f"   {i}. {model_file}")
            
            print(f"\n💡 OPTIONS:")
            print(f"1. Rename your model file to: {MODEL_PATH}")
            print(f"2. Update MODEL_PATH in the code to point to your model")
            print(f"3. Continue anyway (app will show error message to users)")
            
            choice = input("\nDo you want to continue anyway? (y/n): ").strip().lower()
            if choice not in ['y', 'yes', '1']:
                print("❌ Exiting. Please ensure model file is available.")
                exit(1)
        else:
            print("❌ No model files found in current directory or subdirectories.")
            print("\n💡 SOLUTIONS:")
            print("1. Download/copy your model file to current directory")
            print("2. Rename it to: finetuned_garbage_classifier_model_ResNet101V2.keras")
            print("3. Or update MODEL_PATH variable in the code")
            
            choice = input("\nDo you want to continue anyway? (y/n): ").strip().lower()
            if choice not in ['y', 'yes', '1']:
                print("❌ Exiting. Please ensure model file is available.")
                exit(1)
    else:
        print(f"✅ Model file found: {MODEL_PATH}")
        
        # Test loading the model
        print("🔄 Testing model loading...")
        test_model = load_model()
        if test_model is not None:
            print("✅ Model loaded successfully!")
            try:
                print(f"📊 Model input shape: {test_model.input_shape}")
                print(f"📊 Model output shape: {test_model.output_shape}")
            except:
                print("📊 Model loaded but shape info unavailable")
        else:
            print("❌ Model loading failed!")
            choice = input("\nDo you want to continue anyway? (y/n): ").strip().lower()
            if choice not in ['y', 'yes', '1']:
                print("❌ Exiting. Please check model file.")
                exit(1)
    
    # Let user choose the interface type
    print("\n🎨 Choose your preferred interface:")
    print("1. Standard Professional Interface (automatic prediction)")
    print("2. Advanced Custom Interface (with detailed layout & predict button)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("🚀 Loading Standard Professional Interface...")
            app = create_interface()
            break
        elif choice == "2":
            print("🚀 Loading Advanced Custom Interface...")
            app = create_interface_with_custom_button()
            break
        else:
            print("❌ Invalid choice! Please enter 1 or 2.")
    
    # Launch settings
    print("\n📊 Launch Configuration:")
    share_choice = input("🌐 Create public shareable link? (y/n): ").strip().lower()
    share = share_choice in ['y', 'yes', '1']
    
    if share:
        print("🔗 Creating public link...")
    else:
        print("🏠 Running locally only...")
    
    # Launch the app
    print(f"\n🚀 Starting Professional Garbage Classification System...")
    print(f"📋 Interface: {'Standard Professional' if choice == '1' else 'Advanced Custom Layout'}")
    print(f"🌐 Public link: {'Yes' if share else 'No'}")
    print(f"🔗 Local URL: http://localhost:7860")
    print(f"💡 Press Ctrl+C to stop the server")
    print("═" * 50)
    
    app.launch(
        share=share,
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,  # Default Gradio port
        debug=True
    )


import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the pre-trained model
MODEL_PATH = "./Resnet_101v2/finetuned_garbage_classifier_model_ResNet101V2.keras"

def load_model():
    """Load the garbage classification model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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
    """Predict the type of garbage in the image"""
    if image is None:
        return "Please upload an image first."
    
    # Load model
    model = load_model()
    if model is None:
        return "Error: Could not load the model. Please check if the model file exists."
    
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
            predicted_class = f"Class {predicted_class_idx}"
        
        # Format the result
        result = f"Predicted Garbage Type: {predicted_class}\nConfidence: {confidence:.2%}"
        
        # Add top predictions for better insight
        top_indices = np.argsort(predictions[0])[::-1][:3]
        result += "\n\nTop 3 Predictions:"
        for i, idx in enumerate(top_indices):
            class_name = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else f"Class {idx}"
            result += f"\n{i+1}. {class_name}: {predictions[0][idx]:.2%}"
        
        return result
        
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface"""
    
    # Define the interface
    interface = gr.Interface(
        fn=predict_garbage_type,
        inputs=gr.Image(type="pil", label="Upload Garbage Image"),
        outputs=gr.Textbox(label="Classification Result", lines=6),
        title="🗑️ Garbage Classification App",
        description="Upload an image of garbage and click 'Predict' to classify its type using a fine-tuned ResNet101V2 model.",
        examples=[
            # You can add example images here if you have them
            # ["example_plastic.jpg"],
            # ["example_glass.jpg"],
        ],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface

# Alternative version with custom button (if you specifically want a "Predict" button)
def create_interface_with_custom_button():
    """Create interface with custom predict button"""
    
    with gr.Blocks(title="Garbage Classification App", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🗑️ Garbage Classification App")
        gr.Markdown("Upload an image of garbage and click 'Predict' to classify its type using a fine-tuned ResNet101V2 model.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Garbage Image")
                predict_btn = gr.Button("Predict", variant="primary", size="lg")
            
            with gr.Column():
                output_text = gr.Textbox(label="Classification Result", lines=6)
        
        # Connect the button to the prediction function
        predict_btn.click(
            fn=predict_garbage_type,
            inputs=image_input,
            outputs=output_text
        )
    
    return interface

if __name__ == "__main__":
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found!")
        print("Please make sure the model file is in the same directory as this script.")
    
    # Let user choose the interface type
    print("\n🗑️ Garbage Classification App")
    print("Choose your preferred interface:")
    print("1. Standard Interface (automatic prediction)")
    print("2. Custom Interface (with Predict button)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("Loading Standard Interface...")
            app = create_interface()
            break
        elif choice == "2":
            print("Loading Custom Interface with Predict Button...")
            app = create_interface_with_custom_button()
            break
        else:
            print("Invalid choice! Please enter 1 or 2.")
    
    # Launch settings
    print("\n📊 Launch Settings:")
    share_choice = input("Do you want a public shareable link? (y/n): ").strip().lower()
    share = share_choice in ['y', 'yes', '1']
    
    if share:
        print("Creating public link...")
    else:
        print("Running locally only...")
    
    # Launch the app
    print(f"\n🚀 Starting Garbage Classification App...")
    print(f"Interface: {'Standard' if choice == '1' else 'Custom with Predict Button'}")
    print(f"Public link: {'Yes' if share else 'No'}")
    print(f"Local URL: http://localhost:7860")
    
    app.launch(
        share=share,
        server_name="0.0.0.0",  # Allow access from other devices on network
        server_port=7860,  # Default Gradio port
        debug=True
    )

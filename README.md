# Garbage Classification Project

AI-powered waste sorting system using deep learning to classify different types of garbage.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/yahiakenawy/Garbage_Classification)

## Overview

This project uses machine learning to automatically classify garbage into different categories. It helps improve waste management and recycling by identifying waste types from images.

**Key Features:**
- 95.2% classification accuracy
- 10 different waste categories
- Real-time image processing
- Easy-to-use web interface
- Two model architectures: CNN and ResNet-101v2

## Classification Categories

The system can identify these waste types:
- Glass
- Paper
- Cardboard
- Plastic
- Organic waste
- E-waste
- Other materials

## Project Structure

```
Garbage-Classifier/
├── .gradio/          # Web interface files
├── CNN/              # CNN model code
├── Deployment/       # Deployment files
├── Resnet 101v2/     # ResNet model code
├── app.py            # Main application
└── requirements.txt  # Dependencies
```

## Quick Start

### Try Online Demo
[🔗 Live Demo on Hugging Face](https://huggingface.co/spaces/yahiakenawy/Garbage_Classification)

### Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gamal1osama/Garbage-Classifier.git
   cd Garbage-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and go to:** `http://localhost:7860`

## How to Use

1. Upload an image of garbage
2. The AI will classify the waste type
3. Get disposal recommendations
4. Switch between CNN and ResNet models for comparison

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- Gradio 3.0+
- Other dependencies in `requirements.txt`

## Models

- **ResNet-101v2**: Pre-trained model fine-tuned for garbage classification
- **CNN**: Custom convolutional neural network built from scratch

## Performance

- **Accuracy**: 95.2%
- **F1-Score**: 0.96
- **Dataset**: 20,000 images across 10 categories

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contributors

- [Gamal Osama]
- [Yahia Kenawy]



**Help make waste management smarter with AI!**

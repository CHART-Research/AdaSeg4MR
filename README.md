# AdaSeg4MR: Adaptive Speech-Guided Instance Segmentation for Mixed Reality

## Overview

AdaSeg4MR is a sophisticated multi-modal AI assistant named after Ada Lovelace, featuring real-time object segmentation, voice interaction, and comprehensive visual analysis capabilities. The system integrates YOLO-based object detection, natural language processing, and computer vision to provide intelligent responses to user queries about visual content.

## Core Features

### 1. Real-Time Object Segmentation
- **YOLO11M Integration**: Uses YOLO11M segmentation model for precise object detection and instance segmentation
- **Multi-Class Support**: Detects 80 COCO classes with customizable target selection
- **Real-Time Processing**: Continuous frame analysis with live overlay visualization
- **Category Label Mapping**: Intelligent translation of natural language to YOLO classes

### 2. Multi-Modal Interaction
- **Voice Recognition**: Speech-to-text using Whisper model
- **Text-to-Speech**: OpenAI TTS with natural voice synthesis
- **Keyboard Input**: Alternative text-based interaction mode
- **Visual Analysis**: Screenshot and webcam capture capabilities

### 3. Intelligent Query Processing
- **Intent Detection**: AI-powered understanding of user intentions
- **Context-Aware Responses**: Maintains conversation history and visual context
- **Multi-Language Support**: Handles various natural language expressions
- **Function Calling**: Automatic selection of appropriate system functions

## System Architecture

### Core Components

#### 1. Object Detection Engine
```python
segmentation_model = YOLO("yolo11m-seg.pt")
```
- **Model**: YOLO11M segmentation model
- **Classes**: 80 COCO object categories
- **Confidence Threshold**: 0.15 (configurable)
- **Output**: Bounding boxes, masks, confidence scores

#### 2. Natural Language Processing
- **Groq API**: Primary LLM for intent detection and response generation
- **OpenAI API**: Secondary LLM for complex reasoning tasks
- **Google Gemini**: Vision analysis and image understanding
- **Whisper**: Speech recognition and transcription

#### 3. Visual Processing Pipeline
- **Frame Buffer**: Real-time video frame storage
- **Overlay System**: Segmentation result visualization
- **Image Mode**: Static image analysis capability
- **Recording System**: Video and audio capture functionality

## Installation and Setup

### Prerequisites

#### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: Optional but recommended for faster processing
- **Storage**: 2GB+ free space

#### Python Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `ultralytics` - YOLO model framework
- `opencv-python` - Computer vision operations
- `groq` - LLM API client
- `openai` - OpenAI API client
- `google-generativeai` - Gemini API client
- `faster-whisper` - Speech recognition
- `speech_recognition` - Audio processing
- `pyaudio` - Audio I/O
- `pyperclip` - Clipboard operations
- `numpy` - Numerical computations
- `PIL` - Image processing
- `pynput` - Keyboard monitoring

### Configuration

#### 1. API Keys Setup
Create `config.json` file:
```json
{
    "groq_api_key": "your_groq_api_key",
    "google_api_key": "your_google_api_key", 
    "openai_api_key": "your_openai_api_key"
}
```

#### 2. Model Download
```bash
# Download YOLO11M model (automatic on first run)
# Model will be cached locally
```

#### 3. Directory Structure
```
AdaSeg4MR/
├── ada.py                 # Main camera-based assistant
├── ada_img.py            # Image analysis mode
├── lovelace_img.py       # Evaluation framework
├── category_labels.json  # Generic label mappings
├── config.json          # API configuration
├── test_images/         # Test image directory
├── ../images/           # COCO dataset (for evaluation)
└── ../test_results/     # Evaluation results
```

## Usage Guide

### 1. Camera Mode (ada.py)

#### Starting the System
```bash
python ada.py
```

#### Voice Commands
- **Activation**: Press spacebar to start/stop recording
- **Commands**: Natural language queries about visual content

#### Text Commands
```bash
Command: find people and cars
Command: how many dogs are there?
Command: where is the laptop?
Command: describe what you see
Command: take screenshot
Command: quit
```

#### Key Functions

##### Object Detection
```python
# Find specific objects
"find the person and the car"

# Find by category
"find fruit"  # Detects apple, orange, banana
"find vehicles"  # Detects all transportation types
"find animals"  # Detects all animal classes
```

##### Counting Objects
```python
# Count specific objects
"how many people are there?"
"count the cars and trucks"
```

##### Position Queries
```python
# Get object locations
"where is the laptop?"
"where are the people?"
```

##### Scene Analysis
```python
# General description
"describe what you see"
"analyze the scene"
```

##### Focused Analysis
```python
# Focused Analysis
"What is the color of the hat?"
"Where is the dog looking?"
```

### 2. Image Mode (ada_img.py)

#### Switching Modes
```bash
Command: image mode    # Switch to image analysis
Command: camera mode   # Switch to camera mode
Command: next          # Next image
Command: previous      # Previous image
```

#### Image Analysis Features
- **Batch Processing**: Analyze multiple images sequentially
- **Persistent Results**: Segmentation results remain visible
- **Navigation**: Easy image browsing with keyboard commands
- **Export**: Save analysis results and visualizations

### 3. Evaluation Framework (lovelace_img.py)

#### Purpose
Comprehensive evaluation of system performance using COCO dataset

#### Setup
```bash
# Download COCO dataset
# Place in ../images/ directory
python lovelace_img.py
```

#### Evaluation Metrics
- **Bounding Box IoU**: Intersection over Union for detection accuracy
- **Mask IoU**: Segmentation mask accuracy
- **Class Accuracy**: Correct object classification rate
- **Response Time**: System latency measurements
- **Count Accuracy**: Object counting precision

#### Test Process
1. **Dataset Preparation**: Select random classes and images
2. **Question Generation**: Create standardized test queries
3. **Automated Testing**: Run all queries on selected images
4. **Result Analysis**: Calculate comprehensive metrics
5. **Report Generation**: Create detailed performance reports

## Performance Optimization

### 1. Computational Efficiency
- **CPU Threading**: Multi-core processing for audio and video
- **GPU Acceleration**: Optional CUDA support for YOLO model
- **Memory Management**: Efficient buffer management
- **Frame Skipping**: Adaptive processing rate

### 2. Response Time Optimization
- **Parallel Processing**: Simultaneous audio and visual analysis
- **Caching**: Model and result caching
- **Streaming**: Real-time audio processing
- **Async Operations**: Non-blocking API calls

### 3. Accuracy Improvements
- **Multi-Frame Averaging**: Reduces detection noise
- **Confidence Thresholding**: Adaptive confidence levels
- **Class-Specific Tuning**: Optimized parameters per object type
- **Context Awareness**: Scene understanding for better detection

## Evaluation and Testing

### 1. Automated Testing Framework

#### Test Structure
```python
class AdaImgTester:
    def __init__(self):
        # Initialize COCO dataset
        # Setup evaluation metrics
        # Create result directories
    
    def run_test(self, num_classes=10, num_images_per_class=5):
        # Automated test execution
        # Metric calculation
        # Report generation
```

#### Test Questions
1. **Object Detection**: "find the [object1], [object2], and [object3]"
2. **Object Counting**: "how many [object]s are there?"
3. **Position Query**: "where is the [object]?"
4. **Description**: "what is the [object] like?"

### 2. Performance Metrics

#### Detection Accuracy
- **Bounding Box IoU**: Measures detection precision
- **Mask IoU**: Evaluates segmentation quality
- **Class Accuracy**: Classification correctness
- **Detection Rate**: Percentage of objects found

#### System Performance
- **Response Time**: End-to-end processing latency
- **Throughput**: Images processed per second
- **Memory Usage**: System resource consumption
- **CPU/GPU Utilization**: Computational efficiency

#### User Experience
- **Query Success Rate**: Percentage of successful queries
- **Response Quality**: Relevance and accuracy of answers
- **Interaction Fluency**: Natural conversation flow
- **Error Recovery**: System robustness

### 3. Result Analysis

#### Generated Reports
- **by_image.csv**: Per-image performance metrics
- **by_classes.csv**: Class-specific accuracy analysis
- **total.csv**: Overall system performance
- **HTML Reports**: Visual result presentation

#### Metric Interpretation
```python
# Example metric calculation
avg_bbox_iou = sum(bbox_ious) / len(bbox_ious)
avg_mask_iou = sum(mask_ious) / len(mask_ious)
class_accuracy = correct_predictions / total_predictions
response_time = end_time - start_time
```

## Troubleshooting

### Common Issues

#### 1. API Connection Problems
```bash
# Check API keys in config.json
# Verify internet connection
# Test API endpoints individually
```

#### 2. Model Loading Issues
```bash
# Ensure sufficient disk space
# Check model file integrity
# Verify CUDA installation (if using GPU)
```

#### 3. Audio Problems
```bash
# Check microphone permissions
# Verify audio drivers
# Test with different audio devices
```

#### 4. Performance Issues
```bash
# Reduce image resolution
# Lower confidence thresholds
# Disable unnecessary features
# Check system resources
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor system resources
# Check API response times
# Verify model predictions
```

## Configuration Options

### Global Variables
```python
# Camera settings
CAMERA_SOURCE = 0  # 0: built-in, 1: external, 2: DroidCam

# Voice interaction
use_voice_interaction = True  # Enable/disable voice

# Recording
ENABLE_RECORDING = True  # Video/audio recording

# Processing
continuous_segmentation = True  # Real-time processing
```

### Model Parameters
```python
# YOLO configuration
conf_threshold = 0.15  # Detection confidence
iou_threshold = 0.5    # Non-maximum suppression
classes = None         # Target classes (None = all)

# Whisper settings
whisper_size = 'base'  # Model size
device = 'cpu'         # Processing device
```

## Future Enhancements

### Planned Features
- **Multi-Language Support**: International language support
- **Advanced Segmentation**: Instance-aware segmentation
- **3D Understanding**: Depth perception and 3D analysis
- **Learning Capabilities**: Adaptive model improvement
- **Cloud Integration**: Remote processing and storage

### Performance Improvements
- **Model Optimization**: Quantization and pruning
- **Hardware Acceleration**: Better GPU utilization
- **Parallel Processing**: Enhanced multi-threading
- **Memory Optimization**: Reduced memory footprint

## Contributing

### Development Setup
```bash
# Clone repository
git clone [repository-url]
cd AdaSeg4MR

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black ada.py ada_img.py
```

### Code Structure
- **Modular Design**: Separate concerns and responsibilities
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Robust exception management
- **Testing**: Automated test coverage

## License and Acknowledgments

### License

Elastic License 2.0

URL: https://www.elastic.co/licensing/elastic-license

### Acknowledgments
- **YOLO**: Ultralytics for object detection models
- **COCO**: Microsoft for evaluation dataset
- **OpenAI**: Language models and TTS
- **Groq**: Fast inference API
- **Google**: Gemini vision models

## Contact and Support

### Documentation
- **API Reference**: Detailed function documentation
- **Examples**: Usage examples and tutorials
- **Troubleshooting**: Common issues and solutions

### Support Channels
- **Issues**: GitHub issue tracker
- **Discussions**: Community forum
- **Email**: Direct support contact

---

*This README provides comprehensive documentation for the AdaSeg4MR system. For specific implementation details, refer to the individual source files and their inline documentation.*

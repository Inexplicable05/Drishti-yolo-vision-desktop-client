# Drishti: YOLO Vision

Drishti Studio is a minimalist, modern desktop application designed for AI object detection using YOLO models. 
It provides a clean, "glass-like" user interface that allows you to load custom YOLO weights and run inference seamlessly.

## Features

- **Modern UI**: Clean, responsive layout with real-time inference feedback.
- **Custom Models**: Load any `.pt` YOLO weights file to perform detection.
- **Adjustable Confidence**: Fine-tune the confidence threshold directly from the UI.
- **Image & Video Support**: Run inference on images or other supported media sources.
- **Metrics**: Real-time evaluation of Inference times (ms) and Frames Per Second (FPS).

## Installation & Running

You can run Drishti: YOLO Vision in two ways:

### Option 1: Run the Executable (No setup required)
For a completely hassle-free experience, just run the compiled executable:
1. Download or clone this repository.
2. Double-click on `Drishti.exe`.

### Option 2: Run via Python
If you prefer to run the source code directly or want to modify it:

1. Clone this repository:
```bash
git clone https://github.com/yourusername/drishti-yolo-vision.git
cd drishti-yolo-vision
```

2. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Requirements (for Python source)
- Python 3.8+
- `ultralytics` (YOLO)
- `opencv-python` (cv2)
- `Pillow` (PIL)
- `tkinter`

## License
MIT License

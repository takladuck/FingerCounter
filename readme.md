# Finger Counter

A real-time computer vision application that detects and counts raised fingers using MediaPipe hand tracking. The application provides a detailed visualization of hand gestures, including finger angles and hand type detection.

## Features

- Real-time finger counting for both hands simultaneously  
- Detailed finger angle measurements  
- Hand type detection (Left/Right)  
- Visual feedback with on-screen measurements  
- Robust thumb detection using multiple metrics  
- Support for up to two hands at once  

## Prerequisites

The following Python packages are required:

- OpenCV (`cv2`)  
- MediaPipe (`mediapipe`)  
- NumPy (`numpy`)  

You can install the dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the script:

```bash
python finger_counter.py
```

A window will open showing your webcam feed with real-time hand tracking:

- **Blue text** shows the total finger count  
- **Green text** displays hand type and finger status (0 for lowered, 1 for raised)  
- **Yellow text** shows detailed angle measurements for each finger  

Press `Esc` to exit the application.

## How It Works

The application uses several sophisticated techniques to accurately detect raised fingers:

### Hand Tracking
- Utilizes MediaPipe's hand tracking solution to detect hand landmarks in real-time.
- Detects up to two hands simultaneously with specified detection and tracking confidence.

### Finger Detection
- **For regular fingers (index, middle, ring, pinky):**
  - Identifies key landmarks: MCP (base), PIP (first joint), DIP (second joint), and TIP (fingertip).
  - Calculates vectors between joints to determine finger straightness.
  - Measures the angle between joints using vector calculations. A bend angle greater than 120° indicates the finger is straight.
  - Verifies extension by comparing the distance from the wrist to the fingertip versus the distance from the wrist to the PIP joint. If the fingertip is farther, the finger is considered raised.

- **For thumb:**
  - Uses a specialized detection algorithm due to the thumb's unique range of motion.
  - Analyzes the angle between the palm and the thumb using vectors from the wrist to the index MCP and from the thumb CMC to the thumb tip.
  - Measures the distance ratio between the thumb tip and the wrist compared to the thumb MCP and wrist. A ratio above 1.2 combined with an angle above 20° indicates the thumb is raised.

### Visualization
- Draws hand landmarks and their connections on the video feed.
- Displays:
  - **Total finger count** in blue text.
  - **Hand type (Left/Right)** and finger status (0 for lowered, 1 for raised) in green text.
  - **Detailed angle measurements** for each finger in yellow text.

### Finger Angle Calculation
- For each finger, calculates the angle between vectors formed by key landmarks:
  - **Thumb angles** are calculated relative to the palm.
  - **Other fingers** use angles between adjacent phalanges to determine straightness and bending.

## Configuration

The `FingerCounter` class can be initialized with custom parameters:

```python
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=2
)
```

- **`min_detection_confidence`**: Minimum confidence for hand detection (default: `0.7`)  
- **`min_tracking_confidence`**: Minimum confidence for hand tracking (default: `0.5`)  
- **`max_num_hands`**: Maximum number of hands to detect (default: `2`)  

## Limitations

- Requires good lighting conditions for optimal performance  
- Hand gestures should be clearly visible to the camera  
- Performance may vary based on hardware capabilities  
- Best results are achieved when hands are facing the camera  

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.


"""
Demo script to visualize retinal processing stages using the simple Retina model.
Press keys 0-9 to switch between different visualization modes.
Press 'q' to quit, 'p' to pause.
"""

import cv2
import numpy as np
from visual.retina_scratch import Retina

# Configuration
USE_WEBCAM = True  # Set to False to use a static image
VIDEO_SOURCE = 0
STATIC_IMAGE_PATH = "data/test_image.jpg"

# Initialize retina model
retina = Retina(
    spike_threshold=10,
    fovea_radius_ratio=0.25,
    motion_persistence=0.2
)

# Visualization modes
MODES = {
    '0': 'raw',
    '1': 'rods',
    '2': 'cones',
    '3': 'horiz_rods',
    '4': 'horiz_cones',
    '5': 'bipolar_on',
    '6': 'bipolar_off',
    '7': 'amacrine_on',
    '8': 'amacrine_off',
    '9': 'ganglion_on',
    'a': 'ganglion_off'
}

current_mode = '0'
paused = False

# Initialize video capture
if USE_WEBCAM:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        exit(1)
    print("Using webcam input")
else:
    frame_static = cv2.imread(STATIC_IMAGE_PATH)
    if frame_static is None:
        print(f"Error: Could not load image {STATIC_IMAGE_PATH}")
        exit(1)
    frame_static = cv2.resize(frame_static, (640, 480))
    print(f"Using static image: {STATIC_IMAGE_PATH}")

print("\nRetina Visualization Demo")
print("=" * 50)
print("Keyboard controls:")
print("  0 - Raw input")
print("  1 - Rods (grayscale, blurred)")
print("  2 - Cones (color, foveated)")
print("  3 - Horizontal cells (rods)")
print("  4 - Horizontal cells (cones)")
print("  5 - Bipolar cells (ON channel)")
print("  6 - Bipolar cells (OFF channel)")
print("  7 - Amacrine cells (motion ON)")
print("  8 - Amacrine cells (motion OFF)")
print("  9 - Ganglion cells (spikes ON)")
print("  A - Ganglion cells (spikes OFF)")
print("  P - Pause/unpause")
print("  Q - Quit")
print("=" * 50)

while True:
    if not paused:
        # Get frame
        if USE_WEBCAM:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            frame = cv2.resize(frame, (640, 480))
        else:
            frame = frame_static.copy()

        # Process through retina
        outputs = retina.forward(frame)

        # Select visualization based on current mode
        mode = MODES.get(current_mode, 'raw')

        if mode == 'raw':
            display = frame.copy()
        elif mode == 'rods':
            display = cv2.cvtColor(outputs['rods_out'], cv2.COLOR_GRAY2BGR)
        elif mode == 'cones':
            display = cv2.cvtColor(outputs['cones_out'], cv2.COLOR_HSV2BGR)
        elif mode == 'horiz_rods':
            display = cv2.cvtColor(outputs['rods_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == 'horiz_cones':
            display = cv2.cvtColor(outputs['cones_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == 'bipolar_on':
            display = cv2.cvtColor(outputs['on_channel'], cv2.COLOR_GRAY2BGR)
        elif mode == 'bipolar_off':
            display = cv2.cvtColor(outputs['off_channel'], cv2.COLOR_GRAY2BGR)
        elif mode == 'amacrine_on':
            display = cv2.cvtColor(outputs['motion_on'], cv2.COLOR_GRAY2BGR)
        elif mode == 'amacrine_off':
            display = cv2.cvtColor(outputs['motion_off'], cv2.COLOR_GRAY2BGR)
        elif mode == 'ganglion_on':
            display = cv2.cvtColor(outputs['on_spikes'], cv2.COLOR_GRAY2BGR)
        elif mode == 'ganglion_off':
            display = cv2.cvtColor(outputs['off_spikes'], cv2.COLOR_GRAY2BGR)
        else:
            display = frame.copy()

        # Add mode label
        cv2.putText(display, f"Mode: {mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show display
        cv2.imshow("Retina Processing", display)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
        status = "PAUSED" if paused else "RUNNING"
        print(f"Status: {status}")
    elif chr(key) in MODES:
        current_mode = chr(key)
        mode_name = MODES[current_mode]
        print(f"Switched to mode: {mode_name}")

# Cleanup
if USE_WEBCAM:
    cap.release()
cv2.destroyAllWindows()
print("\nDemo ended.")

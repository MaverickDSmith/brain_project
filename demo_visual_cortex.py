"""
Visual Cortex Complete Demo
============================
Demonstrates the full visual processing pipeline from retina through cortex.

Features:
- Real-time webcam or static image input
- Complete retina → V1 → V2 → Dorsal/Ventral processing
- Multiple visualization modes
- Statistics and logging
- Object recognition display
- Motion and attention visualization

Controls:
- Number keys (0-9): Switch visualization modes
- P: Pause/unpause
- R: Reset all states
- L: Toggle statistics logging
- Q: Quit
"""

import cv2
import numpy as np
import torch
import time
import sys

from visual.visual_cortex import VisualCortex
from visual.color_learning import ColorLearningSNN, build_color_wheel

# ==================== Configuration ====================
USE_WEBCAM = True
VIDEO_SOURCE = 0
STATIC_IMAGE_PATH = "data/test_image.jpg"

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# ==================== Initialize Visual Cortex ====================
print("\nInitializing Visual Cortex...")
print("This may take a moment...")

visual_cortex = VisualCortex(device=DEVICE, spike_threshold=10)

# Initialize color learning SNN
print("Initializing Color Learning SNN...")
color_snn = ColorLearningSNN(num_hue_bins=12, num_neurons=64, device=DEVICE)

print("Visual Cortex initialized!")
print("\nArchitecture:")
print("  Retina → V1 → V2 → {Dorsal (WHERE/HOW), Ventral (WHAT)}")
print("\nDorsal Stream: V3 → V5/MT → Parietal Cortex")
print("Ventral Stream: V4 → IT (Inferotemporal Cortex)")
print("\nColor Learning: SNN with STDP (64 neurons, 12 hue bins)")

# ==================== Visualization Modes ====================
VISUALIZATION_MODES = {
    '0': 'summary',           # Multi-panel summary
    '1': 'retina_motion',     # Retina motion detection
    '2': 'v1_orientations',   # V1 orientation map
    '3': 'v2_features',       # V2 combined features
    '4': 'dorsal_motion',     # Dorsal stream motion field
    '5': 'dorsal_attention',  # Dorsal stream attention map
    '6': 'ventral_color',     # Ventral stream color features
    '7': 'ventral_shape',     # Ventral stream shape features
    '8': 'ventral_categories',# Ventral stream category activations
    '9': 'original'           # Original input
}

current_mode = '0'
paused = False
log_statistics = False

# Color learning state
clicked_hsv = None
color_test_results = None
color_wheel_bgr, color_wheel_hsv = build_color_wheel(size=256)

# ==================== Input Setup ====================
if USE_WEBCAM:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        sys.exit(1)
    print(f"\nUsing webcam: {VIDEO_SOURCE}")
else:
    frame_static = cv2.imread(STATIC_IMAGE_PATH)
    if frame_static is None:
        print(f"Error: Could not load image {STATIC_IMAGE_PATH}")
        sys.exit(1)
    frame_static = cv2.resize(frame_static, (640, 480))
    print(f"\nUsing static image: {STATIC_IMAGE_PATH}")

# ==================== UI Info ====================
print("\n" + "=" * 60)
print("VISUAL CORTEX DEMO - CONTROLS")
print("=" * 60)
print("Visualization Modes:")
print("  0 - Summary (multi-panel view)")
print("  1 - Retina motion detection")
print("  2 - V1 orientation selectivity")
print("  3 - V2 contour/texture features")
print("  4 - Dorsal: Motion field")
print("  5 - Dorsal: Spatial attention")
print("  6 - Ventral: Color processing (V4)")
print("  7 - Ventral: Shape processing (V4)")
print("  8 - Ventral: Object categories (IT)")
print("  9 - Original input")
print("\nControls:")
print("  P - Pause/Unpause")
print("  R - Reset all states")
print("  L - Toggle statistics logging")
print("  C - Toggle color learning")
print("  Q - Quit")
print("\nColor Learning:")
print("  - Click on color wheel to test classification")
print("  - Neurons learn colors from video stream")
print("  - Green border = locked/learned neuron")
print("=" * 60 + "\n")

# ==================== Performance Tracking ====================
frame_times = []
processing_times = []

# ==================== Color Learning Helper Functions ====================
def draw_neuron_color_map(color_snn, width=512, height=256, cell_size=40, padding=2):
    """Draw a grid showing each neuron's preferred color and confidence."""
    neuron_colors_confidences = color_snn.get_neuron_colors()
    locked_status = color_snn.neuron_locked.cpu().numpy()

    neurons_per_row = max(1, width // (cell_size + padding))
    n_rows = (color_snn.num_neurons + neurons_per_row - 1) // neurons_per_row

    actual_width = neurons_per_row * (cell_size + padding) - padding
    actual_height = n_rows * (cell_size + padding) - padding

    neuron_img = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)

    for i, (color, conf) in enumerate(neuron_colors_confidences):
        row = i // neurons_per_row
        col = i % neurons_per_row

        x0 = col * (cell_size + padding)
        y0 = row * (cell_size + padding)
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        # Draw neuron cell
        cv2.rectangle(neuron_img, (x0, y0), (x1, y1), color, -1)

        # Border: green if locked, gray otherwise
        border_color = (0, 255, 0) if locked_status[i] else (64, 64, 64)
        border_thickness = 2 if locked_status[i] else 1
        cv2.rectangle(neuron_img, (x0, y0), (x1, y1), border_color, border_thickness)

        # Show confidence
        text = f"{conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        text_x = x0 + (cell_size - text_size[0]) // 2
        text_y = y0 + cell_size // 2 + text_size[1] // 2

        cv2.putText(neuron_img, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2)
        cv2.putText(neuron_img, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return neuron_img


def on_color_wheel_click(event, x, y, flags, param):
    """Handle clicks on the color wheel"""
    global clicked_hsv, color_test_results
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get HSV value at clicked position
        clicked_hsv = color_wheel_hsv[y, x]
        h, s, v = clicked_hsv

        # Extract features for this color
        test_hsv = np.full((10, 10, 3), clicked_hsv, dtype=np.uint8)
        features = color_snn.extract_color_features(test_hsv)

        # Classify
        color_test_results = color_snn.classify_color(features, top_k=3)

        print(f"\nColor clicked: H={h}, S={s}, V={v}")
        print(f"Top 3 neuron responses:")
        for idx, activation, color in zip(
                color_test_results['neuron_indices'],
                color_test_results['activations'],
                color_test_results['preferred_colors']):
            print(f"  Neuron {idx}: activation={activation:.3f}, color={color}")


# ==================== Info Panel ====================
def create_info_panel(summary, width=400, height=300):
    """Create an info panel showing current perception"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    y_offset = 30
    line_height = 25

    # Frame info
    cv2.putText(panel, f"Frame: {summary['frame']}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += line_height

    # Motion info
    cv2.putText(panel, "=== MOTION (Dorsal) ===", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += line_height

    cv2.putText(panel, f"Direction: {summary['motion']['dominant_direction']}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height

    cv2.putText(panel, f"Attention: {summary['spatial_attention_peak']:.3f}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += line_height + 10

    # Object info
    cv2.putText(panel, "=== OBJECTS (Ventral) ===", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_offset += line_height

    for i, (obj, conf) in enumerate(zip(summary['objects']['detected_objects'][:5],
                                         summary['objects']['confidence'][:5])):
        text = f"{i+1}. {obj}: {conf:.2f}"
        cv2.putText(panel, text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += line_height

    # V1 activity
    y_offset += 10
    cv2.putText(panel, f"V1 Spikes: {summary['v1_spikes']:.1f}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return panel

# ==================== Setup Mouse Callback ====================
cv2.namedWindow("Color Wheel (click to test)")
cv2.setMouseCallback("Color Wheel (click to test)", on_color_wheel_click)

# ==================== Main Loop ====================
print("Starting main loop...\n")

try:
    while True:
        loop_start = time.time()

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

            # Process through visual cortex
            process_start = time.time()
            outputs = visual_cortex.forward(frame)
            process_time = time.time() - process_start
            processing_times.append(process_time)

            # Color learning: extract features and learn
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_features = color_snn.extract_color_features(frame_hsv)
            color_spikes, color_mem = color_snn.forward(color_features)

            # Learn from the current frame
            if color_snn.learning_enabled:
                color_snn.learn(torch.from_numpy(color_features).to(DEVICE),
                               color_spikes)

            # Get visualizations
            visualizations = visual_cortex.visualize_complete_pipeline(
                frame, outputs, output_size=(640, 480))

            # Get summary
            summary = visual_cortex.get_visual_summary(outputs)

            # Log statistics if enabled
            if log_statistics and visual_cortex.frame_count % 30 == 0:
                visual_cortex.log_statistics(outputs)

            # Select display based on mode
            mode = VISUALIZATION_MODES.get(current_mode, 'summary')

            if mode == 'summary':
                display = visual_cortex.create_summary_display(frame, outputs, visualizations)
            elif mode == 'original':
                display = frame.copy()
            elif mode == 'retina_motion':
                display = visualizations['retina_motion']
            elif mode == 'v1_orientations':
                display = visualizations['v1_orientations']
            elif mode == 'v2_features':
                display = visualizations['v2_features']
            elif mode == 'dorsal_motion':
                display = visualizations['dorsal_motion_field']
            elif mode == 'dorsal_attention':
                display = visualizations['dorsal_attention']
            elif mode == 'ventral_color':
                display = visualizations['ventral_color']
            elif mode == 'ventral_shape':
                display = visualizations['ventral_shape']
            elif mode == 'ventral_categories':
                display = visualizations['ventral_categories']
            else:
                display = frame.copy()

            # Add mode label (except for summary which has its own labels)
            if mode != 'summary':
                cv2.putText(display, f"Mode: {mode.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display, f"Frame: {visual_cortex.frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Create info panel
            info_panel = create_info_panel(summary)

            # Show displays
            cv2.imshow("Visual Cortex", display)
            cv2.imshow("Perception Info", info_panel)

            # Show color wheel and neuron map
            color_wheel_display = color_wheel_bgr.copy()

            # If color test results exist, overlay them
            if color_test_results is not None:
                # Draw top predicted colors
                patch_size = 30
                x_start, y_start = 10, 10
                for i, (color, activation) in enumerate(zip(
                        color_test_results['preferred_colors'],
                        color_test_results['activations'])):
                    x0 = x_start
                    y0 = y_start + i * (patch_size + 5)
                    x1, y1 = x0 + patch_size, y0 + patch_size
                    cv2.rectangle(color_wheel_display, (x0, y0), (x1, y1), color, -1)
                    cv2.rectangle(color_wheel_display, (x0, y0), (x1, y1), (255, 255, 255), 2)

                    # Show activation strength
                    text = f"{activation:.2f}"
                    cv2.putText(color_wheel_display, text, (x1 + 5, y0 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show learning status
            learning_status = "ON" if color_snn.learning_enabled else "OFF"
            n_locked = torch.sum(color_snn.neuron_locked).item()
            cv2.putText(color_wheel_display, f"Learning: {learning_status}", (10, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(color_wheel_display, f"Locked: {n_locked}/{color_snn.num_neurons}",
                       (10, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imshow("Color Wheel (click to test)", color_wheel_display)

            # Draw neuron color map
            neuron_map = draw_neuron_color_map(color_snn, width=512, height=256)
            cv2.imshow("Color Neurons", neuron_map)

            # Periodic GPU cache clearing to prevent memory fragmentation
            if DEVICE == 'cuda' and visual_cortex.frame_count % 30 == 0:
                torch.cuda.empty_cache()

        # Calculate FPS
        loop_time = time.time() - loop_start
        frame_times.append(loop_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
            processing_times.pop(0)

        avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        avg_process_time = (sum(processing_times) / len(processing_times)) if processing_times else 0

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSED" if paused else "RUNNING"
            print(f"\nStatus: {status}")
        elif key == ord('r'):
            print("\nResetting visual cortex and color learning states...")
            visual_cortex.reset_states()
            color_snn.reset_states()
            color_test_results = None
            print("Reset complete!")
        elif key == ord('l'):
            log_statistics = not log_statistics
            status = "ENABLED" if log_statistics else "DISABLED"
            print(f"\nStatistics logging: {status}")
        elif key == ord('c'):
            color_snn.learning_enabled = not color_snn.learning_enabled
            status = "ENABLED" if color_snn.learning_enabled else "DISABLED"
            print(f"\nColor learning: {status}")
        elif chr(key) in VISUALIZATION_MODES:
            current_mode = chr(key)
            mode_name = VISUALIZATION_MODES[current_mode]
            print(f"\nSwitched to mode: {mode_name}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")

except Exception as e:
    print(f"\n\nError occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print("\nCleaning up...")
    if USE_WEBCAM:
        cap.release()
    cv2.destroyAllWindows()

    # Print final statistics
    print("\n" + "=" * 60)
    print("SESSION STATISTICS")
    print("=" * 60)
    print(f"Total frames processed: {visual_cortex.frame_count}")
    if processing_times:
        avg_process_time = (sum(processing_times) / len(processing_times)) if processing_times else 0
        avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        print(f"Average processing time: {avg_process_time*1000:.2f} ms/frame")
        print(f"Average FPS: {avg_fps:.2f}")
    print("=" * 60)
    print("\nDemo ended.")

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from visual.retina import Retina

# -------------------- Config --------------------
USE_VIDEO = True
VIDEO_SOURCE = 0
STATIC_IMAGE_PATH = "data/looserotsnakes2.jpg"
paused = False
overlay_colors = None
clicked_hsv = None
clicked_hsv_hue = None

# -------------------- Initialize Retina --------------------
retina = Retina(
    spike_threshold=40,
    fovea_radius_ratio=0.25,
    num_hue_bins=22,
    n_neurons=44,
    temporal_window=4,
    ds_w=40,
    ds_h=30,
    learning_lr=0.3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    motion_persistence=0.2
)

# -------------------- Input setup --------------------
if USE_VIDEO:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {VIDEO_SOURCE}")
else:
    frame_static = cv2.imread(STATIC_IMAGE_PATH)
    if frame_static is None:
        raise RuntimeError(f"Could not load image {STATIC_IMAGE_PATH}")
    frame_static = cv2.resize(frame_static, (640, 480))

# -------------------- Color Wheel --------------------
COLOR_WHEEL_SIZE = 256
def build_color_wheel(size=COLOR_WHEEL_SIZE):
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r_max = size // 2
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)
            if r > r_max:
                hsv[y,x,1] = 0
                hsv[y,x,2] = 255
                hsv[y,x,0] = 0
            else:
                angle = (np.arctan2(dy, dx) + np.pi) / (2*np.pi)
                hsv[y,x,0] = int(angle * 180)
                hsv[y,x,1] = int(r / r_max * 255)
                hsv[y,x,2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, hsv

color_wheel_bgr, color_wheel_hsv = build_color_wheel()

def on_color_click(event, x, y, flags, param):
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_hsv = color_wheel_hsv[y, x]

cv2.namedWindow("Color Wheel")
cv2.setMouseCallback("Color Wheel", on_color_click)

# -------------------- Neuron Map --------------------
def draw_neuron_map(retina, width=600, cell_size=50, padding=2):
    """
    Draw a grid showing each neuron's preferred color and confidence.
    Automatically wraps to multiple rows when needed.
    Locked neurons are indicated with a border.
    
    Args:
        retina: Retina instance
        width: Maximum width of the display
        cell_size: Size of each neuron cell (square)
        padding: Padding between cells
    """
    weights = retina.snn.fc.weight.detach().cpu().numpy()  # (NUM_NEURONS, NUM_HUE_BINS)
    neuron_colors = []
    confidences = []
    
    for neuron_idx in range(weights.shape[0]):
        w = weights[neuron_idx]
        pref_bin = np.argmax(w)
        hue = int((pref_bin + 0.5) * (180.0 / retina.NUM_HUE_BINS)) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
        neuron_colors.append(tuple(int(c) for c in bgr))
        confidences.append(np.max(w))
    
    n_neurons = weights.shape[0]
    locked_status = retina.neuron_locked.cpu().numpy()
    
    # Calculate grid dimensions
    neurons_per_row = max(1, width // (cell_size + padding))
    n_rows = (n_neurons + neurons_per_row - 1) // neurons_per_row  # Ceiling division
    
    # Calculate actual dimensions
    actual_width = neurons_per_row * (cell_size + padding) - padding
    actual_height = n_rows * (cell_size + padding) - padding
    
    neuron_img = np.zeros((actual_height, actual_width, 3), dtype=np.uint8)
    
    for i, (color, conf) in enumerate(zip(neuron_colors, confidences)):
        row = i // neurons_per_row
        col = i % neurons_per_row
        
        x0 = col * (cell_size + padding)
        y0 = row * (cell_size + padding)
        x1 = x0 + cell_size
        y1 = y0 + cell_size
        
        # Draw colored rectangle
        cv2.rectangle(neuron_img, (x0, y0), (x1, y1), color, -1)
        
        # Draw border - thick green if locked, thin black if not
        if locked_status[i]:
            cv2.rectangle(neuron_img, (x0, y0), (x1, y1), (0, 255, 0), 3)
        else:
            cv2.rectangle(neuron_img, (x0, y0), (x1, y1), (0, 0, 0), 1)
        
        # Draw confidence text
        text = f"{conf:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = x0 + (cell_size - text_size[0]) // 2
        text_y = y0 + cell_size // 2 + text_size[1] // 2
        
        # Draw text with black outline for visibility
        cv2.putText(neuron_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(neuron_img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return neuron_img

# -------------------- Mode Map --------------------
mode_map = {
    "0": "raw",
    "1": "rods",
    "2": "cones",
    "3": "horiz_rods",
    "4": "horiz_cones",
    "5": "bipolar_on",
    "6": "bipolar_off",
    "7": "amacrine_on",
    "8": "amacrine_off",
    "9": "ganglion_on",
    "a": "ganglion_off",
    "s": "dir_horiz",
    "d": "dir_vert",
    "b": "belief"
}
current_mode = "0"

# -------------------- Helper --------------------
def test_snn_on_hue(retina, hsv_pixel, top_k=3):
    h = int(hsv_pixel[0])
    bin_idx = int(h / (180.0 / retina.NUM_HUE_BINS))
    bin_idx = np.clip(bin_idx, 0, retina.NUM_HUE_BINS-1)
    onehot = np.zeros((1, retina.NUM_HUE_BINS), dtype=np.float32)
    onehot[0, bin_idx] = 1.0
    x_t = torch.from_numpy(onehot.astype(np.float32)).to(retina.DEVICE)

    with torch.no_grad():
        weights_np = retina.snn.fc.weight.detach().cpu().numpy()
        activ = x_t.cpu().numpy() @ weights_np.T
        activ = activ[0]
        top_idxs = np.argsort(-activ)[:top_k]
        pref_bins = np.argmax(weights_np[top_idxs], axis=1)
        pref_colors = [retina.hue_bin_to_bgr(b) for b in pref_bins]
    return top_idxs, activ[top_idxs], pref_colors



# -------------------- Main Loop --------------------
while True:
    if not paused:
        if USE_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            repeat_static = False
        else:
            frame = frame_static.copy()
            repeat_static = True

        # Run forward pass ONCE - don't call it again in visualize
        outputs = retina.forward(frame, repeat_static=repeat_static)

        # Manually build visualization from outputs (don't call visualize which would call forward again)
        mode = mode_map[current_mode]
        
        if mode == "raw":
            output_img = frame.copy()
        elif mode == "rods":
            output_img = cv2.cvtColor(outputs['rods_out'], cv2.COLOR_GRAY2BGR)
        elif mode == "cones":
            output_img = cv2.cvtColor(outputs['cones_out'], cv2.COLOR_HSV2BGR)
        elif mode == "horiz_rods":
            output_img = cv2.cvtColor(outputs['rods_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == "horiz_cones":
            output_img = cv2.cvtColor(outputs['cones_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == "bipolar_on":
            output_img = cv2.cvtColor(outputs['on_chan'], cv2.COLOR_GRAY2BGR)
        elif mode == "bipolar_off":
            output_img = cv2.cvtColor(outputs['off_chan'], cv2.COLOR_GRAY2BGR)
        elif mode == "amacrine_on":
            output_img = cv2.cvtColor(outputs['motion_on'], cv2.COLOR_GRAY2BGR)
        elif mode == "amacrine_off":
            output_img = cv2.cvtColor(outputs['motion_off'], cv2.COLOR_GRAY2BGR)
        elif mode == "ganglion_on":
            output_img = cv2.cvtColor(outputs['on_spikes'], cv2.COLOR_GRAY2BGR)
        elif mode == "ganglion_off":
            output_img = cv2.cvtColor(outputs['off_spikes'], cv2.COLOR_GRAY2BGR)
        elif mode == "dir_horiz":
            output_img = cv2.cvtColor(outputs['horiz_dir'], cv2.COLOR_GRAY2BGR)
        elif mode == "dir_vert":
            output_img = cv2.cvtColor(outputs['vert_dir'], cv2.COLOR_GRAY2BGR)
        elif mode == "belief":
            weights = retina.snn.fc.weight.detach().cpu().numpy()
            pref_bins = np.argmax(weights, axis=1)
            assigned_map = outputs['belief']
            belief_hsv = np.zeros((retina.DS_H, retina.DS_W, 3), dtype=np.uint8)
            for y in range(retina.DS_H):
                for x in range(retina.DS_W):
                    neuron_idx = int(assigned_map[y,x])
                    bin_idx = int(pref_bins[neuron_idx])
                    hue = int((bin_idx + 0.5) * (180.0 / retina.NUM_HUE_BINS)) % 180
                    belief_hsv[y,x,0] = hue
                    belief_hsv[y,x,1] = 255
                    belief_hsv[y,x,2] = 255
            output_img = cv2.cvtColor(cv2.resize(belief_hsv, (frame.shape[1], frame.shape[0]),
                                                interpolation=cv2.INTER_NEAREST), cv2.COLOR_HSV2BGR)
        else:
            output_img = frame.copy()

        # Add overlay colors if present
        if overlay_colors is not None:
            patch_h, patch_w = 40, 40
            x_start, y_start = 10, 10
            for i, col in enumerate(overlay_colors):
                x0 = x_start + i*(patch_w + 5)
                y0 = y_start
                x1, y1 = x0 + patch_w, y0 + patch_h
                cv2.rectangle(output_img, (x0,y0), (x1,y1), col, -1)
                cv2.rectangle(output_img, (x0,y0), (x1,y1), (0,0,0), 1)
            cv2.putText(output_img, "Top neuron colors", (x_start, y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # Overlay mode text
        cv2.putText(output_img, f"Mode: {mode}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------------- Handle color wheel clicks ----------------
        if clicked_hsv is not None:
            neuron_idxs, activations, pref_colors = test_snn_on_hue(retina, clicked_hsv, top_k=3)
            clicked_hsv_hue = int(clicked_hsv[0])
            clicked_hsv = None
            overlay_colors = pref_colors
            print("Top neuron responses:")
            for idx, act, col in zip(neuron_idxs, activations, pref_colors):
                print(f"  Neuron {idx}, activation {act:.3f}, color {col}")

            # Highlight pixels in belief map assigned to these neurons
            belief_map = outputs['belief']
            overlay_img = output_img.copy()
            for neuron_idx, color in zip(neuron_idxs, pref_colors):
                mask = (belief_map == neuron_idx)
                mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                color_layer = np.zeros_like(output_img, dtype=np.uint8)
                color_layer[:, :] = color
                overlay_img = cv2.addWeighted(overlay_img,
                                              1.0,
                                              cv2.bitwise_and(color_layer, color_layer, mask=mask_resized),
                                              0.5, 0)
            output_img = overlay_img

        # Show outputs
        cv2.imshow("Retina Output", output_img)
        cv2.imshow("Color Wheel", color_wheel_bgr)
        neuron_map_img = draw_neuron_map(retina)
        cv2.imshow("Neuron Map", neuron_map_img)

    # ---------------- Key Handling ----------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused
    elif key == ord("r"):
        # Reset temporal buffer and SNN state
        retina.temporal_buffer = [np.zeros((retina.DS_H, retina.DS_W, retina.NUM_HUE_BINS), dtype=np.uint8)
                                  for _ in range(retina.TEMPORAL_WINDOW)]
        retina.snn_state = retina.snn.init_state(retina.BATCH_PIXELS)
        retina.prev_bipolar_on = None
        retina.prev_bipolar_off = None
        retina.prev_frame_gray = None
        retina.motion_on_accumulator = None
        retina.motion_off_accumulator = None
        retina.neuron_locked = torch.zeros(retina.N_NEURONS, dtype=torch.bool, device=retina.DEVICE)
        overlay_colors = None
        print("Retina state reset.")
    elif key in range(48, 58):  # 0-9
        current_mode = chr(key)
    elif key in [ord('a'), ord('s'), ord('d'), ord('b')]:
        current_mode = chr(key)

# ---------------- Cleanup ----------------
if USE_VIDEO:
    cap.release()
cv2.destroyAllWindows()
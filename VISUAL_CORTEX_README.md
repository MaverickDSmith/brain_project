# Visual Cortex Neural Model

A biologically-inspired spiking neural network model of the human visual system, from retina through cortical processing.

## Architecture Overview

```
Input (Camera/Image)
        ↓
    RETINA (retina_scratch.py)
    ├─ Rods (low-light, grayscale)
    ├─ Cones (color, foveal)
    ├─ Horizontal cells (lateral inhibition)
    ├─ Bipolar cells (ON/OFF channels)
    ├─ Amacrine cells (motion detection)
    └─ Ganglion cells (spike generation)
        ↓
    V1 - Primary Visual Cortex (v1.py)
    ├─ Simple cells (orientation-selective)
    ├─ Complex cells (position-invariant)
    └─ Gabor filtering
        ↓
    V2 - Secondary Visual Cortex (v2.py)
    ├─ Contour integration
    ├─ Texture processing
    └─ Figure-ground segmentation
        ↓
    ┌───────────────────┴───────────────────┐
    ↓                                       ↓
DORSAL STREAM                        VENTRAL STREAM
"WHERE/HOW"                          "WHAT"
(dorsal_stream.py)                   (ventral_stream.py)
    ↓                                       ↓
V3 - Dynamic Form                    V4 - Color & Form
    ↓                                ├─ Color processing
V5/MT - Motion                       └─ Shape processing
├─ Direction selectivity                   ↓
├─ Speed tuning                      IT - Object Recognition
└─ Opponent motion                   ├─ Category neurons
    ↓                                ├─ Object embedding
Parietal Cortex                      └─ Invariant representation
├─ Spatial attention
├─ Action affordances
└─ Visuomotor integration
```

## Files Structure

### Core Modules

- **`visual/retina_scratch.py`**: Retinal processing (photoreceptors → ganglion cells)
- **`visual/v1.py`**: Primary visual cortex (orientation/direction selectivity)
- **`visual/v2.py`**: Secondary visual cortex (contours, textures)
- **`visual/dorsal_stream.py`**: Spatial processing pathway (V3 → V5/MT → Parietal)
- **`visual/ventral_stream.py`**: Object recognition pathway (V4 → IT)
- **`visual/visual_cortex.py`**: Complete integration module

### Demo Scripts

- **`demo_retina_scratch.py`**: Visualize retinal processing stages
- **`demo_visual_cortex.py`**: Complete visual cortex pipeline demo

## Installation

```bash
# Install dependencies
pip install -r requirements_visual.txt

# For GPU support (CUDA), install PyTorch with CUDA:
# Visit: https://pytorch.org/get-started/locally/
```

## Quick Start

### 1. Test Retina Processing

```bash
python demo_retina_scratch.py
```

**Controls:**
- `0-9, A`: Switch between visualization modes
- `P`: Pause/unpause
- `Q`: Quit

**Modes:**
- `0`: Raw input
- `1`: Rods (grayscale, blurred)
- `2`: Cones (color, foveal)
- `3`: Horizontal cells (rods)
- `5-6`: Bipolar cells (ON/OFF)
- `7-8`: Amacrine cells (motion)
- `9-A`: Ganglion cells (spikes)

### 2. Test Complete Visual Cortex

```bash
python demo_visual_cortex.py
```

**Controls:**
- `0-9`: Switch visualization modes
- `P`: Pause/unpause
- `R`: Reset all states
- `L`: Toggle statistics logging
- `Q`: Quit

**Visualization Modes:**
- `0`: Summary (multi-panel view)
- `1`: Retina motion detection
- `2`: V1 orientation selectivity
- `3`: V2 contour/texture features
- `4`: Dorsal stream motion field
- `5`: Dorsal stream spatial attention
- `6`: Ventral stream color processing
- `7`: Ventral stream shape processing
- `8`: Ventral stream object categories
- `9`: Original input

## Usage in Code

### Basic Usage

```python
from visual.visual_cortex import VisualCortex
import cv2

# Initialize
visual_cortex = VisualCortex(device='cuda', spike_threshold=10)

# Process frame
frame = cv2.imread('image.jpg')
outputs = visual_cortex.forward(frame)

# Get high-level summary
summary = visual_cortex.get_visual_summary(outputs)
print(f"Motion: {summary['motion']['dominant_direction']}")
print(f"Objects: {summary['objects']['detected_objects'][:3]}")

# Get action affordances (from parietal cortex)
affordances = visual_cortex.get_action_affordances(outputs)

# Get object embedding (from IT cortex)
embedding = visual_cortex.get_object_embedding(outputs)
```

### Access Specific Outputs

```python
# Retina outputs
retina_out = outputs['retina']
on_spikes = retina_out['on_spikes']
off_spikes = retina_out['off_spikes']
motion = retina_out['motion_on']

# V1 outputs
v1_out = outputs['v1']
orientation_map = v1_out['orientation_map']
simple_spikes = v1_out['simple_spikes']

# Dorsal stream (motion, spatial)
dorsal_out = outputs['dorsal']
motion_field = dorsal_out['mt']['mt_direction_spikes']
attention_map = dorsal_out['parietal']['attention_map']
action_vector = dorsal_out['parietal']['action_vector']

# Ventral stream (objects)
ventral_out = outputs['ventral']
color_features = ventral_out['v4']['v4_color_spikes']
object_categories = ventral_out['it']['it_category_activations']
object_embedding = ventral_out['it']['it_object_embedding']
```

## Biological Accuracy

### Retina
- **Rods**: Low-light sensitivity, peripheral vision, grayscale
- **Cones**: Color vision, high acuity in fovea
- **Horizontal cells**: Lateral inhibition via gap junctions
- **Bipolar cells**: ON/OFF center-surround receptive fields
- **Amacrine cells**: Temporal processing and motion detection
- **Ganglion cells**: Spike generation with wide-field inhibition

### V1 (Primary Visual Cortex)
- **Simple cells**: Position-specific orientation selectivity
- **Complex cells**: Position-invariant orientation detection
- **Gabor filters**: Model of receptive field structure
- **Spiking dynamics**: LIF (Leaky Integrate-and-Fire) neurons

### V2 (Secondary Visual Cortex)
- **Contour integration**: Long-range horizontal connections
- **Texture processing**: Multi-scale analysis
- **Border ownership**: Figure-ground segmentation

### Dorsal Stream ("WHERE/HOW")
- **V3**: Dynamic form, shape-from-motion
- **V5/MT**: Direction selectivity, speed tuning, opponent motion
- **Parietal**: Spatial attention, action affordances, visuomotor integration

### Ventral Stream ("WHAT")
- **V4**: Color combinations, complex shapes, curvature
- **IT**: Object-selective neurons, invariant representations, category organization

## SNN Backend

The implementation supports two SNN backends:

1. **snntorch** (default, recommended)
   - Easy to use
   - Good documentation
   - Active development

2. **norse**
   - Alternative backend
   - Can be faster for some operations

The code automatically detects which backend is installed and uses it.

## Performance

### CPU
- ~5-10 FPS on modern CPU
- Good for testing and development

### GPU (CUDA)
- ~20-60 FPS depending on GPU
- Recommended for real-time applications

### Memory
- ~1-2 GB GPU memory for default settings
- Can be reduced by lowering network sizes

## Customization

### Adjust Network Sizes

```python
# In visual_cortex.py __init__
self.v1 = V1(
    simple_hidden=64,  # Reduce from 128
    complex_size=32,   # Reduce from 64
    ...
)
```

### Change Spike Threshold

```python
visual_cortex = VisualCortex(
    spike_threshold=20  # Higher = fewer spikes
)
```

### Modify Retina Parameters

```python
from visual.retina_scratch import Retina

retina = Retina(
    spike_threshold=15,
    fovea_radius_ratio=0.3,  # Larger fovea
    motion_persistence=0.5   # More persistent motion
)
```

## Output for Downstream Processing

The visual cortex provides several outputs suitable for downstream brain areas:

### For Motor Control
```python
action_affordances = visual_cortex.get_action_affordances(outputs)
action_vector = action_affordances['action_vector']  # (512,) vector
```

### For Memory/Recognition
```python
object_embedding = visual_cortex.get_object_embedding(outputs)  # (512,) invariant representation
```

### For Attention/Saliency
```python
attention_map = outputs['dorsal']['parietal']['attention_map']  # (H, W) spatial attention
```

### For Decision Making
```python
summary = visual_cortex.get_visual_summary(outputs)
# Contains: motion direction, detected objects, confidence levels
```

## Future Extensions

### Planned Features
- [ ] Binocular vision and depth perception
- [ ] Eye movement control (saccades)
- [ ] Visual working memory
- [ ] Top-down attention modulation
- [ ] Face-selective neurons (FFA)
- [ ] Place cells integration
- [ ] Predictive coding framework

### Integration Points
- **Prefrontal cortex**: For attention control and working memory
- **Motor cortex**: For visuomotor transformations
- **Hippocampus**: For spatial navigation and visual memory
- **Auditory system**: For multisensory integration

## References

This implementation is inspired by:
- Hubel & Wiesel's work on V1
- Ungerleider & Mishkin's two-stream hypothesis
- DiCarlo & Cox's work on IT cortex
- Goodale & Milner's dorsal/ventral stream functions

## Troubleshooting

### "No SNN library found"
Install snntorch: `pip install snntorch`

### CUDA out of memory
Reduce network sizes or batch size, or use CPU

### Slow performance
- Use GPU if available
- Reduce image resolution
- Decrease network complexity

## License

This is a research/educational implementation. Feel free to use and modify.

## Contributing

This is part of a larger brain modeling project. Contributions welcome!

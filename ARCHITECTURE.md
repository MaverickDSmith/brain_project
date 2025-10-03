# Visual Cortex Architecture Documentation

## System Overview

This implementation models the human visual system from the retina through cortical processing, using biologically-inspired spiking neural networks.

## Layer-by-Layer Breakdown

### 1. RETINA (`retina_scratch.py`)

**Purpose**: Convert light into neural spike signals

**Processing Flow**:
```
Input Image (BGR)
    ↓
[Rods] ──────────→ Grayscale, low spatial frequency
    ↓
[Cones] ─────────→ Color (HSV), foveal enhancement
    ↓
[Horizontal] ────→ Lateral inhibition (center-surround)
    ↓
[Bipolar] ───────→ ON/OFF channels (contrast detection)
    ↓
[Amacrine] ──────→ Motion detection (temporal derivative)
    ↓
[Ganglion] ──────→ Spike generation (thresholding)
    ↓
Output: ON/OFF spike trains
```

**Key Features**:
- Foveal vision (sharp center, blurry periphery)
- ON/OFF pathways (light increment/decrement)
- Temporal motion detection
- Spiking output (binary signals)

**Biological Accuracy**:
- ~5 layers (actual retina has 5 main layers)
- Center-surround receptive fields
- Motion persistence
- Spike-based encoding

---

### 2. V1 - Primary Visual Cortex (`v1.py`)

**Purpose**: Detect oriented edges and motion direction

**Components**:

#### Simple Cells
- **Function**: Orientation-selective, position-specific
- **Implementation**: Convolutional SNNs with orientation-specific filters
- **Receptive Fields**: Modeled with Gabor filters
- **Output**: 8 orientation channels (0°, 22.5°, 45°, ..., 157.5°)

#### Complex Cells
- **Function**: Position-invariant orientation detection
- **Implementation**: Pooling over simple cells
- **Output**: Downsampled orientation features

**Processing Flow**:
```
Retina ON/OFF Spikes
    ↓
[Simple Cells] ──→ Orientation-selective spiking
    ↓             (8 orientations × spatial locations)
[Complex Cells] ─→ Position-invariant pooling
    ↓
Output: Orientation features + spike maps
```

**Biological Parallels**:
- Simple cells ≈ Orientation-selective neurons (Hubel & Wiesel)
- Complex cells ≈ Phase-invariant neurons
- Gabor filters ≈ Receptive field structure
- LIF neurons ≈ Cortical spiking dynamics

---

### 3. V2 - Secondary Visual Cortex (`v2.py`)

**Purpose**: Integrate edges into contours and detect textures

**Components**:

#### Contour Integration
- **Function**: Link local edges into global contours
- **Implementation**: Long-range horizontal connections
- **Method**: Convolutions with large receptive fields per orientation

#### Texture Processing
- **Function**: Detect surface patterns and textures
- **Implementation**: Multi-scale filtering (3, 5, 7 pixel kernels)
- **Output**: Texture features at multiple scales

**Processing Flow**:
```
V1 Simple + Complex Features
    ↓
[Contour Integration] ─→ Link collinear edges
    ↓
[Texture Processing] ──→ Multi-scale texture features
    ↓
[V2 Cells (SNN)] ──────→ Spiking integration
    ↓
Output: Integrated features for dorsal/ventral streams
```

**Biological Parallels**:
- Contour integration ≈ Association field connections
- Texture ≈ V2 surface property encoding
- Border ownership (implicit in contour integration)

---

### 4. DORSAL STREAM - "WHERE/HOW" Pathway (`dorsal_stream.py`)

**Purpose**: Process spatial location, motion, and action affordances

#### V3 - Dynamic Form
```
V2 Features
    ↓
[Motion-Sensitive Processing]
    ↓
[Temporal Integration] ─→ Compare with previous frame
    ↓
[Spiking Neurons]
    ↓
Output: Motion-enhanced features
```

**Function**: Shape-from-motion, global motion patterns

#### V5/MT - Middle Temporal Area
```
V3 Features
    ↓
[Direction Detectors] ──→ 8 direction channels
    ↓
[Opponent Motion] ──────→ Relative motion computation
    ↓
[Speed Tuning] ─────────→ Multi-scale motion (different strides)
    ↓
[Direction SNNs] ───────→ Spiking per direction
    ↓
Output: Motion field (direction + speed)
```

**Function**: Motion direction, speed, and optical flow

**Key Features**:
- Direction selectivity (8 directions)
- Speed tuning (slow/medium/fast)
- Opponent motion (relative movement)

#### Parietal Cortex
```
MT Motion Features
    ↓
[Spatial Attention] ────→ Where to look (attention map)
    ↓
[Action Encoder] ───────→ What actions are possible
    ↓
[Spatial Features] ─────→ Egocentric spatial representation
    ↓
Output: Attention map + action vector + spatial features
```

**Sub-regions modeled**:
- LIP-like: Attention and saliency
- AIP-like: Action affordances
- VIP-like: Near-body space (implicit)

**Outputs for downstream areas**:
- Attention map: (H, W) spatial attention
- Action vector: (512,) possible actions
- Spatial features: (B, C, H, W) spatial representation

---

### 5. VENTRAL STREAM - "WHAT" Pathway (`ventral_stream.py`)

**Purpose**: Recognize objects, shapes, and colors

#### V4 - Color and Form
```
V2 Features
    ↓
┌──────────────┴──────────────┐
↓                             ↓
[Color Pathway]          [Shape Pathway]
    ↓                         ↓
[Color SNNs]             [Shape SNNs]
    ↓                         ↓
└──────────────┬──────────────┘
               ↓
        [Integration]
               ↓
Output: Color + shape features
```

**Function**: Complex color combinations, curvature, form

**Key Features**:
- Separate color and shape pathways
- Spiking neurons for each modality
- Integration of color-shape features

#### IT - Inferotemporal Cortex
```
V4 Features
    ↓
[Hierarchical Processing] ──→ 2 conv layers with pooling
    ↓
[Category Neurons] ─────────→ 64 object categories
    ↓                         (each with own SNN)
[Global Pooling]
    ↓
[Object Embedding] ─────────→ 512-D invariant representation
    ↓
Output: Category activations + object embedding
```

**Function**: High-level object recognition, invariant representation

**Key Features**:
- 64 object category channels
- Columnar organization (category-specific neurons)
- Invariant representation (global pooling)
- Object embedding for memory/comparison

**Outputs for downstream areas**:
- Category activations: (64,) confidence per category
- Object embedding: (512,) invariant representation
- Category spikes: Spiking activity per category

---

## Information Flow Summary

### Forward Propagation
```
Camera/Image (640×480×3)
    ↓
Retina: ON/OFF spikes (480×640 each)
    ↓
V1: 8 orientation maps (480×640 each) → Complex features (64×240×320)
    ↓
V2: Integrated features (256×240×320)
    ↓
    ┌───────────────┴────────────────┐
    ↓                                ↓
Dorsal Stream                 Ventral Stream
    ↓                                ↓
V3 (256 features)             V4 (512 features)
    ↓                                ↓
MT (8 directions)             IT (64 categories)
    ↓                                ↓
Parietal:                     Object Embedding:
- Attention map (H×W)         - Category vector (64)
- Action vector (512)         - Embedding (512)
- Spatial features            - Recognition confidence
```

### Output Dimensions

| Stage | Output Name | Shape | Description |
|-------|-------------|-------|-------------|
| Retina | on_spikes | (H, W) | ON ganglion spikes |
| Retina | off_spikes | (H, W) | OFF ganglion spikes |
| V1 | orientation_map | (H, W) | Dominant orientation |
| V1 | complex_features | (B, 64, H/2, W/2) | Position-invariant |
| V2 | v2_spikes | (B, 256, H/2, W/2) | Integrated features |
| Dorsal | attention_map | (B, 1, H/4, W/4) | Spatial attention |
| Dorsal | action_vector | (B, 512) | Action affordances |
| Ventral | category_activations | (B, 64) | Object categories |
| Ventral | object_embedding | (B, 512) | Invariant representation |

---

## Spiking Neural Network Architecture

### LIF (Leaky Integrate-and-Fire) Neurons

Used throughout V1, V2, Dorsal, and Ventral streams.

**Dynamics**:
```
V(t+1) = β·V(t) + I(t)
S(t) = 1 if V(t) > θ else 0
V(t) = V(t) - θ·S(t)  (reset after spike)
```

Where:
- V(t): Membrane potential
- β: Decay rate (0.7-0.9)
- I(t): Input current
- θ: Threshold (typically 1.0)
- S(t): Spike output

**Implementation**:
- Backend: snntorch or norse
- Surrogate gradients: Fast sigmoid (for backprop through spikes)
- Time steps: 3-4 per frame (temporal window)

---

## Memory and State Management

### Temporal States Maintained

1. **Retina**:
   - Previous ON/OFF channels (for motion)
   - Motion accumulators (with persistence)

2. **V1**:
   - Simple cell membrane potentials (per orientation)

3. **V2**:
   - V2 cell membrane potentials

4. **Dorsal Stream**:
   - V3: Previous features (for temporal integration)
   - MT: Direction-specific membrane potentials
   - Parietal: Membrane potentials

5. **Ventral Stream**:
   - V4: Color, shape, combined membrane potentials
   - IT: Category-specific membrane potentials

### State Reset
Call `visual_cortex.reset_states()` to clear all temporal information.

---

## Computational Complexity

### Per-Frame Processing

| Stage | Operations | Relative Cost |
|-------|-----------|---------------|
| Retina | O(HW) | Low (NumPy/OpenCV) |
| V1 | O(HW·K²·C) | Medium (Conv + SNN) |
| V2 | O(HW·K²·C²) | Medium-High |
| Dorsal | O(HW·K²·C²) | High (3 stages) |
| Ventral | O(HW·K²·C²) | High (2 stages) |

Where:
- H, W: Image height/width
- K: Kernel size
- C: Number of channels

### Performance Targets

- **CPU**: 5-10 FPS (640×480)
- **GPU**: 20-60 FPS (640×480)
- **Memory**: 1-2 GB GPU RAM

---

## Biological vs. Computational Tradeoffs

### Simplifications Made

1. **No feedback connections** (yet)
   - Real cortex has extensive feedback
   - Could add top-down attention modulation

2. **Simplified temporal dynamics**
   - Real neurons have complex dynamics
   - Using simple LIF for computational efficiency

3. **Fixed number of neurons**
   - Real cortex has millions of neurons per area
   - Using 100s-1000s per layer for tractability

4. **No learning during inference**
   - Real cortex constantly adapts
   - Could add online learning (STDP, etc.)

5. **Discrete processing**
   - Real neurons fire asynchronously
   - Processing in discrete time steps

### Preserved Biological Features

1. ✓ Spiking communication
2. ✓ Hierarchical processing
3. ✓ Parallel pathways (dorsal/ventral)
4. ✓ Receptive field structure
5. ✓ ON/OFF pathways
6. ✓ Orientation selectivity
7. ✓ Motion direction selectivity
8. ✓ Position invariance
9. ✓ Category-specific neurons

---

## Integration with Other Brain Areas

### Inputs to Visual Cortex
- **Retina**: Photoreceptor signals
- (Future) **Thalamus (LGN)**: Could add relay processing
- (Future) **Attention system**: Top-down modulation

### Outputs from Visual Cortex

#### To Motor System
```python
action_affordances = visual_cortex.get_action_affordances(outputs)
# Use action_affordances['action_vector'] for motor planning
```

#### To Memory System
```python
object_embedding = visual_cortex.get_object_embedding(outputs)
# Store in hippocampus or visual working memory
```

#### To Decision Making
```python
summary = visual_cortex.get_visual_summary(outputs)
# Use for high-level decisions
```

#### To Attention System
```python
attention_map = outputs['dorsal']['parietal']['attention_map']
# Guide eye movements or attention shifts
```

---

## Extension Points

### Adding New Features

1. **Face Detection** (FFA - Fusiform Face Area)
   ```python
   # Add to ventral stream after IT
   class FusiformFaceArea(nn.Module):
       def __init__(self, it_features):
           # Face-selective neurons
   ```

2. **Place Cells** (Parahippocampal Place Area)
   ```python
   # Add to ventral stream parallel to IT
   class PlaceArea(nn.Module):
       def __init__(self, v4_features):
           # Scene-selective neurons
   ```

3. **Top-Down Attention**
   ```python
   # Add feedback from prefrontal to V1/V2
   def apply_attention_modulation(features, attention_signal):
       return features * attention_signal
   ```

4. **Predictive Coding**
   ```python
   # Add prediction error computation
   def compute_prediction_error(bottom_up, top_down_prediction):
       return bottom_up - top_down_prediction
   ```

---

## Performance Optimization Tips

### For Speed
1. Reduce network sizes (simple_hidden, complex_size)
2. Use GPU (CUDA)
3. Lower input resolution
4. Reduce temporal window size
5. Use fewer orientation channels

### For Memory
1. Use float16 instead of float32
2. Reduce batch size to 1
3. Clear CUDA cache regularly
4. Use gradient checkpointing (if training)

### For Accuracy
1. Increase network sizes
2. Add more orientation channels
3. Increase temporal window
4. Add more object categories
5. Use higher input resolution

---

## Citation

If using this code for research:

```
Neural Visual Cortex Model (2024)
Retina → V1 → V2 → {Dorsal, Ventral} Streams
Biologically-inspired spiking neural network implementation
```

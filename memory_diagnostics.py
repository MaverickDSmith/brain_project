"""
Run this diagnostic to identify the exact source of the memory leak.
This will profile memory usage and show you exactly what's accumulating.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import gc
import torch
import numpy as np
from visual.retina import Retina
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Minimal config for testing
retina = Retina(
    spike_threshold=10,
    fovea_radius_ratio=0.25,
    num_hue_bins=8,
    n_neurons=8,
    temporal_window=2,
    ds_w=20,
    ds_h=15,
    learning_lr=0.03,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    motion_persistence=0.3,
    n_bipolar_types=2,
    n_amacrine_types=3,
    starburst_directions=4,
    gap_junction_strength=0.4,
    gaussian_ksize_rods=3,
    wide_field_size=3
)

# Create dummy frame
dummy_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

print("Starting memory leak test...")
print("=" * 60)

# Get baseline
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated() / 1024**2
    print(f"Baseline GPU memory: {baseline:.2f} MB")

snapshot1 = tracemalloc.take_snapshot()

# Run 50 iterations
for i in range(50):
    outputs = retina.forward(dummy_frame)
    
    # Extract minimal data (simulate visualization)
    _ = outputs['on_spikes'].copy()
    belief = outputs['belief'].copy()
    
    # Delete immediately
    del outputs, belief
    
    if i % 10 == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current = torch.cuda.memory_allocated() / 1024**2
            peak = torch.cuda.max_memory_allocated() / 1024**2
            growth = current - baseline
            print(f"Iter {i:3d}: Current={current:.2f}MB, Peak={peak:.2f}MB, Growth={growth:.2f}MB")

# Final memory snapshot
snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')

print("\n" + "=" * 60)
print("TOP 15 MEMORY ALLOCATIONS (Python side):")
print("=" * 60)
for stat in top_stats[:15]:
    print(stat)

if torch.cuda.is_available():
    print("\n" + "=" * 60)
    print("CUDA MEMORY SUMMARY:")
    print("=" * 60)
    print(torch.cuda.memory_summary())

tracemalloc.stop()

# Now test specific components
print("\n" + "=" * 60)
print("TESTING INDIVIDUAL COMPONENTS:")
print("=" * 60)

def test_component(name, func, iterations=20):
    """Test memory growth of individual component"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated() / 1024**2
    
    for _ in range(iterations):
        func()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        end = torch.cuda.memory_allocated() / 1024**2
        growth = end - start
        print(f"{name:30s}: {growth:+.2f} MB")
        return growth
    return 0

# Test each layer
test_component("Rods", lambda: retina.rods(dummy_frame))
test_component("Cones", lambda: retina.cones(dummy_frame))

rods_out = retina.rods(dummy_frame)
cones_out = retina.cones(dummy_frame)
test_component("Horizontal", lambda: retina.horizontal_layer(rods_out, cones_out))

rods_dog, cones_dog = retina.horizontal_layer(rods_out, cones_out)
test_component("Bipolar", lambda: retina.multiple_bipolar_cells(rods_dog, cones_dog))

bipolar_outputs = retina.multiple_bipolar_cells(rods_dog, cones_dog)
on_chan, off_chan = bipolar_outputs[0]
test_component("Amacrine", lambda: retina.amacrine_layer(on_chan, off_chan))
test_component("Starburst", lambda: retina.starburst_amacrine(on_chan))
test_component("Diverse Amacrine", lambda: retina.diverse_amacrine_cells(on_chan, off_chan))

motion_on, motion_off = retina.amacrine_layer(on_chan, off_chan)
test_component("Ganglion", lambda: retina.ganglion_layer(motion_on, motion_off))

frame_gray = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2GRAY)
test_component("Direction Selective", lambda: retina.direction_selective(frame_gray))

# Test SNN forward pass
on_spikes, off_spikes = retina.ganglion_layer(motion_on, motion_off)
onehot_ds, _ = retina.cones_hue_onehot_downsample(cones_out)
combined = retina.spatiotemporal_encoding(on_spikes, off_spikes, onehot_ds)

def snn_forward_test():
    seq = torch.stack(list(retina.temporal_buffer), dim=0)
    T, H, W, C = seq.shape
    seq_reshaped = seq.reshape(T, H*W, C)
    
    local_state = retina.snn_state
    spikes = []
    for t in range(T):
        spk, local_state = retina.snn(seq_reshaped[t], local_state)
        spikes.append(spk.detach())
    
    del seq, seq_reshaped, spikes

test_component("SNN Forward Pass", snn_forward_test)

# Test STDP
def stdp_test():
    pre = torch.rand(100, retina.snn_input_size, device=retina.DEVICE)
    post = torch.rand(100, retina.N_NEURONS, device=retina.DEVICE)
    retina.stdp_batch_update(retina.snn.fc.weight, pre, post, lr=0.03)
    del pre, post

test_component("STDP Update", stdp_test)

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print("\nLook for components with positive growth (+X.XX MB)")
print("Those are your memory leak sources.")
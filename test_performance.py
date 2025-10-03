"""
Performance Test - Measure FPS and find bottlenecks
====================================================
"""

import time
import torch
import numpy as np
import cv2
from visual.visual_cortex import VisualCortex

print("=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  Running on CPU - will be slower")

# Create test frame
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 0, 0), -1)
cv2.circle(test_frame, (500, 240), 80, (0, 255, 0), -1)

# Initialize
print("\nInitializing visual cortex...")
start = time.time()
visual_cortex = VisualCortex(device=device, spike_threshold=10)
init_time = time.time() - start
print(f"Initialization: {init_time:.2f}s")

# Warmup (first few frames are slower)
print("\nWarming up (3 frames)...")
for i in range(3):
    _ = visual_cortex.forward(test_frame)
print("Warmup complete")

# Detailed timing breakdown
print("\n" + "=" * 60)
print("TIMING BREAKDOWN (single frame)")
print("=" * 60)

# Retina
start = time.time()
retina_out = visual_cortex.retina.forward(test_frame)
retina_time = time.time() - start

# V1
start = time.time()
with torch.no_grad():
    v1_out = visual_cortex.v1(retina_out['on_spikes'], retina_out['off_spikes'])
v1_time = time.time() - start

# V2
start = time.time()
with torch.no_grad():
    v2_out = visual_cortex.v2(v1_out)
v2_time = time.time() - start

# Dorsal
start = time.time()
with torch.no_grad():
    dorsal_out = visual_cortex.dorsal_stream(v2_out)
dorsal_time = time.time() - start

# Ventral
start = time.time()
with torch.no_grad():
    ventral_out = visual_cortex.ventral_stream(v2_out)
ventral_time = time.time() - start

total_breakdown = retina_time + v1_time + v2_time + dorsal_time + ventral_time

print(f"Retina:  {retina_time*1000:6.1f}ms ({retina_time/total_breakdown*100:5.1f}%)")
print(f"V1:      {v1_time*1000:6.1f}ms ({v1_time/total_breakdown*100:5.1f}%)")
print(f"V2:      {v2_time*1000:6.1f}ms ({v2_time/total_breakdown*100:5.1f}%)")
print(f"Dorsal:  {dorsal_time*1000:6.1f}ms ({dorsal_time/total_breakdown*100:5.1f}%)")
print(f"Ventral: {ventral_time*1000:6.1f}ms ({ventral_time/total_breakdown*100:5.1f}%)")
print(f"{'='*40}")
print(f"Total:   {total_breakdown*1000:6.1f}ms")

# Full pipeline timing
print("\n" + "=" * 60)
print("FULL PIPELINE TEST (10 frames)")
print("=" * 60)

times = []
for i in range(10):
    start = time.time()
    outputs = visual_cortex.forward(test_frame)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Frame {i+1:2d}: {elapsed*1000:6.1f}ms", end='')
    if i == 0:
        print(" (first frame, may include overhead)")
    else:
        print()

avg_time = np.mean(times)
min_time = np.min(times)
max_time = np.max(times)
fps = 1.0 / avg_time

print(f"\n{'='*40}")
print(f"Average: {avg_time*1000:6.1f}ms")
print(f"Min:     {min_time*1000:6.1f}ms")
print(f"Max:     {max_time*1000:6.1f}ms")
print(f"FPS:     {fps:6.1f}")

# Verdict
print("\n" + "=" * 60)
print("PERFORMANCE VERDICT")
print("=" * 60)

if fps >= 40:
    print("✅ EXCELLENT - Real-time capable!")
    print(f"   {fps:.1f} FPS is great for demos and real-time use")
elif fps >= 20:
    print("✅ GOOD - Usable for real-time")
    print(f"   {fps:.1f} FPS is acceptable for demos")
elif fps >= 10:
    print("⚠️  ACCEPTABLE - Borderline real-time")
    print(f"   {fps:.1f} FPS may feel a bit sluggish")
elif fps >= 5:
    print("⚠️  SLOW - Not ideal for real-time")
    print(f"   {fps:.1f} FPS will feel laggy")
else:
    print("❌ TOO SLOW - Not suitable for real-time")
    print(f"   {fps:.1f} FPS is unusable")

# Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)

if fps < 20:
    print("\n📌 To improve performance:")
    print("   1. Lower resolution: frame = cv2.resize(frame, (320, 240))")
    print("   2. Reduce network size: Edit visual_cortex.py")
    print("   3. Process every Nth frame: if frame_count % 2 == 0: ...")

    if device == 'cpu':
        print("   4. ⚠️  USE GPU! You're on CPU - this is MUCH slower")
    else:
        print(f"   4. Check GPU utilization: nvidia-smi")

    print("\nSee PERFORMANCE_FIX.md for details")
else:
    print("\n✅ Performance is good! No changes needed.")
    print("   The demo should run smoothly.")

# GPU memory check
if device == 'cuda':
    print("\n" + "=" * 60)
    print("GPU MEMORY")
    print("=" * 60)
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")

print("\n" + "=" * 60)
print("Ready to run demo:")
print("  python demo_visual_cortex.py")
print("=" * 60 + "\n")

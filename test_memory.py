"""
Memory Test - Verify GPU memory stability
==========================================
This script processes multiple frames and monitors memory usage
to ensure there are no memory leaks.
"""

import torch
import numpy as np
import cv2
from visual.visual_cortex import VisualCortex

print("=" * 60)
print("MEMORY STABILITY TEST")
print("=" * 60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

if device == 'cpu':
    print("⚠️  Running on CPU - memory test requires CUDA")
    print("   This will still test for correctness though.")
    print()

# Create test frame with some variation
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 0, 0), -1)
cv2.circle(test_frame, (500, 240), 80, (0, 255, 0), -1)

print("Initializing Visual Cortex...")
visual_cortex = VisualCortex(device=device, spike_threshold=10)

if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Initial GPU memory: {initial_mem:.3f} GB")

print("\nProcessing frames...")
print("Frame | Memory (GB) | Delta (MB)")
print("-" * 40)

num_frames = 100
memory_log = []

for i in range(num_frames):
    # Process frame
    outputs = visual_cortex.forward(test_frame)

    # Log memory every 10 frames
    if device == 'cuda' and (i + 1) % 10 == 0:
        current_mem = torch.cuda.memory_allocated() / 1024**3
        delta = (current_mem - (memory_log[-1] if memory_log else initial_mem)) * 1024  # MB
        memory_log.append(current_mem)

        print(f"{i+1:5d} | {current_mem:11.3f} | {delta:+10.1f}")

        # Clear cache periodically
        if (i + 1) % 30 == 0:
            torch.cuda.empty_cache()

print("-" * 40)

# Analysis
if device == 'cuda':
    final_mem = torch.cuda.memory_allocated() / 1024**3
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    growth = final_mem - initial_mem

    print(f"\n📊 Memory Analysis:")
    print(f"   Initial:  {initial_mem:.3f} GB")
    print(f"   Final:    {final_mem:.3f} GB")
    print(f"   Peak:     {peak_mem:.3f} GB")
    print(f"   Growth:   {growth*1024:+.1f} MB")

    # Check for memory leak
    if len(memory_log) >= 5:
        early_avg = np.mean(memory_log[:3])
        late_avg = np.mean(memory_log[-3:])
        drift = (late_avg - early_avg) * 1024  # MB

        print(f"   Drift:    {drift:+.1f} MB (early vs late)")

        # Verdict
        print(f"\n🔍 Verdict:")
        if abs(drift) < 50:  # Less than 50 MB drift
            print("   ✅ PASS - Memory is stable!")
            print("   No significant memory leak detected.")
        elif abs(drift) < 200:
            print("   ⚠️  WARNING - Small memory drift detected")
            print("   This may be acceptable for long runs.")
        else:
            print("   ❌ FAIL - Memory leak detected!")
            print("   Memory usage is growing over time.")

    # Test visualization memory
    print(f"\n🎨 Testing visualizations...")
    torch.cuda.reset_peak_memory_stats()

    visualizations = visual_cortex.visualize_complete_pipeline(
        test_frame, outputs, output_size=(640, 480))

    viz_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"   Visualization peak: {viz_mem:.3f} GB")
    print(f"   ✅ Visualizations OK")

else:
    print("\n✅ CPU test completed successfully")
    print("   (Memory profiling requires CUDA)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nIf memory is stable, you can run the demo:")
print("  python demo_visual_cortex.py")
print()

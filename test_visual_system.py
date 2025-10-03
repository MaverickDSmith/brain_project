"""
Visual System Test Suite
=========================
Quick tests to verify all components are working correctly.
"""

import torch
import numpy as np
import cv2
import sys

print("=" * 60)
print("VISUAL SYSTEM TEST SUITE")
print("=" * 60)

# Test 1: Check dependencies
print("\n[1/6] Checking dependencies...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ✗ PyTorch error: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"  ✗ OpenCV error: {e}")
    sys.exit(1)

try:
    import snntorch
    print(f"  ✓ snnTorch (SNN backend available)")
    snn_available = True
except ImportError:
    try:
        import norse
        print(f"  ✓ Norse (SNN backend available)")
        snn_available = True
    except ImportError:
        print(f"  ⚠ No SNN backend found (will use non-spiking fallback)")
        snn_available = False

# Test 2: Import modules
print("\n[2/6] Importing visual system modules...")
try:
    from visual.retina_scratch import Retina
    print("  ✓ Retina")
except Exception as e:
    print(f"  ✗ Retina import failed: {e}")
    sys.exit(1)

try:
    from visual.v1 import V1
    print("  ✓ V1")
except Exception as e:
    print(f"  ✗ V1 import failed: {e}")
    sys.exit(1)

try:
    from visual.v2 import V2
    print("  ✓ V2")
except Exception as e:
    print(f"  ✗ V2 import failed: {e}")
    sys.exit(1)

try:
    from visual.dorsal_stream import DorsalStream
    print("  ✓ Dorsal Stream")
except Exception as e:
    print(f"  ✗ Dorsal Stream import failed: {e}")
    sys.exit(1)

try:
    from visual.ventral_stream import VentralStream
    print("  ✓ Ventral Stream")
except Exception as e:
    print(f"  ✗ Ventral Stream import failed: {e}")
    sys.exit(1)

try:
    from visual.visual_cortex import VisualCortex
    print("  ✓ Visual Cortex")
except Exception as e:
    print(f"  ✗ Visual Cortex import failed: {e}")
    sys.exit(1)

# Test 3: Create test image
print("\n[3/6] Creating test image...")
test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
# Add some patterns
cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 0, 0), -1)
cv2.circle(test_frame, (500, 240), 80, (0, 255, 0), -1)
cv2.line(test_frame, (0, 0), (640, 480), (0, 0, 255), 5)
print("  ✓ Test image created (480x640)")

# Test 4: Test Retina
print("\n[4/6] Testing Retina...")
try:
    retina = Retina()
    retina_out = retina.forward(test_frame)

    assert 'on_spikes' in retina_out, "Missing on_spikes"
    assert 'off_spikes' in retina_out, "Missing off_spikes"
    assert 'motion_on' in retina_out, "Missing motion_on"

    print(f"  ✓ Retina forward pass successful")
    print(f"  ✓ Output shapes: {retina_out['on_spikes'].shape}")
except Exception as e:
    print(f"  ✗ Retina test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test V1
print("\n[5/6] Testing V1...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    v1 = V1(device=device)
    v1_out = v1(retina_out['on_spikes'], retina_out['off_spikes'])

    assert 'simple_spikes' in v1_out, "Missing simple_spikes"
    assert 'complex_features' in v1_out, "Missing complex_features"
    assert 'orientation_map' in v1_out, "Missing orientation_map"

    print(f"  ✓ V1 forward pass successful")
    print(f"  ✓ Orientation map shape: {v1_out['orientation_map'].shape}")
except Exception as e:
    print(f"  ✗ V1 test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test complete visual cortex
print("\n[6/6] Testing complete Visual Cortex...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visual_cortex = VisualCortex(device=device)

    print("  ✓ Visual Cortex initialized")

    # Process frame
    outputs = visual_cortex.forward(test_frame)

    # Check outputs
    assert 'retina' in outputs, "Missing retina output"
    assert 'v1' in outputs, "Missing V1 output"
    assert 'v2' in outputs, "Missing V2 output"
    assert 'dorsal' in outputs, "Missing dorsal output"
    assert 'ventral' in outputs, "Missing ventral output"

    print(f"  ✓ Complete pipeline forward pass successful")

    # Test summary
    summary = visual_cortex.get_visual_summary(outputs)
    print(f"  ✓ Visual summary generated")
    print(f"    - Motion: {summary['motion']['dominant_direction']}")
    print(f"    - Top objects: {summary['objects']['detected_objects'][:3]}")

    # Test visualizations
    visualizations = visual_cortex.visualize_complete_pipeline(test_frame, outputs)
    print(f"  ✓ Visualizations generated ({len(visualizations)} views)")

except Exception as e:
    print(f"  ✗ Visual Cortex test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nYour visual system is ready to use!")
print("\nNext steps:")
print("  1. Run 'python demo_retina_scratch.py' to test retina")
print("  2. Run 'python demo_visual_cortex.py' for complete demo")
print("\nFor webcam input, make sure your camera is connected.")
print("=" * 60)

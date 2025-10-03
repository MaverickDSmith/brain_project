"""
Visual Cortex - Complete Integration
=====================================
Integrates the entire visual processing pipeline from retina to high-level vision.

Pipeline:
    Retina → LGN → V1 → V2 → {Dorsal Stream (WHERE/HOW), Ventral Stream (WHAT)}

This module orchestrates all visual processing and provides unified outputs
for downstream brain areas.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

from visual.retina_scratch import Retina
from visual.v1 import V1
from visual.v2 import V2
from visual.dorsal_stream import DorsalStream
from visual.ventral_stream import VentralStream


class VisualCortex(nn.Module):
    """
    Complete visual cortex model integrating all visual processing streams.

    OPTIMIZED VERSION: Reduced from 46.87M parameters to ~2-3M parameters.
    Uses depthwise-separable convolutions and reduced feature dimensions
    while maintaining biological architecture.
    """
    def __init__(self, device='cpu', spike_threshold=10):
        super().__init__()
        self.device = device

        # Retina (input layer - lightweight NumPy/OpenCV processing)
        self.retina = Retina(
            spike_threshold=spike_threshold,
            fovea_radius_ratio=0.25,
            motion_persistence=0.2
        )

        # V1 (Primary visual cortex - OPTIMIZED: 32 hidden instead of 128)
        self.v1 = V1(
            input_channels=2,  # ON and OFF channels
            num_orientations=8,
            simple_hidden=32,  # Reduced from 128
            complex_size=32,   # Reduced from 64
            device=device
        )

        # V2 (Secondary visual cortex - OPTIMIZED: 64 hidden instead of 256)
        self.v2 = V2(
            v1_simple_hidden=32,  # Match V1 simple_hidden parameter
            num_orientations=8,
            hidden_size=64,  # Reduced from 256
            device=device
        )

        # Dorsal Stream (WHERE/HOW - OPTIMIZED: 64 channels throughout)
        self.dorsal_stream = DorsalStream(
            v2_channels=64,  # Reduced from 256
            num_directions=8,
            device=device
        )

        # Ventral Stream (WHAT - OPTIMIZED: 64 channels, 32 categories)
        self.ventral_stream = VentralStream(
            v2_channels=64,  # Reduced from 256
            num_object_categories=32,  # Reduced from 64
            device=device
        )

        # Frame counter
        self.frame_count = 0

        # Statistics
        self.processing_times = []

    def reset_states(self):
        """Reset all internal states across the entire visual cortex"""
        self.retina.prev_bipolar_on = None
        self.retina.prev_bipolar_off = None
        self.retina.motion_on_accumulator = None
        self.retina.motion_off_accumulator = None

        self.v1.reset_states()
        self.v2.reset_states()
        self.dorsal_stream.reset_states()
        self.ventral_stream.reset_states()

        self.frame_count = 0
        self.processing_times = []

    def forward(self, frame):
        """
        Process a single frame through the entire visual cortex.

        Args:
            frame: Input BGR image (numpy array, uint8)

        Returns:
            Complete visual cortex outputs including:
                - Retinal processing
                - V1 orientation features
                - V2 contour/texture features
                - Dorsal stream (spatial/motion)
                - Ventral stream (object recognition)
        """
        # Use no_grad for inference to prevent memory accumulation
        with torch.no_grad():
            # Stage 1: Retina
            retina_outputs = self.retina.forward(frame)

            # Stage 2: V1 (Primary visual cortex)
            v1_outputs = self.v1(
                retina_outputs['on_spikes'],
                retina_outputs['off_spikes']
            )

            # Stage 3: V2 (Secondary visual cortex)
            v2_outputs = self.v2(v1_outputs)

            # Stage 4a: Dorsal Stream (WHERE/HOW pathway)
            dorsal_outputs = self.dorsal_stream(v2_outputs)

            # Stage 4b: Ventral Stream (WHAT pathway)
            ventral_outputs = self.ventral_stream(v2_outputs)

        self.frame_count += 1

        return {
            'retina': retina_outputs,
            'v1': v1_outputs,
            'v2': v2_outputs,
            'dorsal': dorsal_outputs,
            'ventral': ventral_outputs,
            'frame_count': self.frame_count
        }

    def get_visual_summary(self, outputs):
        """
        Extract high-level summary of what the visual system perceives.

        Returns human-readable summary of:
        - Detected motion and direction
        - Spatial attention focus
        - Recognized objects
        - Orientation energy
        """
        # Motion summary from dorsal stream
        motion_summary = self.dorsal_stream.get_motion_summary(outputs['dorsal'])

        # Object summary from ventral stream
        object_summary = self.ventral_stream.get_object_summary(outputs['ventral'])

        # V1 orientation activity
        v1_stats = self.v1.get_statistics()

        summary = {
            'frame': self.frame_count,
            'motion': motion_summary,
            'objects': object_summary,
            'v1_spikes': v1_stats['avg_spikes_per_frame'],
            'spatial_attention_peak': motion_summary['attention_peak']
        }

        return summary

    def get_action_affordances(self, outputs):
        """
        Extract action affordances from parietal cortex.
        These represent possible actions based on the visual scene.

        Returns:
            Action vector and spatial attention map
        """
        parietal = outputs['dorsal']['parietal']

        return {
            'action_vector': parietal['action_vector'].detach().cpu().numpy(),
            'attention_map': parietal['attention_map'].detach().cpu().numpy(),
            'spatial_features': parietal['spatial_features'].shape
        }

    def get_object_embedding(self, outputs):
        """
        Get invariant object representation from IT cortex.
        This can be used for object comparison, memory, etc.

        Returns:
            Object embedding vector (high-dimensional representation)
        """
        it_embedding = outputs['ventral']['it']['it_object_embedding']

        return it_embedding.detach().cpu().numpy()

    def visualize_complete_pipeline(self, frame, outputs, output_size=(640, 480)):
        """
        Create a comprehensive visualization showing all processing stages.

        Returns a dictionary of visualization images.
        """
        visualizations = {}

        # Retina outputs
        visualizations['retina_rods'] = cv2.cvtColor(
            outputs['retina']['rods_out'], cv2.COLOR_GRAY2BGR)
        visualizations['retina_on_spikes'] = cv2.cvtColor(
            outputs['retina']['on_spikes'], cv2.COLOR_GRAY2BGR)
        visualizations['retina_off_spikes'] = cv2.cvtColor(
            outputs['retina']['off_spikes'], cv2.COLOR_GRAY2BGR)
        visualizations['retina_motion'] = cv2.cvtColor(
            outputs['retina']['motion_on'], cv2.COLOR_GRAY2BGR)

        # V1 outputs
        visualizations['v1_orientations'] = self.v1.visualize_orientation_map(
            outputs['v1']['orientation_map'], output_size)

        # V2 outputs
        visualizations['v2_features'] = self.v2.visualize_features(
            outputs['v2'], output_size)

        # Dorsal stream outputs
        visualizations['dorsal_motion_field'] = self.dorsal_stream.visualize_motion_field(
            outputs['dorsal']['mt'], output_size)
        visualizations['dorsal_attention'] = self.dorsal_stream.visualize_attention(
            outputs['dorsal']['parietal'], output_size)

        # Ventral stream outputs
        visualizations['ventral_color'] = self.ventral_stream.visualize_color_features(
            outputs['ventral']['v4'], output_size)
        visualizations['ventral_shape'] = self.ventral_stream.visualize_shape_features(
            outputs['ventral']['v4'], output_size)
        visualizations['ventral_categories'] = self.ventral_stream.visualize_category_activations(
            outputs['ventral']['it'])

        # Resize all to match output size (except category bar chart)
        for key in visualizations:
            if key != 'ventral_categories' and visualizations[key].shape[:2] != output_size[::-1]:
                visualizations[key] = cv2.resize(visualizations[key], output_size)

        return visualizations

    def create_summary_display(self, frame, outputs, visualizations):
        """
        Create a multi-panel summary display showing key outputs.

        Layout:
        [Original] [V1 Orient] [Motion]
        [Attention] [Color] [Shape]
        """
        output_size = (320, 240)

        # Resize frame
        frame_resized = cv2.resize(frame, output_size)

        # Get visualizations (already resized)
        v1_orient = cv2.resize(visualizations['v1_orientations'], output_size)
        motion = cv2.resize(visualizations['dorsal_motion_field'], output_size)
        attention = cv2.resize(visualizations['dorsal_attention'], output_size)
        color = cv2.resize(visualizations['ventral_color'], output_size)
        shape = cv2.resize(visualizations['ventral_shape'], output_size)

        # Create rows
        row1 = np.hstack([frame_resized, v1_orient, motion])
        row2 = np.hstack([attention, color, shape])

        # Stack rows
        display = np.vstack([row1, row2])

        # Add labels
        cv2.putText(display, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "V1 Orientations", (output_size[0] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Dorsal: Motion", (output_size[0] * 2 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Dorsal: Attention", (10, output_size[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Ventral: Color", (output_size[0] + 10, output_size[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Ventral: Shape", (output_size[0] * 2 + 10, output_size[1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add frame counter
        cv2.putText(display, f"Frame: {self.frame_count}", (10, display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display

    def log_statistics(self, outputs):
        """
        Print processing statistics to console.
        """
        summary = self.get_visual_summary(outputs)

        print(f"\n=== Frame {summary['frame']} ===")
        print(f"Motion: {summary['motion']['dominant_direction']}")
        print(f"Top Objects: {summary['objects']['detected_objects'][:3]}")
        print(f"Attention Peak: {summary['spatial_attention_peak']:.3f}")
        print(f"V1 Avg Spikes: {summary['v1_spikes']:.1f}")

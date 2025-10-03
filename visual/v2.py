"""
V2 (Secondary Visual Cortex) - Extrastriate Cortex
===================================================
Processes more complex visual features than V1.

Key functions:
- Contour integration (linking edges into coherent shapes)
- Figure-ground segmentation
- Texture and pattern analysis
- Higher-order orientation combinations
- Border ownership
- Illusory contours

Input: V1 simple and complex cell outputs
Output: Integrated contour and texture features for dorsal/ventral streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

try:
    import snntorch as snn
    from snntorch import surrogate
    SNN_BACKEND = 'snntorch'
except ImportError:
    try:
        import norse.torch as norse
        SNN_BACKEND = 'norse'
    except ImportError:
        SNN_BACKEND = None


class ContourIntegration(nn.Module):
    """
    Contour integration cells connect local edge elements into longer contours.
    This is crucial for object boundary detection.

    Optimized: Reduced expansion factor and simplified grouping.
    """
    def __init__(self, num_orientations=8, channels_per_orientation=4, device='cpu'):
        super().__init__()
        self.num_orientations = num_orientations
        self.channels_per_orientation = channels_per_orientation
        self.device = device

        # Lightweight horizontal connections for grouping collinear edges
        # Each orientation has its own processing layer
        self.contour_grouping = nn.ModuleList([
            nn.Conv2d(channels_per_orientation, channels_per_orientation,
                     kernel_size=5, padding=2,
                     groups=channels_per_orientation, bias=False)  # Depthwise
            for _ in range(num_orientations)
        ])

        # Lightweight integration across orientations (smaller expansion)
        total_channels = channels_per_orientation * num_orientations
        self.cross_orientation = nn.Conv2d(
            total_channels,
            total_channels,  # No expansion, just integration
            kernel_size=1,
            bias=False
        )

        self.to(device)

    def forward(self, v1_complex_features, v1_simple_spikes):
        """
        Integrate contours from V1 outputs.

        Args:
            v1_complex_features: Complex cell features from V1
            v1_simple_spikes: List of simple cell spikes per orientation

        Returns:
            Integrated contour features
        """
        # Group contours for each orientation
        grouped = []
        for i, (spikes, grouper) in enumerate(zip(v1_simple_spikes, self.contour_grouping)):
            # Long-range horizontal connections
            grouped_contour = grouper(spikes)
            grouped.append(grouped_contour)

        # Stack orientation channels
        stacked = torch.cat(grouped, dim=1)

        # Cross-orientation integration
        integrated = self.cross_orientation(stacked)
        integrated = F.relu(integrated)

        return integrated


class TextureProcessing(nn.Module):
    """
    V2 is sensitive to texture patterns and surface properties.

    Optimized: Lightweight depthwise-separable convolutions for texture analysis.
    """
    def __init__(self, input_channels, output_channels=64, device='cpu'):
        super().__init__()
        self.device = device
        self.output_channels = output_channels

        # Simplified 2-scale texture analysis (instead of 3)
        channels_per_scale = output_channels // 2
        remainder = output_channels % 2

        scale_channels = [
            channels_per_scale + (1 if i < remainder else 0)
            for i in range(2)
        ]

        # Lightweight multi-scale texture analysis (depthwise-separable)
        self.texture_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=k,
                         padding=k//2, bias=False, groups=input_channels),  # Depthwise
                nn.Conv2d(input_channels, scale_channels[i], kernel_size=1, bias=False),  # Pointwise
                nn.BatchNorm2d(scale_channels[i]),
                nn.ReLU()
            )
            for i, k in enumerate([3, 5])  # Two receptive field sizes
        ])

        # Texture integration
        combined_channels = sum(scale_channels)
        self.texture_combine = nn.Conv2d(combined_channels, output_channels,
                                         kernel_size=1, bias=False)

        self.to(device)

    def forward(self, features):
        """Extract texture features at multiple scales"""
        texture_features = []

        for scale_processor in self.texture_scales:
            texture = scale_processor(features)
            texture_features.append(texture)

        # Concatenate scales
        combined = torch.cat(texture_features, dim=1)

        # Final integration
        output = self.texture_combine(combined)

        return output


class V2Cells(nn.Module):
    """
    V2 neurons with spiking dynamics.

    Optimized: Lightweight processing with depthwise-separable convolutions.
    """
    def __init__(self, input_channels, hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        # Lightweight main processing (depthwise-separable)
        self.conv1_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                  padding=1, bias=False, groups=input_channels)
        self.conv1_pw = nn.Conv2d(input_channels, hidden_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)

        self.conv2_dw = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1, bias=False, groups=hidden_size)
        self.conv2_pw = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        # Spiking neurons
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.85
            self.lif1 = snn.Leaky(beta=self.beta, threshold=1.0,
                                 spike_grad=surrogate.fast_sigmoid())
            self.lif2 = snn.Leaky(beta=self.beta, threshold=1.0,
                                 spike_grad=surrogate.fast_sigmoid())
        elif SNN_BACKEND == 'norse':
            self.lif1 = norse.LIFCell()
            self.lif2 = norse.LIFCell()

        self.mem1 = None
        self.mem2 = None

        self.to(device)

    def reset_states(self):
        self.mem1 = None
        self.mem2 = None

    def forward(self, x):
        """
        Process through V2 spiking neurons.

        Args:
            x: Input features

        Returns:
            spikes: Output spikes
            features: Non-spiking features for downstream processing
        """
        # First layer (depthwise-separable)
        x = self.conv1_dw(x)
        x = self.conv1_pw(x)
        x = self.bn1(x)

        if SNN_BACKEND == 'snntorch':
            spk1, self.mem1 = self.lif1(x, self.mem1)
        elif SNN_BACKEND == 'norse':
            spk1, self.mem1 = self.lif1(x, self.mem1)
        else:
            spk1 = F.relu(x)

        # Second layer (depthwise-separable)
        x = self.conv2_dw(spk1)
        x = self.conv2_pw(x)
        x = self.bn2(x)

        if SNN_BACKEND == 'snntorch':
            spk2, self.mem2 = self.lif2(x, self.mem2)
        elif SNN_BACKEND == 'norse':
            spk2, self.mem2 = self.lif2(x, self.mem2)
        else:
            spk2 = F.relu(x)

        return spk2, spk1  # Return both for multi-level features


class V2(nn.Module):
    """
    Complete V2 model integrating contours, textures, and complex features.

    Optimized: Reduced hidden dimensions and efficient processing.
    Target: <100K parameters instead of 5.78M.
    """
    def __init__(self, v1_simple_hidden=32, num_orientations=8,
                 hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.num_orientations = num_orientations

        # Calculate channels per orientation from V1 simple cells
        # V1 simple cells output: hidden_size // num_orientations per orientation
        channels_per_orientation = v1_simple_hidden // num_orientations

        # Contour integration (lightweight)
        self.contour_integrator = ContourIntegration(
            num_orientations=num_orientations,
            channels_per_orientation=channels_per_orientation,
            device=device
        )

        # Texture processing (lightweight)
        # Contour integrator outputs: channels_per_orientation * num_orientations (no expansion)
        contour_output_channels = channels_per_orientation * num_orientations
        texture_output_channels = contour_output_channels  # Match contour channels
        self.texture_processor = TextureProcessing(
            input_channels=contour_output_channels,
            output_channels=texture_output_channels,
            device=device
        )

        # V2 cells (spiking, lightweight)
        combined_channels = contour_output_channels + texture_output_channels
        self.v2_cells = V2Cells(
            input_channels=combined_channels,  # Texture + contour
            hidden_size=hidden_size,
            device=device
        )

        # Statistics
        self.frame_count = 0
        self.spike_history = []

    def reset_states(self):
        """Reset all internal states"""
        self.v2_cells.reset_states()
        self.frame_count = 0
        self.spike_history = []

    def forward(self, v1_outputs):
        """
        Process V1 outputs through V2.

        Args:
            v1_outputs: Dictionary from V1 containing:
                - complex_features: Position-invariant features
                - simple_spikes: List of orientation-specific spikes

        Returns:
            dict with:
                - v2_spikes: Spiking output
                - contour_features: Integrated contours
                - texture_features: Texture representations
                - combined_features: Full V2 representation
        """
        v1_complex = v1_outputs['complex_features']
        v1_simple_spikes = v1_outputs['simple_spikes']

        # Contour integration
        contour_features = self.contour_integrator(v1_complex, v1_simple_spikes)

        # Texture processing
        texture_features = self.texture_processor(contour_features)

        # Combine contour and texture
        combined = torch.cat([contour_features, texture_features], dim=1)

        # V2 spiking cells
        v2_spikes, v2_intermediate = self.v2_cells(combined)

        # Logging
        self.frame_count += 1
        total_spikes = v2_spikes.sum().item()
        self.spike_history.append(total_spikes)

        return {
            'v2_spikes': v2_spikes,
            'v2_intermediate': v2_intermediate,
            'contour_features': contour_features,
            'texture_features': texture_features,
            'combined_features': combined,
            'total_spikes': total_spikes,
            'frame_count': self.frame_count
        }

    def get_statistics(self):
        """Get statistics for logging/visualization"""
        stats = {
            'frame_count': self.frame_count,
            'total_spikes_history': self.spike_history,
            'avg_spikes_per_frame': np.mean(self.spike_history) if self.spike_history else 0
        }
        return stats

    def visualize_features(self, v2_outputs, output_size=(640, 480)):
        """
        Visualize V2 feature activations.
        """
        # Get spike map
        spikes = v2_outputs['v2_spikes'][0]  # First batch

        # Sum across channels to get activation map
        activation_map = spikes.sum(dim=0).detach().cpu().numpy()

        # Normalize to 0-255
        if activation_map.max() > 0:
            activation_map = (activation_map / activation_map.max() * 255).astype(np.uint8)
        else:
            activation_map = np.zeros_like(activation_map, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

        # Resize
        resized = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)

        return resized

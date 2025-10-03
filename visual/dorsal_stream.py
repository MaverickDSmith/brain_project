"""
Dorsal Stream - "WHERE/HOW" Pathway
====================================
Processes spatial locations, motion, and visuomotor transformations.

Path: V1 → V2 → V3 → V5/MT → Parietal Cortex

Key functions:
- Motion analysis
- Spatial awareness
- Depth perception
- Visuomotor coordination
- Navigation

Output: Spatial maps, motion vectors, action affordances
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


class V3(nn.Module):
    """
    V3 - Dynamic Form Processing
    Processes global motion patterns and dynamic shapes.
    Sensitive to shape-from-motion.

    Optimized: Lightweight depthwise-separable convolutions.
    """
    def __init__(self, input_channels=64, hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device

        # Lightweight motion processing (depthwise-separable)
        self.motion_conv1_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                         padding=1, bias=False, groups=input_channels)
        self.motion_conv1_pw = nn.Conv2d(input_channels, hidden_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)

        # Lightweight temporal integration
        self.temporal_conv_dw = nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3,
                                          padding=1, bias=False, groups=hidden_size * 2)
        self.temporal_conv_pw = nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        # Spiking layer
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.8
            self.lif = snn.Leaky(beta=self.beta, threshold=1.0,
                                spike_grad=surrogate.fast_sigmoid())
        elif SNN_BACKEND == 'norse':
            self.lif = norse.LIFCell()

        self.prev_features = None
        self.mem = None

        self.to(device)

    def reset_states(self):
        self.prev_features = None
        self.mem = None

    def forward(self, v2_features):
        """
        Process V2 features for dynamic form.

        Args:
            v2_features: Features from V2

        Returns:
            V3 motion-enhanced features and spikes
        """
        # Process current features (depthwise-separable)
        x = self.motion_conv1_dw(v2_features)
        x = self.motion_conv1_pw(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Temporal integration (compare with previous frame)
        if self.prev_features is not None:
            # Concatenate current and previous
            temporal = torch.cat([x, self.prev_features], dim=1)
            x = self.temporal_conv_dw(temporal)
            x = self.temporal_conv_pw(x)
            x = self.bn2(x)
            x = F.relu(x)

        # Update previous features
        self.prev_features = x.detach()

        # Spiking neurons
        if SNN_BACKEND == 'snntorch':
            spikes, self.mem = self.lif(x, self.mem)
        elif SNN_BACKEND == 'norse':
            spikes, self.mem = self.lif(x, self.mem)
        else:
            spikes = F.relu(x)

        return {
            'v3_spikes': spikes,
            'v3_features': x
        }


class V5_MT(nn.Module):
    """
    V5/MT (Middle Temporal) - Motion Processing
    Specialized for detecting motion direction and speed.
    Contains neurons tuned to specific motion directions.

    Optimized: Lightweight direction detectors with depthwise convs.
    """
    def __init__(self, input_channels=64, num_directions=8, hidden_size=64, device='cpu'):
        super().__init__()
        self.device = device
        self.num_directions = num_directions

        # Lightweight direction-selective channels (depthwise)
        self.direction_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size=5, padding=2,
                         bias=False, groups=input_channels),  # Depthwise
                nn.Conv2d(input_channels, hidden_size // num_directions,
                         kernel_size=1, bias=False),  # Pointwise
                nn.BatchNorm2d(hidden_size // num_directions),
                nn.ReLU()
            )
            for _ in range(num_directions)
        ])

        # Lightweight opponent motion (depthwise-separable)
        self.opponent_motion_dw = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                            padding=1, bias=False, groups=hidden_size)
        self.opponent_motion_pw = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)

        # Speed tuning
        # Divide hidden_size evenly among 3 speed layers
        channels_per_speed = hidden_size // 3
        remainder = hidden_size % 3
        speed_channels = [
            channels_per_speed + (1 if i < remainder else 0)
            for i in range(3)
        ]

        self.speed_layers = nn.ModuleList([
            nn.Conv2d(hidden_size, speed_channels[i], kernel_size=3,
                     stride=s, padding=1, bias=False)
            for i, s in enumerate([1, 2, 3])  # Different speeds via different strides
        ])

        # Spiking neurons for each direction
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.75
            self.direction_lifs = nn.ModuleList([
                snn.Leaky(beta=self.beta, threshold=1.0,
                         spike_grad=surrogate.fast_sigmoid())
                for _ in range(num_directions)
            ])
        elif SNN_BACKEND == 'norse':
            self.direction_lifs = nn.ModuleList([
                norse.LIFCell() for _ in range(num_directions)
            ])

        self.direction_mems = [None] * num_directions

        self.to(device)

    def reset_states(self):
        self.direction_mems = [None] * self.num_directions

    def forward(self, v3_outputs):
        """
        Extract motion direction and speed.

        Args:
            v3_outputs: Dictionary from V3

        Returns:
            Motion representation with direction and speed
        """
        v3_features = v3_outputs['v3_features']

        # Direction-selective processing
        direction_responses = []
        direction_spikes = []

        for i, (detector, lif) in enumerate(zip(self.direction_detectors, self.direction_lifs)):
            # Detect specific direction
            direction_feat = detector(v3_features)
            direction_responses.append(direction_feat)

            # Spiking response
            if SNN_BACKEND == 'snntorch':
                spk, self.direction_mems[i] = lif(direction_feat, self.direction_mems[i])
            elif SNN_BACKEND == 'norse':
                spk, self.direction_mems[i] = lif(direction_feat, self.direction_mems[i])
            else:
                spk = direction_feat

            direction_spikes.append(spk)

        # Combine directions
        combined_directions = torch.cat(direction_responses, dim=1)

        # Opponent motion processing (depthwise-separable)
        opponent = self.opponent_motion_dw(combined_directions)
        opponent = self.opponent_motion_pw(opponent)

        # Speed tuning
        speed_responses = []
        for speed_layer in self.speed_layers:
            speed_feat = speed_layer(opponent)
            # Upsample to match original size
            speed_feat = F.interpolate(speed_feat, size=opponent.shape[2:],
                                      mode='bilinear', align_corners=False)
            speed_responses.append(speed_feat)

        speed_combined = torch.cat(speed_responses, dim=1)

        return {
            'mt_direction_spikes': direction_spikes,
            'mt_direction_features': combined_directions,
            'mt_motion_features': opponent,
            'mt_speed_features': speed_combined
        }

    def compute_motion_field(self, direction_spikes):
        """
        Compute motion field visualization.
        Returns: Motion vector field as numpy array
        """
        # Average over batch and channels
        motion_maps = [d[0].sum(dim=0).detach().cpu().numpy() for d in direction_spikes]

        # Compute dominant direction at each location
        stacked = np.stack(motion_maps, axis=0)  # (num_directions, H, W)
        dominant_dir = np.argmax(stacked, axis=0)
        motion_strength = np.max(stacked, axis=0)

        return dominant_dir, motion_strength


class ParietalCortex(nn.Module):
    """
    Parietal Cortex - Spatial Awareness and Action
    Integrates visual and spatial information for action planning.

    Sub-regions:
    - LIP (Lateral Intraparietal): Attention and eye movements
    - VIP (Ventral Intraparietal): Near-body space
    - AIP (Anterior Intraparietal): Grasping and manipulation

    Optimized: Lightweight depthwise-separable convolutions.
    """
    def __init__(self, input_channels, hidden_size=128, device='cpu'):
        super().__init__()
        self.device = device

        # Lightweight spatial attention (LIP-like, depthwise-separable)
        self.attention_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                      padding=1, bias=False, groups=input_channels)
        self.attention_pw = nn.Conv2d(input_channels, hidden_size // 4, kernel_size=1, bias=False)
        self.attention_bn = nn.BatchNorm2d(hidden_size // 4)
        self.attention_final = nn.Conv2d(hidden_size // 4, 1, kernel_size=1)

        # Lightweight action representation (AIP-like)
        self.action_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                   padding=1, bias=False, groups=input_channels)
        self.action_pw = nn.Conv2d(input_channels, hidden_size, kernel_size=1, bias=False)
        self.action_bn = nn.BatchNorm2d(hidden_size)
        self.action_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight spatial encoding (depthwise-separable)
        self.spatial_dw = nn.Conv2d(input_channels + 1, input_channels + 1, kernel_size=3,
                                    padding=1, bias=False, groups=input_channels + 1)
        self.spatial_pw = nn.Conv2d(input_channels + 1, hidden_size, kernel_size=1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(hidden_size)

        # Spiking output
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.7
            self.lif = snn.Leaky(beta=self.beta, threshold=1.0,
                                spike_grad=surrogate.fast_sigmoid())
        elif SNN_BACKEND == 'norse':
            self.lif = norse.LIFCell()

        self.mem = None

        self.to(device)

    def reset_states(self):
        self.mem = None

    def forward(self, mt_outputs):
        """
        Process MT outputs into spatial and action representations.

        Args:
            mt_outputs: Dictionary from V5/MT

        Returns:
            Parietal representations for spatial awareness and action
        """
        # Combine motion and speed features
        motion_features = mt_outputs['mt_motion_features']
        speed_features = mt_outputs['mt_speed_features']

        combined = torch.cat([motion_features, speed_features], dim=1)

        # Compute spatial attention map (depthwise-separable)
        x = self.attention_dw(combined)
        x = self.attention_pw(x)
        x = self.attention_bn(x)
        x = F.relu(x)
        attention_map = self.attention_final(x)
        attention_map = torch.sigmoid(attention_map)

        # Action encoding (depthwise-separable)
        x = self.action_dw(combined)
        x = self.action_pw(x)
        x = self.action_bn(x)
        x = F.relu(x)
        action_vector = self.action_pool(x)
        action_vector = action_vector.squeeze(-1).squeeze(-1)  # (B, hidden_size)

        # Spatial encoding with attention (depthwise-separable)
        spatial_input = torch.cat([combined, attention_map], dim=1)
        x = self.spatial_dw(spatial_input)
        x = self.spatial_pw(x)
        x = self.spatial_bn(x)
        spatial_features = F.relu(x)

        # Spiking output
        if SNN_BACKEND == 'snntorch':
            spikes, self.mem = self.lif(spatial_features, self.mem)
        elif SNN_BACKEND == 'norse':
            spikes, self.mem = self.lif(spatial_features, self.mem)
        else:
            spikes = F.relu(spatial_features)

        return {
            'parietal_spikes': spikes,
            'attention_map': attention_map,
            'action_vector': action_vector,
            'spatial_features': spatial_features
        }


class DorsalStream(nn.Module):
    """
    Complete Dorsal Stream: V3 → V5/MT → Parietal
    "WHERE/HOW" pathway for spatial processing and action.

    Optimized: Reduced dimensions throughout - target <500K params instead of 18.37M.
    """
    def __init__(self, v2_channels=64, num_directions=8, device='cpu'):
        super().__init__()
        self.device = device

        # V3: Dynamic form (lightweight)
        self.v3 = V3(input_channels=v2_channels, hidden_size=64, device=device)

        # V5/MT: Motion (lightweight)
        self.mt = V5_MT(input_channels=64, num_directions=num_directions,
                        hidden_size=64, device=device)

        # Parietal: Spatial awareness and action (lightweight)
        mt_output_channels = 64 + 64  # motion + speed
        self.parietal = ParietalCortex(input_channels=mt_output_channels,
                                       hidden_size=128, device=device)

        # Logging
        self.frame_count = 0

    def reset_states(self):
        """Reset all internal states"""
        self.v3.reset_states()
        self.mt.reset_states()
        self.parietal.reset_states()
        self.frame_count = 0

    def forward(self, v2_outputs):
        """
        Process through complete dorsal stream.

        Args:
            v2_outputs: Dictionary from V2

        Returns:
            Complete dorsal stream outputs including spatial and action info
        """
        # Extract V2 spikes
        v2_features = v2_outputs['v2_spikes']

        # V3: Dynamic form
        v3_outputs = self.v3(v2_features)

        # V5/MT: Motion analysis
        mt_outputs = self.mt(v3_outputs)

        # Parietal: Spatial and action
        parietal_outputs = self.parietal(mt_outputs)

        self.frame_count += 1

        return {
            'v3': v3_outputs,
            'mt': mt_outputs,
            'parietal': parietal_outputs,
            'frame_count': self.frame_count
        }

    def get_motion_summary(self, outputs):
        """
        Extract human-readable motion summary.
        """
        # Get dominant motion direction
        direction_spikes = outputs['mt']['mt_direction_spikes']
        total_activity = [d.sum().item() for d in direction_spikes]
        dominant_direction = np.argmax(total_activity)

        # Direction names
        direction_names = ['Right', 'Down-Right', 'Down', 'Down-Left',
                          'Left', 'Up-Left', 'Up', 'Up-Right']

        return {
            'dominant_direction': direction_names[dominant_direction % 8],
            'direction_strengths': total_activity,
            'attention_peak': outputs['parietal']['attention_map'].max().item()
        }

    def visualize_attention(self, parietal_outputs, output_size=(640, 480)):
        """
        Visualize the attention map from parietal cortex.
        """
        attention = parietal_outputs['attention_map'][0, 0].detach().cpu().numpy()

        # Normalize
        attention = (attention * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(attention, cv2.COLORMAP_HOT)

        # Resize
        resized = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def visualize_motion_field(self, mt_outputs, output_size=(640, 480)):
        """
        Visualize motion direction field.
        """
        direction_spikes = mt_outputs['mt_direction_spikes']
        dominant_dir, strength = self.mt.compute_motion_field(direction_spikes)

        # Create HSV visualization
        h, w = dominant_dir.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue = direction, Value = strength
        hsv[:, :, 0] = (dominant_dir * (180 // len(direction_spikes))).astype(np.uint8)
        hsv[:, :, 1] = 255

        # Normalize strength
        if strength.max() > 0:
            hsv[:, :, 2] = (strength / strength.max() * 255).astype(np.uint8)

        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Resize
        resized = cv2.resize(bgr, output_size, interpolation=cv2.INTER_NEAREST)

        return resized

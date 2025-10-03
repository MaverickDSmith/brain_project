"""
Ventral Stream - "WHAT" Pathway
================================
Processes object identity, form, and recognition.

Path: V1 0> V2 -> V4 -> IT (Inferotemporal Cortex)

Key functions:
- Color processing (V4)
- Shape and form recognition
- Object categorization
- Invariant object representation
- Visual memory

Output: Object identity, semantic features, recognition signals
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


class V4(nn.Module):
    """
    V4 - Color and Form Processing
    Processes complex shapes, curvature, and color combinations.
    Bridge between early vision and object recognition.

    Optimized: Lightweight depthwise-separable convolutions.
    """
    def __init__(self, input_channels=64, hidden_size=128, device='cpu'):
        super().__init__()
        self.device = device

        # Lightweight color processing (depthwise-separable)
        self.color_dw1 = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                   padding=1, bias=False, groups=input_channels)
        self.color_pw1 = nn.Conv2d(input_channels, hidden_size // 2, kernel_size=1, bias=False)
        self.color_bn1 = nn.BatchNorm2d(hidden_size // 2)

        # Lightweight shape processing (depthwise-separable)
        self.shape_dw1 = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                   padding=1, bias=False, groups=input_channels)
        self.shape_pw1 = nn.Conv2d(input_channels, hidden_size // 2, kernel_size=1, bias=False)
        self.shape_bn1 = nn.BatchNorm2d(hidden_size // 2)

        # Lightweight integration (depthwise-separable)
        self.integration_dw = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                        padding=1, bias=False, groups=hidden_size)
        self.integration_pw = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.integration_bn = nn.BatchNorm2d(hidden_size)

        # Spiking neurons
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.8
            self.lif_color = snn.Leaky(beta=self.beta, threshold=1.0,
                                      spike_grad=surrogate.fast_sigmoid())
            self.lif_shape = snn.Leaky(beta=self.beta, threshold=1.0,
                                      spike_grad=surrogate.fast_sigmoid())
            self.lif_combined = snn.Leaky(beta=self.beta, threshold=1.0,
                                         spike_grad=surrogate.fast_sigmoid())
        elif SNN_BACKEND == 'norse':
            self.lif_color = norse.LIFCell()
            self.lif_shape = norse.LIFCell()
            self.lif_combined = norse.LIFCell()

        self.mem_color = None
        self.mem_shape = None
        self.mem_combined = None

        self.to(device)

    def reset_states(self):
        self.mem_color = None
        self.mem_shape = None
        self.mem_combined = None

    def forward(self, v2_features):
        """
        Process V2 features into color and shape representations.

        Args:
            v2_features: Features from V2

        Returns:
            V4 color, shape, and combined features
        """
        # Color pathway (depthwise-separable)
        x = self.color_dw1(v2_features)
        x = self.color_pw1(x)
        x = self.color_bn1(x)
        color_features = F.relu(x)

        if SNN_BACKEND == 'snntorch':
            color_spikes, self.mem_color = self.lif_color(color_features, self.mem_color)
        elif SNN_BACKEND == 'norse':
            color_spikes, self.mem_color = self.lif_color(color_features, self.mem_color)
        else:
            color_spikes = color_features

        # Shape pathway (depthwise-separable)
        x = self.shape_dw1(v2_features)
        x = self.shape_pw1(x)
        x = self.shape_bn1(x)
        shape_features = F.relu(x)

        if SNN_BACKEND == 'snntorch':
            shape_spikes, self.mem_shape = self.lif_shape(shape_features, self.mem_shape)
        elif SNN_BACKEND == 'norse':
            shape_spikes, self.mem_shape = self.lif_shape(shape_features, self.mem_shape)
        else:
            shape_spikes = shape_features

        # Combine color and shape (depthwise-separable)
        combined = torch.cat([color_spikes, shape_spikes], dim=1)
        x = self.integration_dw(combined)
        x = self.integration_pw(x)
        x = self.integration_bn(x)
        integrated = F.relu(x)

        if SNN_BACKEND == 'snntorch':
            combined_spikes, self.mem_combined = self.lif_combined(integrated, self.mem_combined)
        elif SNN_BACKEND == 'norse':
            combined_spikes, self.mem_combined = self.lif_combined(integrated, self.mem_combined)
        else:
            combined_spikes = integrated

        return {
            'v4_color_spikes': color_spikes,
            'v4_shape_spikes': shape_spikes,
            'v4_combined_spikes': combined_spikes,
            'v4_features': integrated
        }


class InferotemporalCortex(nn.Module):
    """
    IT (Inferotemporal Cortex) - High-Level Object Recognition
    Contains neurons selective for complex objects (faces, objects, etc.).
    Provides invariant object representations.

    Optimized: Reduced categories and lightweight depthwise-separable convolutions.
    """
    def __init__(self, input_channels=128, num_object_categories=32,
                 hidden_size=256, device='cpu'):
        super().__init__()
        self.device = device
        self.num_object_categories = num_object_categories

        # Lightweight hierarchical feature extraction (depthwise-separable)
        self.conv1_dw = nn.Conv2d(input_channels, input_channels, kernel_size=3,
                                  padding=1, bias=False, groups=input_channels)
        self.conv1_pw = nn.Conv2d(input_channels, hidden_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_dw = nn.Conv2d(hidden_size, hidden_size, kernel_size=3,
                                  padding=1, bias=False, groups=hidden_size)
        self.conv2_pw = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.pool2 = nn.MaxPool2d(2)

        # Lightweight object category neurons (1x1 convs only)
        self.category_layers = nn.ModuleList([
            nn.Conv2d(hidden_size, hidden_size // num_object_categories,
                     kernel_size=1, bias=False)
            for _ in range(num_object_categories)
        ])

        # Global object representation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Lightweight object embedding
        self.object_embedding = nn.Linear(hidden_size, 128)

        # Spiking neurons for each category
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.85
            self.category_lifs = nn.ModuleList([
                snn.Leaky(beta=self.beta, threshold=1.0,
                         spike_grad=surrogate.fast_sigmoid())
                for _ in range(num_object_categories)
            ])
        elif SNN_BACKEND == 'norse':
            self.category_lifs = nn.ModuleList([
                norse.LIFCell() for _ in range(num_object_categories)
            ])

        self.category_mems = [None] * num_object_categories

        # For tracking what objects are detected
        self.category_names = [f"Category_{i}" for i in range(num_object_categories)]

        self.to(device)

    def reset_states(self):
        self.category_mems = [None] * self.num_object_categories

    def forward(self, v4_outputs):
        """
        Recognize objects from V4 representations.

        Args:
            v4_outputs: Dictionary from V4

        Returns:
            Object recognition outputs and embeddings
        """
        v4_features = v4_outputs['v4_features']

        # Hierarchical processing (depthwise-separable)
        x = self.conv1_dw(v4_features)
        x = self.conv1_pw(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2_dw(x)
        x = self.conv2_pw(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Category-specific processing
        category_responses = []
        category_spikes = []

        for i, (cat_layer, lif) in enumerate(zip(self.category_layers, self.category_lifs)):
            # Process for this category
            cat_features = cat_layer(x)

            # Global pooling for category
            cat_pooled = self.global_pool(cat_features).squeeze(-1).squeeze(-1)

            # Spiking response
            if SNN_BACKEND == 'snntorch':
                # Reshape for LIF (needs spatial dimensions)
                cat_spatial = cat_features
                spk, self.category_mems[i] = lif(cat_spatial, self.category_mems[i])
                category_spikes.append(spk)
            elif SNN_BACKEND == 'norse':
                cat_spatial = cat_features
                spk, self.category_mems[i] = lif(cat_spatial, self.category_mems[i])
                category_spikes.append(spk)
            else:
                category_spikes.append(cat_features)

            category_responses.append(cat_pooled)

        # Stack category responses
        category_vector = torch.stack(category_responses, dim=1)  # (B, num_categories, channels)

        # Global object embedding (invariant representation)
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)
        object_embedding = self.object_embedding(pooled)

        # Compute category activations
        category_activations = category_vector.mean(dim=-1)  # (B, num_categories)

        return {
            'it_category_spikes': category_spikes,
            'it_category_activations': category_activations,
            'it_object_embedding': object_embedding,
            'it_features': x
        }

    def get_top_categories(self, it_outputs, top_k=5):
        """
        Get the top-k activated object categories.
        """
        activations = it_outputs['it_category_activations'][0]  # First batch
        top_values, top_indices = torch.topk(activations, k=min(top_k, len(activations)))

        results = []
        for val, idx in zip(top_values, top_indices):
            results.append({
                'category': self.category_names[idx.item()],
                'activation': val.item(),
                'index': idx.item()
            })

        return results


class VentralStream(nn.Module):
    """
    Complete Ventral Stream: V4 → IT
    "WHAT" pathway for object recognition and identification.

    Optimized: Reduced dimensions throughout - target <500K params instead of 22.55M.
    """
    def __init__(self, v2_channels=64, num_object_categories=32, device='cpu'):
        super().__init__()
        self.device = device

        # V4: Color and form (lightweight)
        self.v4 = V4(input_channels=v2_channels, hidden_size=128, device=device)

        # IT: Object recognition (lightweight)
        self.it = InferotemporalCortex(
            input_channels=128,
            num_object_categories=num_object_categories,
            hidden_size=256,
            device=device
        )

        # Logging
        self.frame_count = 0
        self.recognition_history = []

    def reset_states(self):
        """Reset all internal states"""
        self.v4.reset_states()
        self.it.reset_states()
        self.frame_count = 0
        self.recognition_history = []

    def forward(self, v2_outputs):
        """
        Process through complete ventral stream.

        Args:
            v2_outputs: Dictionary from V2

        Returns:
            Complete ventral stream outputs including object recognition
        """
        # Extract V2 features
        v2_features = v2_outputs['v2_spikes']

        # V4: Color and form
        v4_outputs = self.v4(v2_features)

        # IT: Object recognition
        it_outputs = self.it(v4_outputs)

        self.frame_count += 1

        # Track recognition over time
        top_categories = self.it.get_top_categories(it_outputs, top_k=3)
        self.recognition_history.append(top_categories)

        return {
            'v4': v4_outputs,
            'it': it_outputs,
            'frame_count': self.frame_count,
            'top_categories': top_categories
        }

    def get_object_summary(self, outputs):
        """
        Get human-readable object recognition summary.
        """
        top_cats = outputs['top_categories']

        summary = {
            'detected_objects': [cat['category'] for cat in top_cats],
            'confidence': [cat['activation'] for cat in top_cats],
            'object_embedding_norm': outputs['it']['it_object_embedding'].norm().item()
        }

        return summary

    def visualize_color_features(self, v4_outputs, output_size=(640, 480)):
        """
        Visualize V4 color feature activations.
        """
        color_spikes = v4_outputs['v4_color_spikes'][0]  # First batch

        # Sum across channels
        activation_map = color_spikes.sum(dim=0).detach().cpu().numpy()

        # Normalize
        if activation_map.max() > 0:
            activation_map = (activation_map / activation_map.max() * 255).astype(np.uint8)
        else:
            activation_map = np.zeros_like(activation_map, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(activation_map, cv2.COLORMAP_RAINBOW)

        # Resize
        resized = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def visualize_shape_features(self, v4_outputs, output_size=(640, 480)):
        """
        Visualize V4 shape feature activations.
        """
        shape_spikes = v4_outputs['v4_shape_spikes'][0]  # First batch

        # Sum across channels
        activation_map = shape_spikes.sum(dim=0).detach().cpu().numpy()

        # Normalize
        if activation_map.max() > 0:
            activation_map = (activation_map / activation_map.max() * 255).astype(np.uint8)
        else:
            activation_map = np.zeros_like(activation_map, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(activation_map, cv2.COLORMAP_VIRIDIS)

        # Resize
        resized = cv2.resize(colored, output_size, interpolation=cv2.INTER_LINEAR)

        return resized

    def visualize_category_activations(self, it_outputs):
        """
        Create a bar chart visualization of category activations.
        """
        activations = it_outputs['it_category_activations'][0].detach().cpu().numpy()

        # Create image for bar chart
        height = 400
        width = 800
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Normalize activations
        if activations.max() > 0:
            normalized = activations / activations.max()
        else:
            normalized = activations

        # Draw bars
        num_cats = len(activations)
        bar_width = width // num_cats
        max_bar_height = height - 50

        for i, val in enumerate(normalized):
            x1 = i * bar_width
            x2 = x1 + bar_width - 2
            bar_height = int(val * max_bar_height)
            y1 = height - bar_height
            y2 = height

            # Color based on activation
            color = (0, int(val * 255), int((1 - val) * 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        return img

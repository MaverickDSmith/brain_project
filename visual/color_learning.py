"""
Color Learning Module - SNN-based Color Classification
=======================================================
Lightweight spiking neural network for learning and classifying colors.
Biologically inspired by color-selective neurons in V4 and IT.

This module allows the visual cortex to learn color representations through
STDP (Spike-Timing-Dependent Plasticity) and classify colors based on
the learned representations.
"""

import torch
import torch.nn as nn
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


class ColorLearningSNN(nn.Module):
    """
    Lightweight SNN for color learning and classification.

    Uses STDP to learn color representations from visual input.
    Neurons become selective to specific colors over time.

    Biologically inspired by:
    - Color-selective neurons in V4
    - Object-selective neurons in IT
    - Plasticity mechanisms in visual cortex
    """

    def __init__(self, num_hue_bins=12, num_neurons=64, device='cpu'):
        super().__init__()
        self.num_hue_bins = num_hue_bins
        self.num_neurons = num_neurons
        self.device = device

        # Input: hue histogram + saturation + value
        self.input_size = num_hue_bins + 2

        # Spiking neuron layer
        self.fc = nn.Linear(self.input_size, num_neurons, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

        # LIF neurons
        if SNN_BACKEND == 'snntorch':
            self.beta = 0.9
            self.lif = snn.Leaky(beta=self.beta, threshold=1.0,
                                spike_grad=surrogate.fast_sigmoid())
        elif SNN_BACKEND == 'norse':
            self.lif = norse.LIFCell()

        # Learning parameters
        self.learning_rate = 0.03
        self.saturation_threshold = 0.95
        self.neuron_locked = torch.zeros(num_neurons, dtype=torch.bool, device=device)

        # State
        self.mem = None
        self.learning_enabled = True

        self.to(device)

    def reset_states(self):
        """Reset membrane potentials"""
        self.mem = None

    def extract_color_features(self, frame_hsv):
        """
        Extract color features from HSV frame.

        Args:
            frame_hsv: HSV image (H, W, 3) numpy array

        Returns:
            Feature vector with hue histogram + saturation + value
        """
        # Compute hue histogram
        hue = frame_hsv[:, :, 0]
        sat = frame_hsv[:, :, 1]
        val = frame_hsv[:, :, 2]

        # Only consider pixels with sufficient saturation and value
        mask = (sat > 30) & (val > 30)

        if mask.sum() > 0:
            valid_hues = hue[mask]
            hist, _ = np.histogram(valid_hues, bins=self.num_hue_bins,
                                  range=(0, 180), density=True)
            avg_sat = sat[mask].mean() / 255.0
            avg_val = val[mask].mean() / 255.0
        else:
            hist = np.zeros(self.num_hue_bins)
            avg_sat = 0.0
            avg_val = 0.0

        # Combine features
        features = np.concatenate([hist, [avg_sat, avg_val]])
        return features.astype(np.float32)

    def forward(self, features_np):
        """
        Process color features through SNN.

        Args:
            features_np: Color features (numpy array or tensor)

        Returns:
            spikes: Neuron activation spikes
            mem: Membrane potentials
        """
        # Convert to tensor if needed
        if isinstance(features_np, np.ndarray):
            features = torch.from_numpy(features_np).to(self.device)
        else:
            features = features_np.to(self.device)

        # Add batch dimension if needed
        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            # Linear projection
            x = self.fc(features)

            # Spiking neurons
            if SNN_BACKEND == 'snntorch':
                spikes, self.mem = self.lif(x, self.mem)
            elif SNN_BACKEND == 'norse':
                if self.mem is None:
                    self.mem = None  # Norse manages state internally
                spikes, self.mem = self.lif(x, self.mem)
            else:
                # Simple threshold
                if self.mem is None:
                    self.mem = torch.zeros_like(x)
                self.mem = self.mem + torch.tanh(x)
                spikes = (self.mem > 1.0).float()
                self.mem = self.mem * (1.0 - spikes)

        return spikes, self.mem

    def learn(self, features, spikes):
        """
        Update weights using STDP.

        Args:
            features: Input features (tensor)
            spikes: Output spikes (tensor)
        """
        if not self.learning_enabled:
            return

        with torch.no_grad():
            # Ensure correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            if spikes.dim() == 1:
                spikes = spikes.unsqueeze(0)

            # Find winner neuron(s)
            activations = features @ self.fc.weight.t()
            winners = torch.argmax(activations, dim=1)

            # Check for saturation (lock neurons that have learned)
            max_weights = torch.max(self.fc.weight.data, dim=1)[0]
            newly_locked = (max_weights >= self.saturation_threshold) & (~self.neuron_locked)
            self.neuron_locked = self.neuron_locked | newly_locked

            # STDP update
            batch_size = features.shape[0]
            feature_dim = features.shape[1]

            # Potentiation (Hebbian): strengthen weights for winner neurons
            post_diag = spikes[torch.arange(batch_size, device=self.device), winners]
            contrib = features * post_diag.unsqueeze(1)

            dw = torch.zeros_like(self.fc.weight.data)
            idx = winners.unsqueeze(1).expand(-1, feature_dim)
            dw = dw.scatter_add_(0, idx, contrib)

            # Normalize by number of updates per neuron
            counts = torch.zeros((self.fc.weight.shape[0],), device=self.device)
            counts = counts.scatter_add_(0, winners, torch.ones_like(winners, dtype=torch.float32))
            counts = counts.clamp_min(1.0).unsqueeze(1)
            dw = dw / counts

            # Depression: weaken non-winner connections slightly
            depression = 0.01 * (spikes.t() @ (1.0 - features)) / max(1.0, float(batch_size))

            # Apply updates only to unlocked neurons
            not_locked_mask = (~self.neuron_locked).float().unsqueeze(1)
            delta = self.learning_rate * (dw - depression) * not_locked_mask

            self.fc.weight.data.add_(delta)
            self.fc.weight.data.clamp_(0.0, 1.0)

    def classify_color(self, features_np, top_k=3):
        """
        Classify color based on learned representations.

        Args:
            features_np: Color features
            top_k: Number of top neurons to return

        Returns:
            Dictionary with neuron indices, activations, and preferred colors
        """
        # Get spikes
        spikes, _ = self.forward(features_np)

        # Get activations
        features = torch.from_numpy(features_np).to(self.device)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        with torch.no_grad():
            activations = features @ self.fc.weight.t()
            activations = activations[0].cpu().numpy()

        # Get top neurons
        top_indices = np.argsort(-activations)[:top_k]
        top_activations = activations[top_indices]

        # Get preferred colors for top neurons
        weights_np = self.fc.weight.detach().cpu().numpy()
        hue_weights = weights_np[:, :self.num_hue_bins]

        preferred_colors = []
        for idx in top_indices:
            # Find peak hue bin
            pref_bin = np.argmax(hue_weights[idx])
            hue = int((pref_bin + 0.5) * (180.0 / self.num_hue_bins)) % 180

            # Convert to BGR
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            preferred_colors.append(tuple(int(c) for c in bgr))

        return {
            'neuron_indices': top_indices,
            'activations': top_activations,
            'preferred_colors': preferred_colors,
            'spikes': spikes[0].cpu().numpy()
        }

    def get_neuron_colors(self):
        """
        Get the preferred color for each neuron.

        Returns:
            List of (BGR color, confidence) tuples
        """
        weights_np = self.fc.weight.detach().cpu().numpy()
        hue_weights = weights_np[:, :self.num_hue_bins]

        neuron_colors = []
        for neuron_idx in range(self.num_neurons):
            w = hue_weights[neuron_idx]
            pref_bin = np.argmax(w)
            confidence = np.max(w)

            # Convert to BGR
            hue = int((pref_bin + 0.5) * (180.0 / self.num_hue_bins)) % 180
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            color = tuple(int(c) for c in bgr)

            neuron_colors.append((color, confidence))

        return neuron_colors

    def get_memory_usage(self):
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0


def build_color_wheel(size=256):
    """
    Build HSV color wheel for visualization and interaction.

    Args:
        size: Diameter of color wheel in pixels

    Returns:
        (bgr_wheel, hsv_wheel): Color wheels in BGR and HSV formats
    """
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r_max = size // 2

    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            r = np.sqrt(dx**2 + dy**2)

            if r > r_max:
                # Outside circle: white
                hsv[y, x, 0] = 0
                hsv[y, x, 1] = 0
                hsv[y, x, 2] = 255
            else:
                # Inside circle: full saturation color wheel
                angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
                hsv[y, x, 0] = int(angle * 180)
                hsv[y, x, 1] = int(r / r_max * 255)
                hsv[y, x, 2] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr, hsv

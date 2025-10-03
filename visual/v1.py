"""
V1 (Primary Visual Cortex) - Striate Cortex
============================================
First cortical processing stage for visual information.

Key functions:
- Orientation selectivity (edge detection at different angles)
- Direction selectivity (motion direction)
- Spatial frequency analysis
- Simple and complex cell responses
- Binocular integration (depth - not implemented yet)

Input: Ganglion cell spikes from retina (ON/OFF channels)
Output: Orientation and direction feature maps
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
        print("Warning: No SNN library found. Install snntorch or norse.")


class GaborFilterBank:
    """
    Gabor filters model V1 simple cells' receptive fields.
    They respond to oriented edges at specific spatial frequencies.
    """
    def __init__(self, num_orientations=8, num_scales=4, ksize=31):
        self.num_orientations = num_orientations
        self.num_scales = num_scales
        self.ksize = ksize
        self.filters = self._build_filters()

    def _build_filters(self):
        """Build a bank of Gabor filters at different orientations and scales"""
        filters = []
        orientations = np.linspace(0, np.pi, self.num_orientations, endpoint=False)

        for scale_idx in range(self.num_scales):
            # Spatial frequency decreases with scale
            wavelength = 4.0 * (2 ** scale_idx)
            sigma = wavelength * 0.5
            gamma = 0.5  # Spatial aspect ratio

            for orientation in orientations:
                # Create Gabor kernel
                kernel = cv2.getGaborKernel(
                    (self.ksize, self.ksize),
                    sigma=sigma,
                    theta=orientation,
                    lambd=wavelength,
                    gamma=gamma,
                    psi=0,  # Phase offset
                    ktype=cv2.CV_32F
                )
                filters.append({
                    'kernel': kernel,
                    'orientation': orientation,
                    'scale': scale_idx,
                    'wavelength': wavelength
                })

        return filters

    def apply_filters(self, image):
        """
        Apply all Gabor filters to image.
        Returns: List of filtered responses (one per filter)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        responses = []
        for filt in self.filters:
            response = cv2.filter2D(image, cv2.CV_32F, filt['kernel'])
            responses.append(response)

        return responses


class V1SimpleCells(nn.Module):
    """
    V1 Simple Cells: Orientation-selective spiking neurons.
    These respond to oriented edges in specific locations (position-specific).

    Optimized for efficiency: Uses depthwise-separable convolutions and smaller feature dims.
    """
    def __init__(self, input_channels, num_orientations=8, hidden_size=32, device='cpu'):
        super().__init__()
        self.num_orientations = num_orientations
        self.hidden_size = hidden_size
        self.device = device

        if SNN_BACKEND == 'snntorch':
            self.beta = 0.9  # Decay rate
            # Lightweight spatial processing with depthwise-separable conv
            self.conv = nn.Conv2d(input_channels, hidden_size, kernel_size=5,
                                  padding=2, bias=False, groups=input_channels)
            self.conv_1x1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)

            # Orientation-specific processing (lightweight)
            self.orientation_layers = nn.ModuleList([
                nn.Conv2d(hidden_size, hidden_size // num_orientations,
                         kernel_size=3, padding=1, bias=False,
                         groups=hidden_size // num_orientations)
                for _ in range(num_orientations)
            ])
            # LIF neurons
            self.lif_neurons = nn.ModuleList([
                snn.Leaky(beta=self.beta, threshold=1.0,
                         spike_grad=surrogate.fast_sigmoid())
                for _ in range(num_orientations)
            ])
        elif SNN_BACKEND == 'norse':
            self.conv = nn.Conv2d(input_channels, hidden_size, kernel_size=5,
                                  padding=2, bias=False, groups=input_channels)
            self.conv_1x1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=False)
            self.orientation_layers = nn.ModuleList([
                nn.Conv2d(hidden_size, hidden_size // num_orientations,
                         kernel_size=3, padding=1, bias=False,
                         groups=hidden_size // num_orientations)
                for _ in range(num_orientations)
            ])
            self.lif_neurons = nn.ModuleList([
                norse.LIFCell() for _ in range(num_orientations)
            ])

        self.to(device)

    def init_states(self, batch_size, height, width):
        """Initialize membrane potentials for LIF neurons"""
        states = []
        for i in range(self.num_orientations):
            channels = self.hidden_size // self.num_orientations
            if SNN_BACKEND == 'snntorch':
                mem = torch.zeros(batch_size, channels, height, width, device=self.device)
                states.append(mem)
            elif SNN_BACKEND == 'norse':
                states.append(None)  # Norse handles state internally
        return states

    def forward(self, x, states=None):
        """
        Args:
            x: Input tensor (batch, channels, height, width)
            states: List of membrane states per orientation
        Returns:
            spikes: List of spike tensors per orientation
            states: Updated membrane states
        """
        # Shared convolutional processing (depthwise-separable)
        x = self.conv(x)
        x = self.conv_1x1(x)
        x = torch.relu(x)

        # Orientation-specific processing
        spikes = []
        new_states = []

        for i, (orient_layer, lif) in enumerate(zip(self.orientation_layers, self.lif_neurons)):
            # Apply orientation-specific convolution
            oriented = orient_layer(x)

            if SNN_BACKEND == 'snntorch':
                # Update LIF neurons
                spk, mem = lif(oriented, states[i] if states else None)
                spikes.append(spk)
                new_states.append(mem)
            elif SNN_BACKEND == 'norse':
                # Norse LIF
                spk, state = lif(oriented, states[i] if states else None)
                spikes.append(spk)
                new_states.append(state)

        return spikes, new_states


class V1ComplexCells(nn.Module):
    """
    V1 Complex Cells: Position-invariant orientation detectors.
    These pool over simple cells to achieve translation invariance.

    Optimized: Uses lightweight pooling with minimal parameters.
    """
    def __init__(self, num_orientations=8, simple_cell_channels=4,
                 complex_size=32, device='cpu'):
        super().__init__()
        self.num_orientations = num_orientations
        self.device = device

        # Lightweight pooling over simple cell responses
        self.pooling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(simple_cell_channels, complex_size // num_orientations,
                         kernel_size=3, stride=2, padding=1, bias=False,
                         groups=simple_cell_channels),  # Depthwise
                nn.Conv2d(complex_size // num_orientations, complex_size // num_orientations,
                         kernel_size=1, bias=False),  # Pointwise
                nn.BatchNorm2d(complex_size // num_orientations),
                nn.ReLU()
            )
            for _ in range(num_orientations)
        ])

        self.to(device)

    def forward(self, simple_spikes):
        """
        Args:
            simple_spikes: List of spike tensors from simple cells (one per orientation)
        Returns:
            complex_features: Concatenated complex cell responses
        """
        complex_responses = []

        for i, (spikes, pooler) in enumerate(zip(simple_spikes, self.pooling_layers)):
            # Pool over spatial dimensions
            pooled = pooler(spikes)
            complex_responses.append(pooled)

        # Concatenate all orientations
        complex_features = torch.cat(complex_responses, dim=1)

        return complex_features


class V1(nn.Module):
    """
    Complete V1 model combining simple and complex cells.

    Optimized for efficiency with reduced feature dimensions.
    Target: ~10-20K parameters instead of 169K.
    """
    def __init__(self, input_channels=2, num_orientations=8,
                 simple_hidden=32, complex_size=32, device='cpu'):
        super().__init__()
        self.device = device
        self.num_orientations = num_orientations

        # Gabor filter bank (not trainable, models receptive fields)
        self.gabor_bank = GaborFilterBank(num_orientations=num_orientations)

        # Simple cells (orientation-selective, lightweight)
        self.simple_cells = V1SimpleCells(
            input_channels=input_channels,
            num_orientations=num_orientations,
            hidden_size=simple_hidden,
            device=device
        )

        # Complex cells (position-invariant, lightweight)
        self.complex_cells = V1ComplexCells(
            num_orientations=num_orientations,
            simple_cell_channels=simple_hidden // num_orientations,
            complex_size=complex_size,
            device=device
        )

        self.simple_states = None
        self.frame_count = 0

        # Logging
        self.spike_history = []
        self.orientation_responses = []

    def reset_states(self):
        """Reset all internal states"""
        self.simple_states = None
        self.frame_count = 0
        self.spike_history = []
        self.orientation_responses = []

    def forward(self, retina_on_spikes, retina_off_spikes):
        """
        Process retinal ganglion cell spikes through V1.

        Args:
            retina_on_spikes: ON channel spikes (H, W) numpy or (B, 1, H, W) tensor
            retina_off_spikes: OFF channel spikes (H, W) numpy or (B, 1, H, W) tensor

        Returns:
            dict with:
                - simple_spikes: List of orientation-specific spike maps
                - complex_features: Position-invariant orientation features
                - gabor_responses: Classical Gabor filter responses
                - orientation_map: Dominant orientation per location
        """
        # Convert numpy to torch if needed
        if isinstance(retina_on_spikes, np.ndarray):
            retina_on_spikes = torch.from_numpy(retina_on_spikes.astype(np.float32) / 255.0)
            retina_on_spikes = retina_on_spikes.unsqueeze(0).unsqueeze(0).to(self.device)

        if isinstance(retina_off_spikes, np.ndarray):
            retina_off_spikes = torch.from_numpy(retina_off_spikes.astype(np.float32) / 255.0)
            retina_off_spikes = retina_off_spikes.unsqueeze(0).unsqueeze(0).to(self.device)

        # Ensure 4D tensors
        if retina_on_spikes.dim() == 2:
            retina_on_spikes = retina_on_spikes.unsqueeze(0).unsqueeze(0)
        if retina_off_spikes.dim() == 2:
            retina_off_spikes = retina_off_spikes.unsqueeze(0).unsqueeze(0)

        batch_size, _, height, width = retina_on_spikes.shape

        # Combine ON and OFF channels
        retina_input = torch.cat([retina_on_spikes, retina_off_spikes], dim=1)

        # Initialize states if needed
        if self.simple_states is None:
            self.simple_states = self.simple_cells.init_states(batch_size, height, width)

        # Simple cells (orientation-selective spiking)
        simple_spikes, self.simple_states = self.simple_cells(retina_input, self.simple_states)

        # Complex cells (position-invariant)
        complex_features = self.complex_cells(simple_spikes)

        # Skip expensive Gabor filters - only compute orientation map
        # (Gabor responses are only for visualization and are very slow)
        gabor_responses = None  # Skip expensive CPU filtering

        # Compute orientation map (dominant orientation at each location)
        orientation_map = self._compute_orientation_map(simple_spikes)

        # Logging
        self.frame_count += 1
        total_spikes = sum([s.sum().item() for s in simple_spikes])
        self.spike_history.append(total_spikes)

        # Store orientation responses for analysis
        orientation_strengths = [s.sum().item() for s in simple_spikes]
        self.orientation_responses.append(orientation_strengths)

        return {
            'simple_spikes': simple_spikes,
            'complex_features': complex_features,
            'gabor_responses': gabor_responses,
            'orientation_map': orientation_map,
            'total_spikes': total_spikes,
            'frame_count': self.frame_count
        }

    def _compute_orientation_map(self, simple_spikes):
        """
        Compute the dominant orientation at each spatial location.
        Returns: (H, W) array with orientation indices
        """
        # Stack all orientation responses
        stacked = torch.stack(simple_spikes, dim=0)  # (num_orient, B, C, H, W)

        # Sum over batch and channels
        summed = stacked.sum(dim=(1, 2))  # (num_orient, H, W)

        # Find dominant orientation
        orientation_map = torch.argmax(summed, dim=0)  # (H, W)

        return orientation_map.detach().cpu().numpy()

    def get_statistics(self):
        """Get statistics for logging/visualization"""
        stats = {
            'frame_count': self.frame_count,
            'total_spikes_history': self.spike_history,
            'avg_spikes_per_frame': np.mean(self.spike_history) if self.spike_history else 0,
            'orientation_responses': self.orientation_responses
        }
        return stats

    def visualize_orientation_map(self, orientation_map, output_size=(640, 480)):
        """
        Create a colorful visualization of the orientation map.
        Different colors represent different orientations.
        """
        h, w = orientation_map.shape

        # Create HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Map orientation index to hue (color)
        hsv[:, :, 0] = (orientation_map * (180 // self.num_orientations)).astype(np.uint8)
        hsv[:, :, 1] = 255  # Full saturation
        hsv[:, :, 2] = 200  # Brightness

        # Convert to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Resize for display
        bgr_resized = cv2.resize(bgr, output_size, interpolation=cv2.INTER_NEAREST)

        return bgr_resized

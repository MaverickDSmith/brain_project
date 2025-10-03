import cv2
import numpy as np

class Retina:
    def __init__(self, spike_threshold=10, fovea_radius_ratio=0.25, motion_persistence=0.2):
        """
        Initialize retina model with biologically-inspired parameters.

        Args:
            spike_threshold: Intensity threshold for ganglion cell spike generation
            fovea_radius_ratio: Size of high-resolution foveal region (center of vision)
            motion_persistence: Temporal decay factor for motion accumulation (0-1)
        """
        self.SPIKE_THRESHOLD = spike_threshold
        self.FOVEA_RADIUS_RATIO = fovea_radius_ratio
        self.motion_persistence = float(motion_persistence)

        # State tracking for temporal processing
        self.prev_bipolar_on = None
        self.prev_bipolar_off = None
        self.motion_on_accumulator = None
        self.motion_off_accumulator = None

    # ==================== PHOTORECEPTORS ====================

    def rods(self, frame):
        """
        Rod cells: Low-light, peripheral vision, grayscale only.
        Rods are highly sensitive but provide low spatial resolution.

        Returns: Grayscale blurred image (numpy uint8)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to simulate lower spatial resolution
        # Rods provide blurrier vision than cones
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)

        return blurred

    def cones(self, frame):
        """
        Cone cells: Color vision, high-acuity foveal vision.
        Cones are concentrated in the fovea (center) and provide sharp color vision.
        Peripheral vision has fewer cones, so we simulate foveal vision.

        Returns: HSV image with foveal enhancement (numpy uint8)
        """
        # Convert to HSV for better color processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, w = hsv.shape[:2]
        cx, cy = w // 2, h // 2

        # Create foveal mask (center of vision with high cone density)
        r = int(min(h, w) * self.FOVEA_RADIUS_RATIO)
        Y, X = np.ogrid[:h, :w]
        fovea_mask = (X - cx)**2 + (Y - cy)**2 <= r*r

        # Create low-resolution peripheral vision
        hsv_peripheral = cv2.resize(hsv, (w // 4, h // 4))
        hsv_peripheral = cv2.resize(hsv_peripheral, (w, h), interpolation=cv2.INTER_NEAREST)

        # Combine foveal (sharp) and peripheral (blurry) regions
        hsv_foveated = np.where(fovea_mask[..., None], hsv, hsv_peripheral)

        return hsv_foveated

    # ==================== HORIZONTAL CELLS ====================

    def horizontal_cells(self, rods_out, cones_out):
        """
        Horizontal cells: Provide lateral inhibition via gap junctions.
        They create center-surround receptive fields by inhibiting neighboring photoreceptors.
        This enhances edges and contrast.

        Returns: (rods_inhibited, cones_inhibited) both numpy uint8
        """
        # Process rod signal
        rods_blurred = cv2.GaussianBlur(rods_out, (9, 9), 2.0)
        rods_dog = cv2.subtract(rods_out, rods_blurred // 2)  # Difference of Gaussians

        # Process cone signal (use Value channel from HSV)
        if len(cones_out.shape) == 3:
            cones_val = cones_out[:, :, 2]  # V channel from HSV
        else:
            cones_val = cones_out

        cones_blurred = cv2.GaussianBlur(cones_val, (9, 9), 2.0)
        cones_dog = cv2.subtract(cones_val, cones_blurred // 2)

        return rods_dog, cones_dog

    # ==================== BIPOLAR CELLS ====================

    def bipolar_cells(self, center_signal, surround_signal):
        """
        Bipolar cells: Create ON and OFF channels.
        - ON-center bipolar cells: Excited by light in center, inhibited by surround
        - OFF-center bipolar cells: Inhibited by light in center, excited by surround

        This creates the foundational contrast detection mechanism.

        Returns: (on_channel, off_channel) both numpy uint8
        """
        # Convert to float for arithmetic
        center = center_signal.astype(np.float32)
        surround = surround_signal.astype(np.float32)

        # ON channel: center - surround (responds to light increments)
        on_response = center - surround
        on_response = np.clip(on_response, 0, 255).astype(np.uint8)

        # OFF channel: surround - center (responds to light decrements)
        off_response = surround - center
        off_response = np.clip(off_response, 0, 255).astype(np.uint8)

        return on_response, off_response

    # ==================== AMACRINE CELLS ====================

    def amacrine_cells(self, on_channel, off_channel):
        """
        Amacrine cells: Provide temporal processing and motion detection.
        They detect changes over time by comparing current frame to previous frame.

        Returns: (motion_on, motion_off) both numpy uint8
        """
        # Initialize accumulators on first frame
        if self.prev_bipolar_on is None:
            self.prev_bipolar_on = on_channel.copy()
            self.prev_bipolar_off = off_channel.copy()
            self.motion_on_accumulator = np.zeros_like(on_channel, dtype=np.float32)
            self.motion_off_accumulator = np.zeros_like(off_channel, dtype=np.float32)
            return np.zeros_like(on_channel), np.zeros_like(off_channel)

        # Compute motion (temporal derivative)
        motion_on = cv2.absdiff(on_channel, self.prev_bipolar_on).astype(np.float32)
        motion_off = cv2.absdiff(off_channel, self.prev_bipolar_off).astype(np.float32)

        # Accumulate motion with persistence (temporal integration)
        self.motion_on_accumulator = (self.motion_on_accumulator * self.motion_persistence +
                                       motion_on * (1 - self.motion_persistence))
        self.motion_off_accumulator = (self.motion_off_accumulator * self.motion_persistence +
                                        motion_off * (1 - self.motion_persistence))

        # Clip and convert back to uint8
        motion_on_output = np.clip(self.motion_on_accumulator, 0, 255).astype(np.uint8)
        motion_off_output = np.clip(self.motion_off_accumulator, 0, 255).astype(np.uint8)

        # Update previous frame
        self.prev_bipolar_on = on_channel.copy()
        self.prev_bipolar_off = off_channel.copy()

        return motion_on_output, motion_off_output

    # ==================== GANGLION CELLS ====================

    def ganglion_cells(self, motion_on, motion_off):
        """
        Ganglion cells: Final retinal output layer that generates spikes.
        They apply thresholding and wide-field inhibition, then output discrete spikes
        that travel down the optic nerve to the brain.

        Returns: (on_spikes, off_spikes) both numpy uint8 binary (0 or 255)
        """
        # Apply wide-field inhibition (global context suppression)
        # This helps with gain control and prevents saturation
        motion_on_float = motion_on.astype(np.float32)
        motion_off_float = motion_off.astype(np.float32)

        # Large-field average (ambient motion level)
        global_avg_on = cv2.GaussianBlur(motion_on_float, (31, 31), 10.0)
        global_avg_off = cv2.GaussianBlur(motion_off_float, (31, 31), 10.0)

        # Local minus global (enhances local features)
        inhibited_on = np.clip(motion_on_float - global_avg_on / 2, 0, 255)
        inhibited_off = np.clip(motion_off_float - global_avg_off / 2, 0, 255)

        # Threshold to generate spikes (binary output)
        _, on_spikes = cv2.threshold(inhibited_on.astype(np.uint8),
                                       self.SPIKE_THRESHOLD, 255, cv2.THRESH_BINARY)
        _, off_spikes = cv2.threshold(inhibited_off.astype(np.uint8),
                                        self.SPIKE_THRESHOLD, 255, cv2.THRESH_BINARY)

        return on_spikes, off_spikes

    # ==================== FORWARD PASS ====================

    def forward(self, frame):
        """
        Complete retinal processing pipeline from photoreceptors to ganglion cells.

        Process flow:
        1. Photoreceptors (rods & cones) capture light
        2. Horizontal cells provide lateral inhibition
        3. Bipolar cells create ON/OFF channels
        4. Amacrine cells detect motion
        5. Ganglion cells generate spikes for optic nerve

        Args:
            frame: Input BGR image (numpy uint8)

        Returns:
            Dictionary with all intermediate and final outputs for visualization
        """
        # Stage 1: Photoreceptor layer
        rods_out = self.rods(frame)
        cones_out = self.cones(frame)

        # Stage 2: Horizontal cell layer (lateral inhibition)
        rods_dog, cones_dog = self.horizontal_cells(rods_out, cones_out)

        # Stage 3: Bipolar cell layer (contrast detection)
        on_channel, off_channel = self.bipolar_cells(rods_dog, cones_dog)

        # Stage 4: Amacrine cell layer (motion detection)
        motion_on, motion_off = self.amacrine_cells(on_channel, off_channel)

        # Stage 5: Ganglion cell layer (spike generation)
        on_spikes, off_spikes = self.ganglion_cells(motion_on, motion_off)

        # Return all stages for visualization
        return {
            'rods_out': rods_out,
            'cones_out': cones_out,
            'rods_dog': rods_dog,
            'cones_dog': cones_dog,
            'on_channel': on_channel,
            'off_channel': off_channel,
            'motion_on': motion_on,
            'motion_off': motion_off,
            'on_spikes': on_spikes,
            'off_spikes': off_spikes
        }
import cv2
import numpy as np
import torch
import torch.nn as nn
try:
    import norse.torch as norse
    NORSE_AVAILABLE = True
except Exception:
    NORSE_AVAILABLE = False

class Retina:
    def __init__(self, spike_threshold=40, fovea_radius_ratio=0.25,
                 num_hue_bins=12, n_neurons=12, temporal_window=4,
                 ds_w=40, ds_h=30, learning_lr=0.03, device=None,
                 motion_persistence=0.7):
        self.SPIKE_THRESHOLD = spike_threshold
        self.FOVEA_RADIUS_RATIO = fovea_radius_ratio
        self.prev_bipolar_on = None
        self.prev_bipolar_off = None
        self.prev_frame_gray = None
        self.motion_persistence = motion_persistence  # 0.0 = no persistence, 0.9 = high persistence
        self.motion_on_accumulator = None
        self.motion_off_accumulator = None

        # SNN
        self.NUM_HUE_BINS = num_hue_bins
        self.N_NEURONS = n_neurons
        self.TEMPORAL_WINDOW = temporal_window
        self.DS_W = ds_w
        self.DS_H = ds_h
        self.LEARNING_LR = learning_lr
        self.DEVICE = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.snn = self.BatchLIFLayer(self.NUM_HUE_BINS, self.N_NEURONS).to(self.DEVICE)
        self.BATCH_PIXELS = self.DS_W * self.DS_H
        self.snn_state = self.snn.init_state(self.BATCH_PIXELS)
        self.learning_enabled = True
        self.temporal_buffer = [np.zeros((self.DS_H, self.DS_W, self.NUM_HUE_BINS), dtype=np.uint8)
                                for _ in range(self.TEMPORAL_WINDOW)]
        
        # Competitive learning parameters
        self.saturation_threshold = 0.95  # Lock neurons above this confidence
        self.neuron_locked = torch.zeros(self.N_NEURONS, dtype=torch.bool, device=self.DEVICE)
        
        # Cache last outputs for visualization
        self.last_outputs = None

    # ---------------- Retina layers ----------------
    def rods(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (15, 15), 0)

    def cones(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * self.FOVEA_RADIUS_RATIO)
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx)**2 + (Y - cy)**2 <= r*r
        hsv_lowres = cv2.resize(hsv, (w//4, h//4))
        hsv_lowres = cv2.resize(hsv_lowres, (w, h), interpolation=cv2.INTER_NEAREST)
        hsv_foveated = np.where(mask[..., None], hsv, hsv_lowres)
        return hsv_foveated

    def horizontal_layer(self, rods_out, cones_out):
        rods_dog = cv2.GaussianBlur(rods_out, (9,9), 2) - cv2.GaussianBlur(rods_out, (21,21), 4)
        cones_val = cones_out[:,:,2]
        cones_dog = cv2.GaussianBlur(cones_val, (9,9), 2) - cv2.GaussianBlur(cones_val, (21,21), 4)
        return cv2.convertScaleAbs(rods_dog), cv2.convertScaleAbs(cones_dog)

    def bipolar_layer(self, rods_dog, cones_dog):
        combined = cv2.addWeighted(rods_dog, 0.5, cones_dog, 0.5, 0)
        on_channel = cv2.threshold(combined, 128, 255, cv2.THRESH_BINARY)[1]
        off_channel = cv2.threshold(255 - combined, 128, 255, cv2.THRESH_BINARY)[1]
        return on_channel, off_channel

    def amacrine_layer(self, on_channel, off_channel):
        if self.prev_bipolar_on is None or self.prev_bipolar_off is None:
            self.prev_bipolar_on, self.prev_bipolar_off = on_channel.copy(), off_channel.copy()
            self.motion_on_accumulator = np.zeros_like(on_channel, dtype=np.float32)
            self.motion_off_accumulator = np.zeros_like(off_channel, dtype=np.float32)
            return np.zeros_like(on_channel), np.zeros_like(off_channel)
        
        # Detect motion
        motion_on = cv2.absdiff(on_channel, self.prev_bipolar_on).astype(np.float32)
        motion_off = cv2.absdiff(off_channel, self.prev_bipolar_off).astype(np.float32)
        
        # Apply temporal persistence (exponential decay)
        self.motion_on_accumulator = self.motion_on_accumulator * self.motion_persistence + motion_on
        self.motion_off_accumulator = self.motion_off_accumulator * self.motion_persistence + motion_off
        
        # Clip to valid range
        motion_on_persistent = np.clip(self.motion_on_accumulator, 0, 255).astype(np.uint8)
        motion_off_persistent = np.clip(self.motion_off_accumulator, 0, 255).astype(np.uint8)
        
        self.prev_bipolar_on, self.prev_bipolar_off = on_channel.copy(), off_channel.copy()
        return motion_on_persistent, motion_off_persistent

    def ganglion_layer(self, motion_on, motion_off):
        on_spikes = cv2.threshold(motion_on, self.SPIKE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        off_spikes = cv2.threshold(motion_off, self.SPIKE_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        return on_spikes, off_spikes

    def direction_selective(self, frame_gray):
        if self.prev_frame_gray is None:
            self.prev_frame_gray = frame_gray.copy()
            return np.zeros_like(frame_gray), np.zeros_like(frame_gray)
        flow = cv2.calcOpticalFlowFarneback(self.prev_frame_gray, frame_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
        horiz = ((ang < 45) | (ang > 135)).astype(np.uint8) * (mag > 1).astype(np.uint8) * 255
        vert  = ((ang >= 45) & (ang <= 135)).astype(np.uint8) * (mag > 1).astype(np.uint8) * 255
        self.prev_frame_gray = frame_gray.copy()
        return horiz.astype(np.uint8), vert.astype(np.uint8)

    # ---------------- Color encoder ----------------
    def cones_hue_onehot_downsample(self, cones_hsv, sat_thresh=20, val_thresh=20):
        h_full, w_full = cones_hsv.shape[:2]
        down_hsv = cv2.resize(cones_hsv, (self.DS_W, self.DS_H), interpolation=cv2.INTER_AREA)
        h_chan = down_hsv[:,:,0].astype(np.float32)
        s_chan = down_hsv[:,:,1].astype(np.float32)
        v_chan = down_hsv[:,:,2].astype(np.float32)
        mask = (s_chan > sat_thresh) & (v_chan > val_thresh)
        bin_idx = np.floor(h_chan / (180.0 / self.NUM_HUE_BINS)).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.NUM_HUE_BINS-1)
        onehot = np.zeros((self.DS_H, self.DS_W, self.NUM_HUE_BINS), dtype=np.uint8)
        for y in range(self.DS_H):
            for x in range(self.DS_W):
                if mask[y,x]:
                    onehot[y,x,bin_idx[y,x]] = 1
        return onehot, down_hsv
    
    def hue_bin_to_bgr(self, bin_idx):
        """
        Convert a hue bin index to a BGR color for visualization.
        """
        hue = int((bin_idx + 0.5) * (180.0 / self.NUM_HUE_BINS)) % 180
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr[0,0])


    # ---------------- Simple SNN layer ----------------
    class BatchLIFLayer(nn.Module):
        def __init__(self, input_size, n_neurons):
            super().__init__()
            self.fc = nn.Linear(input_size, n_neurons, bias=False)
            nn.init.xavier_uniform_(self.fc.weight)
            self.n_neurons = n_neurons
            self.use_norse = NORSE_AVAILABLE
            if self.use_norse:
                self.lif = norse.LIFCell()
        def forward(self, x_t, state):
            z = self.fc(x_t)
            if self.use_norse:
                spk, new_s = self.lif(z, state)
                return spk, new_s
            else:
                v = state['v'] if state is not None else torch.zeros(x_t.shape[0], self.n_neurons, device=x_t.device)
                v = v + torch.tanh(z)
                thr = 1.0
                spk = (v > thr).float()
                v = v * (1.0 - spk)
                return spk, {'v': v}
        def init_state(self, batch_size):
            if self.use_norse:
                return None
            else:
                return {'v': torch.zeros(batch_size, self.n_neurons, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))}

    # ---------------- STDP update with competitive learning ----------------
    def stdp_batch_update(self, weights, pre_batch, post_batch, lr=0.03, w_min=0.0, w_max=1.0):
        pre = torch.from_numpy(pre_batch.astype(np.float32)) if isinstance(pre_batch, np.ndarray) else pre_batch.float()
        pre = pre.to(self.DEVICE)
        post = post_batch.float().to(self.DEVICE)
        
        # Check which neurons have reached saturation
        max_weights = torch.max(weights, dim=1)[0]  # Max weight per neuron
        newly_locked = (max_weights >= self.saturation_threshold) & (~self.neuron_locked)
        self.neuron_locked = self.neuron_locked | newly_locked
        
        # Winner-take-all: find strongest responding neuron per input
        activations = pre @ weights.t()  # (batch, n_neurons)
        winners = torch.argmax(activations, dim=1)  # (batch,)
        
        # Create learning mask: only update if neuron is winner AND not locked
        learning_mask = torch.zeros_like(weights, dtype=torch.bool)
        for batch_idx in range(pre.shape[0]):
            winner_idx = winners[batch_idx]
            if not self.neuron_locked[winner_idx]:
                learning_mask[winner_idx, :] = True
        
        # Compute Hebbian update only for winners
        dw = torch.zeros_like(weights)
        for batch_idx in range(pre.shape[0]):
            winner_idx = winners[batch_idx]
            if not self.neuron_locked[winner_idx]:
                dw[winner_idx] += post[batch_idx, winner_idx] * pre[batch_idx]
        
        dw = dw / max(1.0, pre.shape[0])
        
        # Small depression term for non-winners (lateral inhibition)
        depression = 0.01 * post.t().matmul(1.0 - pre) / max(1.0, pre.shape[0])
        depression = depression * (~self.neuron_locked.unsqueeze(1))  # Don't modify locked neurons
        
        # Apply updates only to non-locked neurons
        update_mask = (~self.neuron_locked).unsqueeze(1).float()
        weights.data += lr * (dw - depression) * update_mask
        weights.data.clamp_(w_min, w_max)
    
    def get_locked_neurons(self):
        """Return list of locked neuron indices and their preferred colors"""
        locked_indices = torch.where(self.neuron_locked)[0].cpu().numpy()
        weights = self.snn.fc.weight.detach().cpu().numpy()
        locked_info = []
        for idx in locked_indices:
            pref_bin = np.argmax(weights[idx])
            conf = np.max(weights[idx])
            color = self.hue_bin_to_bgr(pref_bin)
            locked_info.append((idx, conf, color, pref_bin))
        return locked_info

 
    def forward(self, frame, repeat_static=False):
        """
        Process a frame through the retina and SNN.

        Args:
            frame: BGR image (numpy array)
            repeat_static: if True, treat this frame as repeated over temporal window

        Returns:
            dict with keys:
                'on_spikes', 'off_spikes', 'motion_on', 'motion_off',
                'horiz_dir', 'vert_dir', 'belief', 'snn_spikes',
                'rods_out', 'cones_out', 'rods_dog', 'cones_dog',
                'on_chan', 'off_chan'
        """
        # --- Retina preprocessing ---
        rods_out = self.rods(frame)
        cones_out = self.cones(frame)
        rods_dog, cones_dog = self.horizontal_layer(rods_out, cones_out)
        on_chan, off_chan = self.bipolar_layer(rods_dog, cones_dog)

        # --- Amacrine (motion) ---
        motion_on, motion_off = self.amacrine_layer(on_chan, off_chan)
        on_spikes, off_spikes = self.ganglion_layer(motion_on, motion_off)

        # --- Direction-selective ---
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        horiz_dir, vert_dir = self.direction_selective(frame_gray)

        # --- SNN input ---
        onehot_ds, down_hsv = self.cones_hue_onehot_downsample(cones_out)
        if repeat_static:
            for t in range(self.TEMPORAL_WINDOW):
                self.temporal_buffer[t] = onehot_ds.copy()
        else:
            self.temporal_buffer.pop(0)
            self.temporal_buffer.append(onehot_ds.copy())

        seq = np.stack(self.temporal_buffer, axis=0)
        T, H_ds, W_ds, bins = seq.shape
        batch = H_ds * W_ds
        input_seq = torch.from_numpy(seq.reshape(T, batch, bins).astype(np.float32)).to(self.DEVICE)

        # --- SNN forward ---
        post_spikes_seq = []
        local_state = self.snn_state
        for t in range(T):
            x_t = input_seq[t]
            spk, local_state = self.snn(x_t, local_state)
            post_spikes_seq.append(spk)
        self.snn_state = local_state

        # --- STDP ---
        if self.learning_enabled:
            pre_batch_flat = (seq[-1] > 0.5).astype(np.uint8).reshape(-1, bins)
            post_last = post_spikes_seq[-1]
            self.stdp_batch_update(self.snn.fc.weight, pre_batch_flat, post_last, lr=self.LEARNING_LR)

        # --- Belief map ---
        weights = self.snn.fc.weight.detach().cpu().numpy()
        pre_last_flat = seq[-1].reshape(batch, bins)
        activ = pre_last_flat @ weights.T
        assigned = np.argmax(activ, axis=1)
        belief_map = assigned.reshape(H_ds, W_ds)

        outputs = {
            'on_spikes': on_spikes,
            'off_spikes': off_spikes,
            'motion_on': motion_on,
            'motion_off': motion_off,
            'horiz_dir': horiz_dir,
            'vert_dir': vert_dir,
            'belief': belief_map,
            'snn_spikes': post_spikes_seq,
            'rods_out': rods_out,
            'cones_out': cones_out,
            'rods_dog': rods_dog,
            'cones_dog': cones_dog,
            'on_chan': on_chan,
            'off_chan': off_chan
        }
        
        # Cache outputs
        self.last_outputs = outputs
        return outputs


    def visualize(self, frame, mode="belief", overlay_colors=None, repeat_static=False):
        """
        Visualize retina outputs.
        Uses cached outputs if available and not in repeat_static mode.
        """
        # Always run forward to update state
        outputs = self.forward(frame, repeat_static=repeat_static)

        if mode == "raw":
            output_img = frame.copy()
        elif mode == "rods":
            output_img = cv2.cvtColor(outputs['rods_out'], cv2.COLOR_GRAY2BGR)
        elif mode == "cones":
            output_img = cv2.cvtColor(outputs['cones_out'], cv2.COLOR_HSV2BGR)
        elif mode == "horiz_rods":
            output_img = cv2.cvtColor(outputs['rods_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == "horiz_cones":
            output_img = cv2.cvtColor(outputs['cones_dog'], cv2.COLOR_GRAY2BGR)
        elif mode == "bipolar_on":
            output_img = cv2.cvtColor(outputs['on_chan'], cv2.COLOR_GRAY2BGR)
        elif mode == "bipolar_off":
            output_img = cv2.cvtColor(outputs['off_chan'], cv2.COLOR_GRAY2BGR)
        elif mode == "amacrine_on":
            output_img = cv2.cvtColor(outputs['motion_on'], cv2.COLOR_GRAY2BGR)
        elif mode == "amacrine_off":
            output_img = cv2.cvtColor(outputs['motion_off'], cv2.COLOR_GRAY2BGR)
        elif mode == "ganglion_on":
            output_img = cv2.cvtColor(outputs['on_spikes'], cv2.COLOR_GRAY2BGR)
        elif mode == "ganglion_off":
            output_img = cv2.cvtColor(outputs['off_spikes'], cv2.COLOR_GRAY2BGR)
        elif mode == "dir_horiz":
            output_img = cv2.cvtColor(outputs['horiz_dir'], cv2.COLOR_GRAY2BGR)
        elif mode == "dir_vert":
            output_img = cv2.cvtColor(outputs['vert_dir'], cv2.COLOR_GRAY2BGR)
        elif mode == "belief":
            weights = self.snn.fc.weight.detach().cpu().numpy()
            pref_bins = np.argmax(weights, axis=1)
            assigned_map = outputs['belief']
            belief_hsv = np.zeros((self.DS_H, self.DS_W, 3), dtype=np.uint8)
            for y in range(self.DS_H):
                for x in range(self.DS_W):
                    neuron_idx = int(assigned_map[y,x])
                    bin_idx = int(pref_bins[neuron_idx])
                    hue = int((bin_idx + 0.5) * (180.0 / self.NUM_HUE_BINS)) % 180
                    belief_hsv[y,x,0] = hue
                    belief_hsv[y,x,1] = 255
                    belief_hsv[y,x,2] = 255
            output_img = cv2.cvtColor(cv2.resize(belief_hsv, (frame.shape[1], frame.shape[0]),
                                                interpolation=cv2.INTER_NEAREST), cv2.COLOR_HSV2BGR)
        else:
            output_img = frame.copy()

        # Optional overlay (clicked colors)
        if overlay_colors is not None:
            patch_h, patch_w = 40, 40
            x_start, y_start = 10, 10
            for i, col in enumerate(overlay_colors):
                x0 = x_start + i*(patch_w + 5)
                y0 = y_start
                x1, y1 = x0 + patch_w, y0 + patch_h
                cv2.rectangle(output_img, (x0,y0), (x1,y1), col, -1)
                cv2.rectangle(output_img, (x0,y0), (x1,y1), (0,0,0), 1)
            cv2.putText(output_img, "Top neuron colors", (x_start, y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        return output_img
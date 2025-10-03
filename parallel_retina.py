"""
Multiprocess Retina - Each layer runs in its own process with shared memory
"""
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Queue, Event
from collections import deque
import time

# Must be called before any CUDA operations
mp.set_start_method('spawn', force=True)

class PhotoreceptorProcess(mp.Process):
    """Rods and Cones layer"""
    def __init__(self, input_queue, output_queue, config, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        
    def run(self):
        # Each process initializes its own CUDA context
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build kernels
        ksize = self.config['gaussian_ksize_rods']
        sigma = max(0.5, float(ksize) / 3.0)
        half = ksize // 2
        x = torch.arange(-half, half+1, device=device, dtype=torch.float32)
        g = torch.exp(-(x**2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        kernel2d = g[:, None] * g[None, :]
        rods_kernel = kernel2d[None, None, :, :].to(device)
        
        print(f"PhotoreceptorProcess started on {device}")
        
        while not self.stop_event.is_set():
            try:
                frame = self.input_queue.get(timeout=0.1)
                if frame is None:
                    break
                
                # Rods processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t = torch.from_numpy(gray.astype(np.float32) / 255.0).to(device)
                t = t.unsqueeze(0).unsqueeze(0).contiguous()
                
                with torch.no_grad():
                    blurred = torch.nn.functional.conv2d(t, rods_kernel, padding=ksize//2)
                rods_out = (blurred.squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                del t, blurred
                
                # Cones processing (foveated HSV)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, w = hsv.shape[:2]
                cx, cy = w // 2, h // 2
                r = int(min(h, w) * self.config['fovea_radius_ratio'])
                Y, X = np.ogrid[:h, :w]
                mask = (X - cx)**2 + (Y - cy)**2 <= r*r
                hsv_lowres = cv2.resize(hsv, (w//4, h//4))
                hsv_lowres = cv2.resize(hsv_lowres, (w, h), interpolation=cv2.INTER_NEAREST)
                cones_out = np.where(mask[..., None], hsv, hsv_lowres)
                
                self.output_queue.put({
                    'rods': rods_out,
                    'cones': cones_out
                })
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"PhotoreceptorProcess error: {e}")
                break
        
        print("PhotoreceptorProcess stopped")


class BipolarProcess(mp.Process):
    """Horizontal + Bipolar cells"""
    def __init__(self, input_queue, output_queue, config, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        
    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build mean kernel for gap junctions
        mean_kernel = torch.ones((1,1,3,3), dtype=torch.float32, device=device) / 9.0
        gap_strength = self.config['gap_junction_strength']
        
        # Bipolar state
        bipolar_taus = [0.7, 0.9, 0.99]
        bipolar_states = [None] * len(bipolar_taus)
        
        print(f"BipolarProcess started on {device}")
        
        while not self.stop_event.is_set():
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    break
                
                rods_out = data['rods']
                cones_out = data['cones']
                
                # Horizontal gap junctions (DoG)
                def gap_junctions(signal_np):
                    t = torch.from_numpy(signal_np.astype(np.float32) / 255.0).to(device)
                    t = t.unsqueeze(0).unsqueeze(0).contiguous()
                    signal = t.clone()
                    
                    with torch.no_grad():
                        for _ in range(5):
                            neighbor_avg = torch.nn.functional.conv2d(signal, mean_kernel, padding=1)
                            signal = (1 - gap_strength) * signal + gap_strength * neighbor_avg
                        
                        inhibition = (t - signal) * 2.0
                        inhibition = inhibition.clamp(0.0, 1.0)
                    
                    out = (inhibition.squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                    del t, signal, inhibition
                    return out
                
                rods_dog = gap_junctions(rods_out)
                cones_dog = gap_junctions(cones_out[:,:,2])  # Value channel
                
                # Bipolar cells
                center_t = torch.from_numpy(rods_dog.astype(np.float32) / 255.0).to(device)
                surround_t = torch.from_numpy(cones_dog.astype(np.float32) / 255.0).to(device)
                
                with torch.no_grad():
                    current = (center_t - surround_t).detach()
                    
                    outputs = []
                    for i, tau in enumerate(bipolar_taus):
                        if bipolar_states[i] is None:
                            bipolar_states[i] = current.clone()
                        else:
                            bipolar_states[i] = (tau * bipolar_states[i] + (1.0 - tau) * current).detach()
                        
                        on_response = torch.relu(bipolar_states[i]) * 1.5
                        off_response = torch.relu(-bipolar_states[i]) * 1.5
                        
                        on_np = (on_response.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                        off_np = (off_response.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                        outputs.append((on_np, off_np))
                
                del center_t, surround_t, current
                
                self.output_queue.put({
                    'rods_dog': rods_dog,
                    'cones_dog': cones_dog,
                    'bipolar': outputs,
                    'cones': cones_out  # Pass through for SNN
                })
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"BipolarProcess error: {e}")
                break
        
        print("BipolarProcess stopped")


class AmacrineProcess(mp.Process):
    """Amacrine cells - motion detection"""
    def __init__(self, input_queue, output_queue, config, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        
    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        prev_on = None
        prev_off = None
        motion_on_acc = None
        motion_off_acc = None
        persistence = self.config['motion_persistence']
        
        print(f"AmacrineProcess started on {device}")
        
        while not self.stop_event.is_set():
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    break
                
                on_chan, off_chan = data['bipolar'][0]
                
                on_t = torch.from_numpy(on_chan.astype(np.float32) / 255.0).to(device)
                off_t = torch.from_numpy(off_chan.astype(np.float32) / 255.0).to(device)
                
                with torch.no_grad():
                    if prev_on is None:
                        prev_on = on_t.clone()
                        prev_off = off_t.clone()
                        motion_on_acc = torch.zeros_like(on_t, device=device)
                        motion_off_acc = torch.zeros_like(off_t, device=device)
                        motion_on_np = np.zeros_like(on_chan)
                        motion_off_np = np.zeros_like(off_chan)
                    else:
                        motion_on = torch.abs(on_t - prev_on)
                        motion_off = torch.abs(off_t - prev_off)
                        
                        motion_on_acc = (motion_on_acc * persistence + motion_on).detach()
                        motion_off_acc = (motion_off_acc * persistence + motion_off).detach()
                        
                        motion_on_np = (motion_on_acc.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                        motion_off_np = (motion_off_acc.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
                        
                        prev_on = on_t.clone()
                        prev_off = off_t.clone()
                
                del on_t, off_t
                
                data['motion_on'] = motion_on_np
                data['motion_off'] = motion_off_np
                self.output_queue.put(data)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"AmacrineProcess error: {e}")
                break
        
        print("AmacrineProcess stopped")


class GanglionProcess(mp.Process):
    """Ganglion cells - spike generation"""
    def __init__(self, input_queue, output_queue, config, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        
    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Wide-field kernel
        ksize = self.config['wide_field_size']
        sigma = float(ksize) / 3.0
        half = ksize // 2
        x = torch.arange(-half, half+1, device=device, dtype=torch.float32)
        g = torch.exp(-(x**2) / (2.0 * sigma * sigma))
        g = g / g.sum()
        kernel2d = g[:, None] * g[None, :]
        wide_kernel = kernel2d[None, None, :, :].to(device)
        
        spike_thresh = self.config['spike_threshold']
        
        print(f"GanglionProcess started on {device}")
        
        while not self.stop_event.is_set():
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    break
                
                def wide_field_inhibit(signal_np):
                    t = torch.from_numpy(signal_np.astype(np.float32) / 255.0).to(device)
                    t = t.unsqueeze(0).unsqueeze(0).contiguous()
                    
                    with torch.no_grad():
                        blurred = torch.nn.functional.conv2d(t, wide_kernel, padding=ksize//2)
                        local_minus_global = (t - (blurred / 4.0)).clamp(0.0, 1.0)
                    
                    out = (local_minus_global.squeeze(0).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                    del t, blurred, local_minus_global
                    return out
                
                motion_on_inh = wide_field_inhibit(data['motion_on'])
                motion_off_inh = wide_field_inhibit(data['motion_off'])
                
                on_spikes = cv2.threshold(motion_on_inh, spike_thresh, 255, cv2.THRESH_BINARY)[1]
                off_spikes = cv2.threshold(motion_off_inh, spike_thresh, 255, cv2.THRESH_BINARY)[1]
                
                data['on_spikes'] = on_spikes
                data['off_spikes'] = off_spikes
                self.output_queue.put(data)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"GanglionProcess error: {e}")
                break
        
        print("GanglionProcess stopped")


class SNNProcess(mp.Process):
    """SNN classification - separate process"""
    def __init__(self, input_queue, output_queue, config, stop_event):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        
    def run(self):
        from visual.retina import Retina  # Import SNN components
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize SNN
        num_hue_bins = self.config['num_hue_bins']
        n_neurons = self.config['n_neurons']
        snn_input_size = num_hue_bins + 2
        ds_w = self.config['ds_w']
        ds_h = self.config['ds_h']
        temporal_window = self.config['temporal_window']
        
        # Create SNN layer
        snn = Retina.BatchLIFLayer(snn_input_size, n_neurons).to(device)
        snn_state = snn.init_state(ds_w * ds_h)
        
        # Pre-allocated buffers
        temporal_buffer = [
            torch.zeros((ds_h, ds_w, snn_input_size), dtype=torch.float32, device=device)
            for _ in range(temporal_window)
        ]
        buffer_idx = 0
        
        seq_buffer = torch.zeros(
            (temporal_window, ds_h, ds_w, snn_input_size),
            device=device, dtype=torch.float32
        )
        
        spike_buffer = torch.zeros(
            (temporal_window, ds_w * ds_h, n_neurons),
            device=device, dtype=torch.float32
        )
        
        print(f"SNNProcess started on {device}")
        
        while not self.stop_event.is_set():
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    break
                
                # Extract features
                on_spikes = data['on_spikes']
                off_spikes = data['off_spikes']
                cones_hsv = data['cones']
                
                # Downsample and encode
                down_hsv = cv2.resize(cones_hsv, (ds_w, ds_h), interpolation=cv2.INTER_AREA)
                h = down_hsv[...,0].astype(np.int32)
                s = down_hsv[...,1]
                v = down_hsv[...,2]
                mask = (s > 20) & (v > 20)
                bin_width = max(1, 180 // num_hue_bins)
                bin_idx = np.clip(h // bin_width, 0, num_hue_bins - 1)
                onehot = np.zeros((ds_h, ds_w, num_hue_bins), dtype=np.float32)
                onehot[mask, bin_idx[mask]] = 1.0
                
                on_ds = cv2.resize(on_spikes, (ds_w, ds_h)) / 255.0
                off_ds = cv2.resize(off_spikes, (ds_w, ds_h)) / 255.0
                combined = np.concatenate([onehot, on_ds[..., None], off_ds[..., None]], axis=-1)
                combined_t = torch.from_numpy(combined.astype(np.float32)).to(device).detach()
                
                # Update circular buffer
                temporal_buffer[buffer_idx].copy_(combined_t)
                buffer_idx = (buffer_idx + 1) % temporal_window
                
                # Build sequence
                indices = [(buffer_idx + i) % temporal_window for i in range(temporal_window)]
                for i, idx in enumerate(indices):
                    seq_buffer[i] = temporal_buffer[idx]
                
                seq_reshaped = seq_buffer.view(temporal_window, -1, snn_input_size)
                
                # SNN forward
                with torch.no_grad():
                    state = snn_state
                    for t in range(temporal_window):
                        spk, state = snn(seq_reshaped[t], state)
                        spike_buffer[t] = spk
                    snn_state = state
                    
                    # Belief map
                    weights_np = snn.fc.weight.cpu().numpy()
                    pre_last_np = seq_reshaped[-1].cpu().numpy()
                    activ = pre_last_np @ weights_np.T
                    assigned = np.argmax(activ, axis=1)
                    belief_map = assigned.reshape(ds_h, ds_w)
                
                data['belief'] = belief_map
                self.output_queue.put(data)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"SNNProcess error: {e}")
                break
        
        print("SNNProcess stopped")


class MultiprocessRetina:
    """Orchestrator for multiprocess retina pipeline"""
    def __init__(self, config):
        self.config = config
        self.stop_event = Event()
        
        # Create queues
        self.photo_to_bipolar = Queue(maxsize=2)
        self.bipolar_to_amacrine = Queue(maxsize=2)
        self.amacrine_to_ganglion = Queue(maxsize=2)
        self.ganglion_to_snn = Queue(maxsize=2)
        self.snn_output = Queue(maxsize=2)
        
        # Create processes
        self.processes = [
            PhotoreceptorProcess(Queue(maxsize=2), self.photo_to_bipolar, config, self.stop_event),
            BipolarProcess(self.photo_to_bipolar, self.bipolar_to_amacrine, config, self.stop_event),
            AmacrineProcess(self.bipolar_to_amacrine, self.amacrine_to_ganglion, config, self.stop_event),
            GanglionProcess(self.amacrine_to_ganglion, self.ganglion_to_snn, config, self.stop_event),
            SNNProcess(self.ganglion_to_snn, self.snn_output, config, self.stop_event)
        ]
        
        # Start all processes
        for p in self.processes:
            p.start()
        
        # Input queue reference
        self.input_queue = self.processes[0].input_queue
        
        print("MultiprocessRetina initialized with 5 parallel processes")
    
    def forward(self, frame):
        """Send frame and get result"""
        self.input_queue.put(frame)
        try:
            return self.snn_output.get(timeout=1.0)
        except:
            return None
    
    def shutdown(self):
        """Gracefully shutdown all processes"""
        print("Shutting down MultiprocessRetina...")
        self.stop_event.set()
        
        # Send None to all queues to unblock
        try:
            self.input_queue.put(None, timeout=0.1)
        except:
            pass
        
        # Wait for all processes
        for p in self.processes:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
        
        print("All processes terminated")


# Usage example
if __name__ == "__main__":
    config = {
        'spike_threshold': 10,
        'fovea_radius_ratio': 0.25,
        'num_hue_bins': 10,
        'n_neurons': 32,
        'temporal_window': 3,
        'ds_w': 80,
        'ds_h': 60,
        'motion_persistence': 0.3,
        'gap_junction_strength': 0.3,
        'gaussian_ksize_rods': 5,
        'wide_field_size': 5
    }
    
    retina = MultiprocessRetina(config)
    
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            result = retina.forward(frame)
            if result is not None:
                # Visualize belief map
                if 'belief' in result:
                    belief = result['belief']
                    belief_viz = cv2.resize(belief.astype(np.uint8) * 20, (640, 480), 
                                           interpolation=cv2.INTER_NEAREST)
                    belief_color = cv2.applyColorMap(belief_viz, cv2.COLORMAP_JET)
                    cv2.imshow("Belief Map", belief_color)
                
                # Show ganglion spikes
                if 'on_spikes' in result:
                    cv2.imshow("Ganglion Spikes", result['on_spikes'])
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        retina.shutdown()
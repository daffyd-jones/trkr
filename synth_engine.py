#!/usr/bin/env python3
"""
TRKR SYNTH ENGINE 

"""

import numpy as np
import sounddevice as sd
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
import time
from scipy import signal
import sys
import os

# Numba for JIT compilation (optional but HIGHLY recommended)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("Numba available - using JIT compilation for maximum performance")
except ImportError:
    NUMBA_AVAILABLE = False
    print("WARNING: Numba not available - performance will be reduced")
    print("Install with: pip install numba")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Try to boost thread priority on supported platforms
try:
    if sys.platform == 'win32':
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        THREAD_PRIORITY_TIME_CRITICAL = 15
        def boost_thread_priority():
            thread = kernel32.GetCurrentThread()
            kernel32.SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL)
    elif sys.platform in ['linux', 'darwin']:
        import ctypes
        libc = ctypes.CDLL('libc.so.6' if sys.platform == 'linux' else 'libc.dylib')
        def boost_thread_priority():
            libc.nice(-20)  # Highest priority
    else:
        def boost_thread_priority():
            pass
except:
    def boost_thread_priority():
        pass


class VoiceType(Enum):
    SINE = "Sine"
    SQUARE = "Square"
    SAW = "Sawtooth"
    TRIANGLE = "Triangle"
    NOISE = "Noise"
    # Drum kits
    TR808_KICK = "808 Kick"
    TR808_SNARE = "808 Snare"
    TR808_CLAP = "808 Clap"
    TR808_HIHAT_CLOSED = "808 HH Closed"
    TR808_HIHAT_OPEN = "808 HH Open"
    TR808_TOM_LOW = "808 Tom Low"
    TR808_TOM_MID = "808 Tom Mid"
    TR808_TOM_HI = "808 Tom Hi"
    TR808_CYMBAL = "808 Cymbal"
    TR808_COWBELL = "808 Cowbell"
    
    TR909_KICK = "909 Kick"
    TR909_SNARE = "909 Snare"
    TR909_CLAP = "909 Clap"
    TR909_HIHAT_CLOSED = "909 HH Closed"
    TR909_HIHAT_OPEN = "909 HH Open"
    TR909_RIDE = "909 Ride"
    TR909_CRASH = "909 Crash"
    
    TR707_KICK = "707 Kick"
    TR707_SNARE = "707 Snare"
    TR707_HIHAT = "707 HiHat"
    TR707_TOM = "707 Tom"
    
    LINN_KICK = "Linn Kick"
    LINN_SNARE = "Linn Snare"
    LINN_HIHAT = "Linn HiHat"
    LINN_CLAP = "Linn Clap"
    
    GLITCH_KICK = "Glitch Kick"
    GLITCH_SNARE = "Glitch Snare"
    GLITCH_HIHAT = "Glitch HH"
    GLITCH_PERC = "Glitch Perc"


class EffectType(Enum):
    CHORUS = "Chorus"
    DELAY = "Delay"
    REVERB = "Reverb"
    COMPRESSION = "Compression"
    CRUSH = "Crush"


# ══════════════════════════════════════════════════════════════════════════════
# NUMBA JIT COMPILED FUNCTIONS - These run at near-C speed
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, fastmath=True)
def reverb_comb_process(buffer, write_pos, buffer_size, delay, mono, 
                        filter_state, damp1, damp2, feedback, length):
    """Process one comb filter for reverb - JIT compiled"""
    output = np.zeros(length, dtype=np.float32)
    
    for i in range(length):
        read_pos = (write_pos - delay) % buffer_size
        delayed = buffer[read_pos]
        filter_state = delayed * damp2 + filter_state * damp1
        buffer[write_pos] = mono[i] + filter_state * feedback
        write_pos = (write_pos + 1) % buffer_size
        output[i] = delayed
    
    # Denormal prevention
    if abs(filter_state) < 1e-10:
        filter_state = 0.0
    
    return output, write_pos, filter_state


@jit(nopython=True, cache=True, fastmath=True)
def reverb_allpass_process(buffer, write_pos, buffer_size, delay, 
                            input_signal, ap_gain, length):
    """Process allpass filter for reverb - JIT compiled"""
    output = np.zeros(length, dtype=np.float32)
    
    for i in range(length):
        read_pos = (write_pos - delay) % buffer_size
        delayed = buffer[read_pos]
        buffer[write_pos] = input_signal[i] + delayed * ap_gain
        output[i] = delayed - input_signal[i] * ap_gain
        write_pos = (write_pos + 1) % buffer_size
    
    return output, write_pos


@jit(nopython=True, cache=True, fastmath=True)
def chorus_interpolate(history, current, delay_samples, history_len, length):
    """Interpolate chorus delay line - JIT compiled"""
    output = np.zeros(length, dtype=np.float32)
    
    for i in range(length):
        delay = delay_samples[i]
        
        # Ensure delay is within valid range
        if delay < 0.0:
            delay = 0.0
        
        # Calculate position in combined history+current buffer
        if delay <= i:
            # Read from current buffer
            pos = i - delay
            idx = int(pos)
            if 0 <= idx < length - 1:
                frac = pos - idx
                output[i] = current[idx] * (1.0 - frac) + current[idx + 1] * frac
            elif idx >= 0 and idx < length:
                output[i] = current[idx]
        else:
            # Read from history
            pos = history_len - (delay - i)
            idx = int(pos)
            if 0 <= idx < history_len - 1:
                frac = pos - idx
                output[i] = history[idx] * (1.0 - frac) + history[idx + 1] * frac
            elif idx >= 0 and idx < history_len:
                output[i] = history[idx]
    
    return output


@jit(nopython=True, cache=True, fastmath=True)
def compress_envelope_follow(input_signal, env_start, attack_coeff, release_coeff, length):
    """Envelope follower for compressor - JIT compiled"""
    envelope = np.zeros(length, dtype=np.float32)
    envelope[0] = env_start
    
    for i in range(1, length):
        level = abs(input_signal[i])
        coeff = attack_coeff if level > envelope[i-1] else release_coeff
        envelope[i] = coeff * envelope[i-1] + (1.0 - coeff) * level
    
    return envelope


# ══════════════════════════════════════════════════════════════════════════════
# CIRCULAR BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class CircularBuffer:
    """Efficient circular buffer using numpy"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.write_pos = 0
    
    def write(self, samples: np.ndarray):
        """Write samples to buffer"""
        n = len(samples)
        if n == 0:
            return
            
        if self.write_pos + n <= self.size:
            self.buffer[self.write_pos:self.write_pos + n] = samples
        else:
            first_chunk = self.size - self.write_pos
            self.buffer[self.write_pos:] = samples[:first_chunk]
            self.buffer[:n - first_chunk] = samples[first_chunk:]
        
        self.write_pos = (self.write_pos + n) % self.size
    
    def read(self, delay_samples: int, length: int) -> np.ndarray:
        """Read samples from buffer with given delay"""
        if delay_samples >= self.size:
            delay_samples = self.size - 1
        if delay_samples < 1:
            delay_samples = 1
        
        read_pos = (self.write_pos - delay_samples) % self.size
        
        if read_pos + length <= self.size:
            return self.buffer[read_pos:read_pos + length].copy()
        else:
            first_chunk = self.size - read_pos
            result = np.empty(length, dtype=np.float32)
            result[:first_chunk] = self.buffer[read_pos:]
            result[first_chunk:] = self.buffer[:length - first_chunk]
            return result
    
    def clear(self):
        """Clear buffer contents"""
        self.buffer.fill(0)
        self.write_pos = 0


# ══════════════════════════════════════════════════════════════════════════════
# EFFECT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EffectParams:
    enabled: bool = False
    wet_mix: float = 0.5


@dataclass
class ChorusParams(EffectParams):
    rate: float = 1.5
    depth: float = 0.02
    feedback: float = 0.1


@dataclass
class DelayParams(EffectParams):
    time: float = 0.3
    feedback: float = 0.4
    cross_feedback: float = 0.0


@dataclass
class ReverbParams(EffectParams):
    room_size: float = 0.5
    damping: float = 0.5
    width: float = 1.0


@dataclass
class CompressionParams(EffectParams):
    threshold: float = -20.0
    ratio: float = 4.0
    attack: float = 0.005
    release: float = 0.1
    makeup_gain: float = 0.0


@dataclass
class CrushParams(EffectParams):
    bits: int = 8
    downsample: int = 1


@dataclass
class EffectsBus:
    effect_type: EffectType
    params: EffectParams
    has_tail: bool = False
    silent_buffer_count: int = 0

    def __post_init__(self):
        if self.effect_type == EffectType.CHORUS:
            self.params = ChorusParams()
        elif self.effect_type == EffectType.DELAY:
            self.params = DelayParams()
        elif self.effect_type == EffectType.REVERB:
            self.params = ReverbParams()
        elif self.effect_type == EffectType.COMPRESSION:
            self.params = CompressionParams()
        elif self.effect_type == EffectType.CRUSH:
            self.params = CrushParams()


@dataclass
class ADSR:
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7
    release: float = 0.2


@dataclass
class Filter:
    cutoff: float = 8000.0
    resonance: float = 1.0
    filter_type: str = "lowpass"


@dataclass
class DrumFilter:
    enabled: bool = False
    cutoff: float = 8000.0
    resonance: float = 1.0
    filter_type: str = "lowpass"


@dataclass
class ChannelVoice:
    voice_type: VoiceType = VoiceType.SINE
    adsr: ADSR = field(default_factory=ADSR)
    filter: Filter = field(default_factory=Filter)
    volume: float = 0.8
    pan: float = 0.5
    detune: float = 0.0
    
    drum_length_multiplier: float = 1.0
    drum_release_envelope: float = 1.0
    drum_filter: DrumFilter = field(default_factory=DrumFilter)
    
    send_chorus: float = 0.0
    send_delay: float = 0.0
    send_reverb: float = 0.0
    send_compression: float = 0.0
    send_crush: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SYNTH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class SynthEngine:
    """Main synthesis engine - MAXIMUM PERFORMANCE VERSION"""

    _COMB_DELAYS_L = [1557, 1617, 1491, 1422]
    _COMB_DELAYS_R = [1557 + 23, 1617 + 23, 1491 + 23, 1422 + 23]
    _ALLPASS_DELAYS = [225, 556]
    _ALLPASS_GAIN = 0.5
    
    SILENCE_THRESHOLD = 1e-5
    TAIL_SILENCE_BUFFERS = 20

    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.global_volume = 0.7
        
        # For unstable systems, you can increase buffer_size to 1024 or 2048
        # This adds latency but reduces CPU load
        
        self.channels = [ChannelVoice() for _ in range(8)]
        
        self.effects_buses = {
            EffectType.CHORUS: EffectsBus(EffectType.CHORUS, ChorusParams()),
            EffectType.DELAY: EffectsBus(EffectType.DELAY, DelayParams()),
            EffectType.REVERB: EffectsBus(EffectType.REVERB, ReverbParams()),
            EffectType.COMPRESSION: EffectsBus(EffectType.COMPRESSION, CompressionParams()),
            EffectType.CRUSH: EffectsBus(EffectType.CRUSH, CrushParams()),
        }
        
        self._preallocate_buffers()
        self._init_effect_state()
        
        self.active_notes: Dict[int, List[Dict]] = {i: [] for i in range(8)}
        self._current_dry: Dict[int, Optional[np.ndarray]] = {i: None for i in range(8)}

        self.stream = None
        self.running = False
        self.note_queue = queue.Queue()
        self._priority_boosted = False
    
    def _preallocate_buffers(self):
        """Pre-allocate all working buffers"""
        max_chorus_samples = int(2.0 * self.sample_rate)
        self._chorus_buf_l = CircularBuffer(max_chorus_samples)
        self._chorus_buf_r = CircularBuffer(max_chorus_samples)
        
        # Pre-fill with silence to prevent reading uninitialized data
        silence = np.zeros(max_chorus_samples, dtype=np.float32)
        self._chorus_buf_l.write(silence)
        self._chorus_buf_r.write(silence)
        
        # Chorus history buffers (avoid concatenation)
        max_delay_samps = int(0.05 * self.sample_rate)  # 50ms max
        self._chorus_hist_l = np.zeros(max_delay_samps, dtype=np.float32)
        self._chorus_hist_r = np.zeros(max_delay_samps, dtype=np.float32)
        
        max_delay_samples = int(5.0 * self.sample_rate)
        self._delay_buf_l = CircularBuffer(max_delay_samples)
        self._delay_buf_r = CircularBuffer(max_delay_samples)
        
        self._send_l = np.zeros(self.buffer_size, dtype=np.float32)
        self._send_r = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Pre-allocate LFO arrays
        self._lfo_phases = np.zeros(self.buffer_size, dtype=np.float32)
        self._delay_samples = np.zeros(self.buffer_size, dtype=np.float32)
    
    def _init_effect_state(self):
        """Initialize effect state"""
        self._effect_state = {
            EffectType.CHORUS: {
                'lfo_phase': 0.0,
            },
            EffectType.DELAY: {},
            EffectType.REVERB: {
                # Store write positions and filter states
                'comb_wpos_l': [0] * len(self._COMB_DELAYS_L),
                'comb_wpos_r': [0] * len(self._COMB_DELAYS_R),
                'comb_flt_l': np.zeros(len(self._COMB_DELAYS_L), dtype=np.float32),
                'comb_flt_r': np.zeros(len(self._COMB_DELAYS_R), dtype=np.float32),
                'ap_wpos_l': [0] * len(self._ALLPASS_DELAYS),
                'ap_wpos_r': [0] * len(self._ALLPASS_DELAYS),
                # Buffers
                'comb_bufs_l': [np.zeros(d + self.buffer_size * 2, dtype=np.float32) 
                                for d in self._COMB_DELAYS_L],
                'comb_bufs_r': [np.zeros(d + self.buffer_size * 2, dtype=np.float32) 
                                for d in self._COMB_DELAYS_R],
                'ap_bufs_l': [np.zeros(d + self.buffer_size * 2, dtype=np.float32) 
                              for d in self._ALLPASS_DELAYS],
                'ap_bufs_r': [np.zeros(d + self.buffer_size * 2, dtype=np.float32) 
                              for d in self._ALLPASS_DELAYS],
            },
            EffectType.COMPRESSION: {
                'env_left': 0.0,
                'env_right': 0.0,
            },
            EffectType.CRUSH: {},
        }
        
    def start(self):
        """Start the audio stream"""
        if self.running:
            return
            
        self.running = True
        self.stream = sd.OutputStream(
            channels=2,
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.stream.start()
        print(f"Audio engine started: {self.sample_rate}Hz, {self.buffer_size} samples buffer")
        if NUMBA_AVAILABLE:
            print("Using Numba JIT compilation - first buffer may compile...")
        
    def stop(self):
        """Stop the audio stream"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    def note_on(self, channel: int, note: int, velocity: int):
        """Trigger a note"""
        if not 0 <= channel < 8:
            return
        self.note_queue.put({
            'action': 'note_on',
            'channel': channel,
            'note': note,
            'velocity': velocity / 127.0,
            'time': time.time()
        })
        
    def note_off(self, channel: int, note: int):
        """Release a note"""
        if not 0 <= channel < 8:
            return
        self.note_queue.put({
            'action': 'note_off',
            'channel': channel,
            'note': note,
            'time': time.time()
        })

    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback - boost priority on first call"""
        if not self._priority_boosted:
            boost_thread_priority()
            self._priority_boosted = True
        
        # Process note queue
        while not self.note_queue.empty():
            try:
                msg = self.note_queue.get_nowait()
                if msg['action'] == 'note_on':
                    self._trigger_note(msg['channel'], msg['note'], msg['velocity'])
                elif msg['action'] == 'note_off':
                    self._release_note(msg['channel'], msg['note'])
            except queue.Empty:
                break

        output = np.zeros((frames, 2), dtype=np.float32)

        # Render dry audio
        for ch_idx in range(8):
            dry = self._render_channel(ch_idx, frames)
            self._current_dry[ch_idx] = dry
            if dry is not None:
                output += dry
            self._feed_sends(ch_idx, dry, frames)

        # Process effects
        effect_out = self._drain_effect_tails(frames)
        output += effect_out

        # Global volume and soft clip
        output *= self.global_volume
        output = np.tanh(output)

        outdata[:] = output.astype(np.float32)
    
    def _trigger_note(self, channel: int, note: int, velocity: float):
        """Trigger a note"""
        voice = self.channels[channel]
        
        if self._is_drum_voice(voice.voice_type):
            drum_sample = self._render_drum_sample(voice.voice_type, note, velocity, voice.adsr, voice)
            note_data = {
                'note': note,
                'velocity': velocity,
                'phase': 0,
                'drum_sample': drum_sample,
                'is_drum': True
            }
        else:
            note_data = {
                'note': note,
                'velocity': velocity,
                'phase': 0.0,
                'envelope_phase': 0.0,
                'state': 'attack',
                'release_time': None,
                'is_drum': False,
                'filter_state': 0.0,  # Explicitly initialize to zero
                'sample_count': 0  # Initialize sample count
            }
        
        self.active_notes[channel].append(note_data)
        
    def _release_note(self, channel: int, note: int):
        """Release a note"""
        for note_data in self.active_notes[channel]:
            if note_data['note'] == note and not note_data.get('is_drum'):
                if note_data['state'] != 'release':
                    note_data['state'] = 'release'
                    note_data['release_time'] = time.time()

    def _render_channel(self, channel: int, frames: int) -> Optional[np.ndarray]:
        """Render dry audio for a channel"""
        if not self.active_notes[channel]:
            return None

        voice = self.channels[channel]
        output = np.zeros((frames, 2), dtype=np.float32)
        notes_to_remove = []

        for note_data in self.active_notes[channel]:
            if note_data.get('is_drum'):
                drum_sample = note_data['drum_sample']
                start_idx = note_data['phase']
                end_idx = min(start_idx + frames, len(drum_sample))

                if start_idx >= len(drum_sample):
                    notes_to_remove.append(note_data)
                    continue

                chunk_len = end_idx - start_idx
                sig = drum_sample[start_idx:end_idx] * voice.volume

                left_gain = np.sqrt(1.0 - voice.pan)
                right_gain = np.sqrt(voice.pan)

                output[:chunk_len, 0] += sig * left_gain
                output[:chunk_len, 1] += sig * right_gain

                note_data['phase'] = end_idx

            else:
                sig = self._generate_oscillator(voice.voice_type, note_data, frames, voice.detune)
                
                # Apply anti-click fade-in at the very start (sample_count already initialized)
                fade_duration = 512  # ~11.6ms at 44.1kHz
                if note_data['sample_count'] < fade_duration:
                    fade_samples = min(fade_duration - note_data['sample_count'], frames)
                    fade_start = note_data['sample_count']
                    
                    # Cubic fade-in: smooth acceleration from zero (smoothstep)
                    t = (fade_start + np.arange(fade_samples)) / fade_duration
                    fade_curve = t * t * (3.0 - 2.0 * t)
                    sig[:fade_samples] *= fade_curve

                note_data['sample_count'] += frames
                
                envelope = self._generate_envelope(note_data, frames, voice.adsr)
                sig *= envelope
                sig, new_filter_state = self._apply_filter(sig, voice.filter, note_data['filter_state'])
                note_data['filter_state'] = new_filter_state
                sig *= note_data['velocity'] * voice.volume

                left_gain = np.sqrt(1.0 - voice.pan)
                right_gain = np.sqrt(voice.pan)

                output[:, 0] += sig * left_gain
                output[:, 1] += sig * right_gain

                note_data['envelope_phase'] += frames / self.sample_rate

                if note_data['state'] == 'release':
                    if note_data['envelope_phase'] > (voice.adsr.attack + voice.adsr.decay + voice.adsr.release):
                        notes_to_remove.append(note_data)

        for note_data in notes_to_remove:
            self.active_notes[channel].remove(note_data)

        return output

    def _feed_sends(self, channel: int, dry_signal: np.ndarray, frames: int):
        """Feed sends to effects"""
        voice = self.channels[channel]
        if dry_signal is None:
            return
            
        send_map = {
            EffectType.CHORUS: voice.send_chorus,
            EffectType.DELAY: voice.send_delay,
            EffectType.REVERB: voice.send_reverb,
            EffectType.COMPRESSION: voice.send_compression,
            EffectType.CRUSH: voice.send_crush,
        }
    
        for effect_type, send_amount in send_map.items():
            bus = self.effects_buses[effect_type]
            if not bus.params.enabled or send_amount <= 0:
                continue
            
            rms = float(np.sqrt(np.mean(dry_signal[:, 0] ** 2 + dry_signal[:, 1] ** 2)))
            if rms > self.SILENCE_THRESHOLD:
                bus.has_tail = True
                bus.silent_buffer_count = 0

    def _drain_effect_tails(self, frames: int) -> np.ndarray:
        """Process all effects and return summed output"""
        output = np.zeros((frames, 2), dtype=np.float32)

        for effect_type, bus in self.effects_buses.items():
            if not bus.params.enabled or not bus.has_tail:
                continue

            # Gather sends
            self._send_l.fill(0)
            self._send_r.fill(0)

            for ch_idx in range(8):
                voice = self.channels[ch_idx]
                send_amount = {
                    EffectType.CHORUS: voice.send_chorus,
                    EffectType.DELAY: voice.send_delay,
                    EffectType.REVERB: voice.send_reverb,
                    EffectType.COMPRESSION: voice.send_compression,
                    EffectType.CRUSH: voice.send_crush,
                }[effect_type]

                if send_amount <= 0:
                    continue

                ch_dry = self._current_dry[ch_idx]
                if ch_dry is not None:
                    self._send_l[:frames] += ch_dry[:, 0] * send_amount
                    self._send_r[:frames] += ch_dry[:, 1] * send_amount

            # Process effect
            if effect_type == EffectType.CHORUS:
                wet_l, wet_r = self._process_chorus(self._send_l[:frames], self._send_r[:frames], bus)
            elif effect_type == EffectType.DELAY:
                wet_l, wet_r = self._process_delay(self._send_l[:frames], self._send_r[:frames], bus)
            elif effect_type == EffectType.REVERB:
                wet_l, wet_r = self._process_reverb(self._send_l[:frames], self._send_r[:frames], bus)
            elif effect_type == EffectType.COMPRESSION:
                wet_l, wet_r = self._process_compression(self._send_l[:frames], self._send_r[:frames], bus)
            elif effect_type == EffectType.CRUSH:
                wet_l, wet_r = self._process_crush(self._send_l[:frames], self._send_r[:frames], bus)
            else:
                continue

            # Check silence
            rms = float(np.sqrt(np.mean(wet_l ** 2 + wet_r ** 2)))
            if rms < self.SILENCE_THRESHOLD:
                bus.silent_buffer_count += 1
                if bus.silent_buffer_count >= self.TAIL_SILENCE_BUFFERS:
                    bus.has_tail = False
                    bus.silent_buffer_count = 0
                    continue
            else:
                bus.silent_buffer_count = 0

            output[:, 0] += wet_l
            output[:, 1] += wet_r

        return output

    # ══════════════════════════════════════════════════════════════════════════
    # EFFECTS PROCESSING - Using JIT where beneficial
    # ══════════════════════════════════════════════════════════════════════════

    def _process_chorus(self, input_left: np.ndarray, input_right: np.ndarray,
                        bus: EffectsBus) -> tuple:
        """Chorus with JIT-compiled interpolation"""
        params = bus.params
        if not params.enabled:
            return np.zeros_like(input_left), np.zeros_like(input_right)

        length = len(input_left)
        state = self._effect_state[EffectType.CHORUS]

        # Get history size
        max_delay = int((params.depth + 0.01) * self.sample_rate)
        
        # Ensure we have enough history - pad with zeros if needed
        hist_left = self._chorus_buf_l.read(max_delay, max_delay)
        hist_right = self._chorus_buf_r.read(max_delay, max_delay)
        
        # Generate LFO
        phase_inc = 2.0 * np.pi * params.rate / self.sample_rate
        lfo_phase_start = state['lfo_phase']
        self._lfo_phases[:length] = lfo_phase_start + np.arange(length) * phase_inc
        
        lfo_l = np.sin(self._lfo_phases[:length])
        lfo_r = np.sin(self._lfo_phases[:length] + np.pi / 2.0)
        
        # Calculate delays - ensure minimum delay to avoid reading current sample
        centre = params.depth * self.sample_rate * 0.5
        min_delay = 10.0  # Minimum 10 samples delay
        delay_samples_l = np.clip(centre + centre * lfo_l, min_delay, max_delay - 1)
        delay_samples_r = np.clip(centre + centre * lfo_r, min_delay, max_delay - 1)
        
        # Use JIT-compiled interpolation
        output_left = chorus_interpolate(hist_left, input_left, delay_samples_l, max_delay, length)
        output_right = chorus_interpolate(hist_right, input_right, delay_samples_r, max_delay, length)
        
        # Gentle feedback
        if params.feedback > 0:
            fb = min(params.feedback * 0.5, 0.3)  # Reduced to prevent oscillation
            output_left = output_left + output_left * fb
            output_right = output_right + output_right * fb
        
        # Write to history
        self._chorus_buf_l.write(input_left)
        self._chorus_buf_r.write(input_right)
        
        # Update phase
        state['lfo_phase'] = (lfo_phase_start + length * phase_inc) % (2.0 * np.pi)
        
        # Apply wet mix
        return output_left * params.wet_mix, output_right * params.wet_mix

    def _process_delay(self, input_left: np.ndarray, input_right: np.ndarray,
                       bus: EffectsBus) -> tuple:
        """Delay effect"""
        params = bus.params
        if not params.enabled:
            return np.zeros_like(input_left), np.zeros_like(input_right)

        length = len(input_left)
        delay_samples = int(params.time * self.sample_rate)
        delay_samples = max(1, min(delay_samples, self._delay_buf_l.size - 1))

        echo_left = self._delay_buf_l.read(delay_samples, length)
        echo_right = self._delay_buf_r.read(delay_samples, length)

        fb = min(params.feedback, 0.95)
        feedback_left = echo_left * fb
        feedback_right = echo_right * fb

        self._delay_buf_l.write(input_left + feedback_left)
        self._delay_buf_r.write(input_right + feedback_right)

        return echo_left * params.wet_mix, echo_right * params.wet_mix

    def _process_reverb(self, input_left: np.ndarray, input_right: np.ndarray,
                        bus: EffectsBus) -> tuple:
        """Reverb using JIT-compiled comb and allpass filters"""
        params = bus.params
        if not params.enabled:
            return np.zeros_like(input_left), np.zeros_like(input_right)

        state = self._effect_state[EffectType.REVERB]
        length = len(input_left)

        # Mix to mono
        mono = (input_left + input_right) * 0.015

        # Coefficients
        feedback = 0.7 + params.room_size * 0.28
        feedback = min(feedback, 0.98)
        damp1 = params.damping * 0.4
        damp2 = 1.0 - damp1

        # Process combs using JIT
        sum_l = np.zeros(length, dtype=np.float32)
        sum_r = np.zeros(length, dtype=np.float32)

        for c in range(len(self._COMB_DELAYS_L)):
            # Left
            out_l, wpos_l, flt_l = reverb_comb_process(
                state['comb_bufs_l'][c],
                state['comb_wpos_l'][c],
                len(state['comb_bufs_l'][c]),
                self._COMB_DELAYS_L[c],
                mono,
                state['comb_flt_l'][c],
                damp1, damp2, feedback, length
            )
            state['comb_wpos_l'][c] = wpos_l
            state['comb_flt_l'][c] = flt_l
            sum_l += out_l
            
            # Right
            out_r, wpos_r, flt_r = reverb_comb_process(
                state['comb_bufs_r'][c],
                state['comb_wpos_r'][c],
                len(state['comb_bufs_r'][c]),
                self._COMB_DELAYS_R[c],
                mono,
                state['comb_flt_r'][c],
                damp1, damp2, feedback, length
            )
            state['comb_wpos_r'][c] = wpos_r
            state['comb_flt_r'][c] = flt_r
            sum_r += out_r

        # Process allpass using JIT
        for a in range(len(self._ALLPASS_DELAYS)):
            # Left
            sum_l, wpos_l = reverb_allpass_process(
                state['ap_bufs_l'][a],
                state['ap_wpos_l'][a],
                len(state['ap_bufs_l'][a]),
                self._ALLPASS_DELAYS[a],
                sum_l,
                self._ALLPASS_GAIN,
                length
            )
            state['ap_wpos_l'][a] = wpos_l
            
            # Right
            sum_r, wpos_r = reverb_allpass_process(
                state['ap_bufs_r'][a],
                state['ap_wpos_r'][a],
                len(state['ap_bufs_r'][a]),
                self._ALLPASS_DELAYS[a],
                sum_r,
                self._ALLPASS_GAIN,
                length
            )
            state['ap_wpos_r'][a] = wpos_r

        # Width control
        if params.width < 1.0:
            mono_wet = (sum_l + sum_r) * 0.5
            wet_factor = params.width
            sum_l = mono_wet * (1.0 - wet_factor) + sum_l * wet_factor
            sum_r = mono_wet * (1.0 - wet_factor) + sum_r * wet_factor

        return (sum_l * params.wet_mix).astype(np.float32), (sum_r * params.wet_mix).astype(np.float32)

    def _process_compression(self, input_left: np.ndarray, input_right: np.ndarray,
                             bus: EffectsBus) -> tuple:
        """Compression using JIT envelope follower"""
        params = bus.params
        if not params.enabled:
            return np.zeros_like(input_left), np.zeros_like(input_right)

        state = self._effect_state[EffectType.COMPRESSION]
        length = len(input_left)

        attack_coeff = np.exp(-1.0 / max(params.attack * self.sample_rate, 1))
        release_coeff = np.exp(-1.0 / max(params.release * self.sample_rate, 1))

        threshold_db = params.threshold
        makeup = 10.0 ** (params.makeup_gain / 20.0)

        # Use JIT for envelope following
        env_l = compress_envelope_follow(input_left, state['env_left'], attack_coeff, release_coeff, length)
        env_r = compress_envelope_follow(input_right, state['env_right'], attack_coeff, release_coeff, length)

        # Gain computation (vectorized)
        env_db_l = 20.0 * np.log10(np.maximum(env_l, 1e-9))
        env_db_r = 20.0 * np.log10(np.maximum(env_r, 1e-9))

        over_l = np.maximum(env_db_l - threshold_db, 0)
        over_r = np.maximum(env_db_r - threshold_db, 0)
        
        gain_db_l = -(over_l - over_l / params.ratio)
        gain_db_r = -(over_r - over_r / params.ratio)

        gain_l = (10.0 ** (gain_db_l / 20.0)) * makeup
        gain_r = (10.0 ** (gain_db_r / 20.0)) * makeup

        compressed_left = input_left * gain_l
        compressed_right = input_right * gain_r

        state['env_left'] = env_l[-1]
        state['env_right'] = env_r[-1]

        # Return difference for parallel compression, scaled by wet_mix
        diff_left = (compressed_left - input_left) * params.wet_mix
        diff_right = (compressed_right - input_right) * params.wet_mix
        
        return diff_left, diff_right

    def _process_crush(self, input_left: np.ndarray, input_right: np.ndarray,
                       bus: EffectsBus) -> tuple:
        """Bit crusher"""
        params = bus.params
        if not params.enabled:
            return np.zeros_like(input_left), np.zeros_like(input_right)

        levels = float(2 ** (params.bits - 1))
        crushed_left = np.round(input_left * levels) / levels
        crushed_right = np.round(input_right * levels) / levels

        if params.downsample > 1:
            ds = int(params.downsample)
            indices = (np.arange(len(crushed_left)) // ds) * ds
            indices = np.clip(indices, 0, len(crushed_left) - 1)
            crushed_left = crushed_left[indices]
            crushed_right = crushed_right[indices]

        return crushed_left * params.wet_mix, crushed_right * params.wet_mix

    def set_effect_enabled(self, effect_type: EffectType, enabled: bool):
        """Enable/disable effect"""
        bus = self.effects_buses[effect_type]
        bus.params.enabled = enabled
        
        if not enabled:
            bus.has_tail = False
            bus.silent_buffer_count = 0
            
            if effect_type == EffectType.CHORUS:
                self._chorus_buf_l.clear()
                self._chorus_buf_r.clear()
                self._effect_state[EffectType.CHORUS]['lfo_phase'] = 0.0
            elif effect_type == EffectType.DELAY:
                self._delay_buf_l.clear()
                self._delay_buf_r.clear()
            elif effect_type == EffectType.REVERB:
                state = self._effect_state[EffectType.REVERB]
                for buf in state['comb_bufs_l'] + state['comb_bufs_r']:
                    buf.fill(0)
                for buf in state['ap_bufs_l'] + state['ap_bufs_r']:
                    buf.fill(0)
                state['comb_flt_l'].fill(0)
                state['comb_flt_r'].fill(0)
                state['comb_wpos_l'] = [0] * len(self._COMB_DELAYS_L)
                state['comb_wpos_r'] = [0] * len(self._COMB_DELAYS_R)
                state['ap_wpos_l'] = [0] * len(self._ALLPASS_DELAYS)
                state['ap_wpos_r'] = [0] * len(self._ALLPASS_DELAYS)
            elif effect_type == EffectType.COMPRESSION:
                self._effect_state[EffectType.COMPRESSION]['env_left'] = 0.0
                self._effect_state[EffectType.COMPRESSION]['env_right'] = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # OSCILLATOR, ENVELOPE, FILTER (unchanged - already efficient)
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_oscillator(self, voice_type: VoiceType, note_data: Dict, 
                            frames: int, detune: float) -> np.ndarray:
        """Generate oscillator waveform"""
        freq = 440.0 * (2.0 ** ((note_data['note'] - 69) / 12.0))
        freq *= (2.0 ** (detune / 1200.0))
        
        phase_inc = 2.0 * np.pi * freq / self.sample_rate
        phases = note_data['phase'] + np.arange(frames) * phase_inc
        note_data['phase'] = (note_data['phase'] + frames * phase_inc) % (2.0 * np.pi)
        
        if voice_type == VoiceType.SINE:
            waveform = np.sin(phases).astype(np.float32)
        elif voice_type == VoiceType.SQUARE:
            waveform = np.sign(np.sin(phases)).astype(np.float32)
        elif voice_type == VoiceType.SAW:
            t = ((phases / (2.0 * np.pi)) + 0.5) % 1.0
            waveform = (2.0 * t - 1.0).astype(np.float32)
        elif voice_type == VoiceType.TRIANGLE:
            t = ((phases / (2.0 * np.pi)) + 0.75) % 1.0
            waveform = (2.0 * np.abs(2.0 * t - 1.0) - 1.0).astype(np.float32)
        elif voice_type == VoiceType.NOISE:
            waveform = (np.random.uniform(-1.0, 1.0, frames)).astype(np.float32)
        else:
            waveform = np.zeros(frames, dtype=np.float32)
        
        return waveform

    def _generate_envelope(self, note_data: Dict, frames: int, 
                          adsr: ADSR) -> np.ndarray:
        """Generate ADSR envelope"""
        t = note_data['envelope_phase']
        envelope = np.ones(frames, dtype=np.float32)
        
        for i in range(frames):
            current_t = t + (i / self.sample_rate)
            
            if note_data['state'] == 'attack':
                if current_t < adsr.attack:
                    envelope[i] = current_t / adsr.attack if adsr.attack > 0 else 1.0
                else:
                    note_data['state'] = 'decay'
                    envelope[i] = 1.0
                    
            if note_data['state'] == 'decay':
                decay_t = current_t - adsr.attack
                if decay_t < adsr.decay:
                    envelope[i] = 1.0 - (1.0 - adsr.sustain) * (decay_t / adsr.decay)
                else:
                    note_data['state'] = 'sustain'
                    envelope[i] = adsr.sustain
                    
            if note_data['state'] == 'sustain':
                envelope[i] = adsr.sustain
                
            if note_data['state'] == 'release':
                release_t = current_t - adsr.attack - adsr.decay
                if release_t < adsr.release:
                    envelope[i] = adsr.sustain * (1.0 - release_t / adsr.release) if adsr.release > 0 else 0
                else:
                    envelope[i] = 0.0
                    
        return envelope

    def _apply_filter(self, signal_in: np.ndarray, filter_params: Filter, 
                     prev_state: float) -> tuple[np.ndarray, float]:
        """Apply filter with proper type support"""
        if filter_params.cutoff >= self.sample_rate / 2:
            return signal_in, signal_in[-1] if len(signal_in) > 0 else 0.0
        
        nyquist = self.sample_rate / 2
        cutoff_norm = filter_params.cutoff / nyquist
        cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
        
        try:
            if filter_params.filter_type == "lowpass":
                # One-pole lowpass (efficient for realtime)
                rc = 1.0 / (2.0 * np.pi * filter_params.cutoff)
                dt = 1.0 / self.sample_rate
                alpha = dt / (rc + dt)
                
                filtered = np.zeros_like(signal_in)
                filtered[0] = prev_state + alpha * (signal_in[0] - prev_state)
                
                for i in range(1, len(signal_in)):
                    filtered[i] = filtered[i-1] + alpha * (signal_in[i] - filtered[i-1])
                
                final_state = filtered[-1] if len(filtered) > 0 else prev_state
                return filtered, final_state
                
            elif filter_params.filter_type == "highpass":
                # Butterworth highpass
                Q = max(filter_params.resonance, 0.5)
                sos = signal.butter(2, cutoff_norm, btype='highpass', output='sos')
                filtered = signal.sosfilt(sos, signal_in)
                return filtered.astype(np.float32), 0.0
                
            elif filter_params.filter_type == "bandpass":
                # Butterworth bandpass centered at cutoff
                Q = max(filter_params.resonance, 0.5)
                bandwidth = cutoff_norm / Q
                low = max(0.01, cutoff_norm - bandwidth / 2)
                high = min(0.99, cutoff_norm + bandwidth / 2)
                
                if low < high:
                    sos = signal.butter(2, [low, high], btype='bandpass', output='sos')
                    filtered = signal.sosfilt(sos, signal_in)
                    return filtered.astype(np.float32), 0.0
                else:
                    return signal_in, 0.0
            else:
                return signal_in, signal_in[-1] if len(signal_in) > 0 else 0.0
                
        except Exception:
            return signal_in, signal_in[-1] if len(signal_in) > 0 else 0.0

    def _is_drum_voice(self, voice_type: VoiceType) -> bool:
        """Check if voice type is a drum"""
        drum_voices = {
            VoiceType.TR808_KICK, VoiceType.TR808_SNARE, VoiceType.TR808_CLAP,
            VoiceType.TR808_HIHAT_CLOSED, VoiceType.TR808_HIHAT_OPEN,
            VoiceType.TR808_TOM_LOW, VoiceType.TR808_TOM_MID, VoiceType.TR808_TOM_HI,
            VoiceType.TR808_CYMBAL, VoiceType.TR808_COWBELL,
            VoiceType.TR909_KICK, VoiceType.TR909_SNARE, VoiceType.TR909_CLAP,
            VoiceType.TR909_HIHAT_CLOSED, VoiceType.TR909_HIHAT_OPEN,
            VoiceType.TR909_RIDE, VoiceType.TR909_CRASH,
            VoiceType.TR707_KICK, VoiceType.TR707_SNARE, VoiceType.TR707_HIHAT,
            VoiceType.TR707_TOM,
            VoiceType.LINN_KICK, VoiceType.LINN_SNARE, VoiceType.LINN_HIHAT,
            VoiceType.LINN_CLAP,
            VoiceType.GLITCH_KICK, VoiceType.GLITCH_SNARE, VoiceType.GLITCH_HIHAT,
            VoiceType.GLITCH_PERC
        }
        return voice_type in drum_voices

    # DRUM SYNTHESIS - Same as original, already efficient
    def _render_drum_sample(self, voice_type: VoiceType, note: int, velocity: float, adsr: ADSR, voice: ChannelVoice) -> np.ndarray:
        if 'KICK' in voice_type.value or 'TOM' in voice_type.value:
            base_length = int(self.sample_rate * 0.8)
        elif 'SNARE' in voice_type.value or 'CLAP' in voice_type.value:
            base_length = int(self.sample_rate * 0.4)
        elif 'HIHAT' in voice_type.value and 'CLOSED' in voice_type.value:
            base_length = int(self.sample_rate * 0.15)
        elif 'HIHAT' in voice_type.value and 'OPEN' in voice_type.value:
            base_length = int(self.sample_rate * 0.6)
        elif 'CYMBAL' in voice_type.value or 'CRASH' in voice_type.value or 'RIDE' in voice_type.value:
            base_length = int(self.sample_rate * 1.5)
        else:
            base_length = int(self.sample_rate * 0.5)
        
        if 'KICK' in voice_type.value or 'SNARE' in voice_type.value:
            length = int(base_length * voice.drum_length_multiplier)
        else:
            length = base_length
        
        if voice_type == VoiceType.TR808_KICK:
            drum = self._synth_kick(length, start_freq=150, end_freq=40, nonlinearity=1.2)
        elif voice_type == VoiceType.TR808_SNARE:
            drum = self._synth_808_snare(length)
        elif voice_type == VoiceType.TR808_CLAP:
            drum = self._synth_clap(length)
        elif voice_type == VoiceType.TR808_HIHAT_CLOSED:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.05), 8000, 0.3)
        elif voice_type == VoiceType.TR808_HIHAT_OPEN:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.6), 9000, 0.25)
        elif voice_type == VoiceType.TR808_TOM_LOW:
            drum = self._synth_tonal_drum(length, 120, 8.0)
        elif voice_type == VoiceType.TR808_TOM_MID:
            drum = self._synth_tonal_drum(length, 240, 4.0)
        elif voice_type == VoiceType.TR808_TOM_HI:
            drum = self._synth_tonal_drum(length, 360, 4.0)
        elif voice_type == VoiceType.TR808_CYMBAL:
            drum = self._synth_noise_drum(length, 10000, 0.15)
        elif voice_type == VoiceType.TR808_COWBELL:
            drum = self._synth_fm_cowbell(length)
        elif voice_type == VoiceType.TR909_KICK:
            drum = self._synth_kick(length, start_freq=180, end_freq=50, nonlinearity=1.5)
        elif voice_type == VoiceType.TR909_SNARE:
            drum = self._synth_909_snare(length)
        elif voice_type == VoiceType.TR909_CLAP:
            drum = self._synth_clap(length)
        elif voice_type == VoiceType.TR909_HIHAT_CLOSED:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.04), 10000, 0.35)
        elif voice_type == VoiceType.TR909_HIHAT_OPEN:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.5), 11000, 0.25)
        elif voice_type == VoiceType.TR909_RIDE:
            drum = self._synth_noise_drum(length, 8500, 0.2)
        elif voice_type == VoiceType.TR909_CRASH:
            drum = self._synth_noise_drum(length, 12000, 0.18)
        elif voice_type == VoiceType.TR707_KICK:
            drum = self._synth_kick(length, start_freq=130, end_freq=45, nonlinearity=1.0)
        elif voice_type == VoiceType.TR707_SNARE:
            drum = self._synth_707_snare(length)
        elif voice_type == VoiceType.TR707_HIHAT:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.06), 9500, 0.3)
        elif voice_type == VoiceType.TR707_TOM:
            drum = self._synth_tonal_drum(length, 110, 4.0)
        elif voice_type == VoiceType.LINN_KICK:
            drum = self._synth_kick(length, start_freq=160, end_freq=42, nonlinearity=1.3)
        elif voice_type == VoiceType.LINN_SNARE:
            drum = self._synth_linn_snare(length)
        elif voice_type == VoiceType.LINN_HIHAT:
            drum = self._synth_noise_drum(int(self.sample_rate * 0.055), 8800, 0.32)
        elif voice_type == VoiceType.LINN_CLAP:
            drum = self._synth_clap(length)
        elif voice_type == VoiceType.GLITCH_KICK:
            drum = self._synth_glitch_kick(length)
        elif voice_type == VoiceType.GLITCH_SNARE:
            drum = self._synth_glitch_snare(length)
        elif voice_type == VoiceType.GLITCH_HIHAT:
            drum = self._synth_glitch_hihat(length)
        elif voice_type == VoiceType.GLITCH_PERC:
            drum = self._synth_glitch_perc(length)
        else:
            drum = np.zeros(length, dtype=np.float32)
        
        drum = self._apply_adsr_to_drum(drum, velocity, adsr, voice)
        return drum
    
    def _apply_adsr_to_drum(self, drum: np.ndarray, velocity: float, adsr: ADSR, voice: ChannelVoice) -> np.ndarray:
        length = len(drum)
        envelope = np.ones(length, dtype=np.float32)
        
        attack_samples = int(adsr.attack * self.sample_rate)
        release_samples = int(adsr.release * self.sample_rate * voice.drum_release_envelope)
        
        if attack_samples > 0 and attack_samples < length:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        if release_samples > 0 and release_samples < length:
            release_start = length - release_samples
            envelope[release_start:] = np.linspace(1, 0, release_samples)
        
        drum = drum * envelope * velocity
        
        # Apply additional click prevention fade-in (very short)
        click_fade_samples = min(64, length)  # ~1.5ms at 44.1kHz
        if click_fade_samples > 0:
            t = np.arange(click_fade_samples) / click_fade_samples
            click_fade = t * t * (3.0 - 2.0 * t)  # Smoothstep
            drum[:click_fade_samples] *= click_fade
        
        if voice.drum_filter.enabled:
            drum = self._apply_drum_filter(drum, voice.drum_filter)
        
        return drum
    
    def _apply_drum_filter(self, drum: np.ndarray, drum_filter: DrumFilter) -> np.ndarray:
        """Apply post-generation filter to drum samples"""
        if not drum_filter.enabled:
            return drum
            
        nyquist = self.sample_rate / 2
        cutoff_norm = drum_filter.cutoff / nyquist
        cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
        
        try:
            Q = max(drum_filter.resonance, 0.707)  # Minimum Q for stability
            
            if drum_filter.filter_type == "lowpass":
                sos = signal.butter(2, cutoff_norm, btype='low', output='sos')
                filtered = signal.sosfilt(sos, drum)
                
            elif drum_filter.filter_type == "highpass":
                sos = signal.butter(2, cutoff_norm, btype='high', output='sos')
                filtered = signal.sosfilt(sos, drum)
                
            elif drum_filter.filter_type == "bandpass":
                # Bandpass with Q control
                bandwidth = cutoff_norm / Q
                low = max(0.01, cutoff_norm - bandwidth / 2)
                high = min(0.99, cutoff_norm + bandwidth / 2)
                
                if low < high:
                    sos = signal.butter(2, [low, high], btype='bandpass', output='sos')
                    filtered = signal.sosfilt(sos, drum)
                else:
                    return drum
            else:
                return drum
            
            # Apply resonance boost if > 1.0
            if drum_filter.resonance > 1.0:
                # Simple resonance simulation: boost around cutoff
                resonance_gain = 1.0 + (drum_filter.resonance - 1.0) * 0.3
                resonance_gain = min(resonance_gain, 2.0)  # Cap to prevent distortion
                filtered = filtered * resonance_gain
            
            # Normalize if clipping
            max_val = np.max(np.abs(filtered))
            if max_val > 1.0:
                filtered = filtered / max_val
                
            return filtered.astype(np.float32)
            
        except Exception:
            return drum
    
    def _synth_kick(self, length, start_freq=150, end_freq=40, nonlinearity=1.0):
        t = np.arange(length) / self.sample_rate
        freq_env = end_freq + (start_freq - end_freq) * np.exp(-t * 20)
        phase = 2 * np.pi * np.cumsum(freq_env) / self.sample_rate
        kick = np.sin(phase)
        amp_envelope = np.exp(-t * 4) * (1 + 3 * np.exp(-t * 50))
        kick = kick * amp_envelope
        cutoff = end_freq * 2
        alpha = 1 - np.exp(-2 * np.pi * cutoff / self.sample_rate)
        kick = signal.lfilter([alpha], [1, alpha - 1], kick)
        kick = np.tanh(nonlinearity * kick)
        max_val = np.max(np.abs(kick))
        if max_val > 0:
            kick = kick / max_val * 0.8
        return kick.astype(np.float32)
    
    def _synth_tonal_drum(self, length, frequency, nonlinearity=1.0):
        t = np.arange(length) / self.sample_rate
        freq_env = frequency * (1 + 1.5 * np.exp(-t * 25))
        phase = 2 * np.pi * np.cumsum(freq_env) / self.sample_rate
        drum_data = np.sin(phase)
        amp_envelope = np.exp(-t * 6) * (1 + 0.5 * np.exp(-t * 40))
        drum_data = drum_data * amp_envelope
        alpha = 1 - np.exp(-2 * np.pi * frequency / self.sample_rate)
        drum_data = signal.lfilter([alpha], [1, alpha - 1], drum_data)
        drum_data = np.tanh(nonlinearity * drum_data)
        max_val = np.max(np.abs(drum_data))
        if max_val > 0:
            drum_data = drum_data / max_val * 0.7
        return drum_data.astype(np.float32)
    
    def _synth_noise_drum(self, length, center_freq=8000, bandwidth=0.3):
        amp_envelope = np.exp(np.linspace(0, -10, length))
        noise = np.random.randn(length)
        low_freq = center_freq * (1 - bandwidth)
        high_freq = center_freq * (1 + bandwidth)
        nyquist = self.sample_rate / 2
        low = max(low_freq / nyquist, 0.01)
        high = min(high_freq / nyquist, 0.99)
        if low < high:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.lfilter(b, a, noise)
        else:
            filtered = noise
        drum_data = amp_envelope * filtered
        max_val = np.max(np.abs(drum_data))
        if max_val > 0:
            drum_data = drum_data / max_val * 0.5
        return drum_data.astype(np.float32)
    
    def _synth_808_snare(self, length):
        tone_len = int(length * 0.35)
        t = np.arange(tone_len) / self.sample_rate
        tone = (np.sin(2 * np.pi * 180 * t) + 0.7 * np.sin(2 * np.pi * 330 * t))
        tone_env = np.exp(-t * 18)
        tone = tone * tone_env
        tone = np.tanh(tone * 1.5)
        noise_len = int(length * 0.35)
        noise = np.random.randn(noise_len)
        b, a = signal.butter(2, [1500 / (self.sample_rate / 2), 5000 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 12)
        filtered = filtered * noise_env
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.4 * tone[:min_len] + 0.6 * filtered[:min_len]
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.5
        return combined
    
    def _synth_909_snare(self, length):
        tone_len = int(length * 0.3)
        t = np.arange(tone_len) / self.sample_rate
        tone = (np.sin(2 * np.pi * 200 * t) + 0.8 * np.sin(2 * np.pi * 350 * t))
        tone_env = np.exp(-t * 22)
        tone = tone * tone_env
        tone = np.tanh(tone * 2.0)
        noise_len = int(length * 0.3)
        noise = np.random.randn(noise_len)
        b, a = signal.butter(2, [2000 / (self.sample_rate / 2), 6000 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 15)
        filtered = filtered * noise_env
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.65 * filtered[:min_len]
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.5
        return combined
    
    def _synth_707_snare(self, length):
        tone_len = int(length * 0.25)
        t = np.arange(tone_len) / self.sample_rate
        tone = np.sin(2 * np.pi * 220 * t)
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        tone = np.tanh(tone * 1.2)
        noise_len = int(length * 0.25)
        noise = np.random.randn(noise_len)
        b, a = signal.butter(2, [1200 / (self.sample_rate / 2), 4500 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 18)
        filtered = filtered * noise_env
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.3 * tone[:min_len] + 0.7 * filtered[:min_len]
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.48
        return combined
    
    def _synth_linn_snare(self, length):
        tone_len = int(length * 0.32)
        t = np.arange(tone_len) / self.sample_rate
        tone = (np.sin(2 * np.pi * 240 * t) + 0.7 * np.sin(2 * np.pi * 380 * t))
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        tone = np.tanh(tone * 1.8)
        noise_len = int(length * 0.32)
        noise = np.random.randn(noise_len)
        b, a = signal.butter(2, [1800 / (self.sample_rate / 2), 5500 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 14)
        filtered = filtered * noise_env
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.65 * filtered[:min_len]
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.52
        return combined
    
    def _synth_clap(self, length):
        clap = np.zeros(length, dtype=np.float32)
        burst_times = [0, 0.03, 0.06]
        burst_length = int(self.sample_rate * 0.05)
        for i, offset in enumerate(burst_times):
            offset_samples = int(offset * self.sample_rate)
            if offset_samples + burst_length > length:
                break
            noise = np.random.randn(burst_length)
            b, a = signal.butter(4, [500 / (self.sample_rate / 2), 2500 / (self.sample_rate / 2)], btype='band')
            filtered = signal.lfilter(b, a, noise)
            env = np.exp(np.linspace(0, -15, burst_length))
            amp = 1.0 - i * 0.2
            clap[offset_samples:offset_samples + burst_length] += filtered * env * amp
        max_val = np.max(np.abs(clap))
        if max_val > 0:
            clap = clap / max_val * 0.75
        return clap
    
    def _synth_fm_cowbell(self, length):
        carrier_freq = 540
        mod_freq = 800
        mod_index = 1.5
        t = np.arange(length) / self.sample_rate
        modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)
        envelope = np.exp(-t * 6)
        cowbell = carrier * envelope
        max_val = np.max(np.abs(cowbell))
        if max_val > 0:
            cowbell = cowbell / max_val * 0.5
        return cowbell.astype(np.float32)
    
    def _synth_glitch_kick(self, length):
        freq = 80 * np.exp(np.linspace(0, -3, length))
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        square = np.sign(np.sin(phase))
        envelope = np.exp(np.linspace(0, -8, length))
        kick = square * envelope
        kick = np.round(kick * 4) / 4
        return (kick * 0.7).astype(np.float32)
    
    def _synth_glitch_snare(self, length):
        tone_len = int(length * 0.4)
        t = np.arange(tone_len) / self.sample_rate
        tone_freq = 150 * np.exp(-t * 15)
        phase = 2 * np.pi * np.cumsum(tone_freq) / self.sample_rate
        tone = np.sign(np.sin(phase))
        tone_env = np.exp(-t * 18)
        tone = tone * tone_env
        tone = np.round(tone * 6) / 6
        noise_len = int(length * 0.4)
        noise = np.random.randn(noise_len)
        noise = np.round(noise * 8) / 8
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 20)
        noise = noise * noise_env
        min_len = min(len(tone), len(noise))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.3 * tone[:min_len] + 0.7 * noise[:min_len]
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.6
        return combined
    
    def _synth_glitch_hihat(self, length):
        noise = np.random.randn(length)
        noise = np.round(noise * 4) / 4
        envelope = np.exp(np.linspace(0, -20, length))
        hihat = noise * envelope
        return (hihat * 0.4).astype(np.float32)
    
    def _synth_glitch_perc(self, length):
        freq = 440 * np.exp(np.linspace(0, -4, length))
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        tri = 2.0 * np.abs(2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0) - 1.0
        tri = np.round(tri * 6) / 6
        envelope = np.exp(np.linspace(0, -10, length))
        perc = tri * envelope
        return (perc * 0.6).astype(np.float32)


def get_voice_categories():
    return {
        "Synth": [
            VoiceType.SINE,
            VoiceType.SQUARE,
            VoiceType.SAW,
            VoiceType.TRIANGLE,
            VoiceType.NOISE,
        ],
        "TR-808": [
            VoiceType.TR808_KICK,
            VoiceType.TR808_SNARE,
            VoiceType.TR808_CLAP,
            VoiceType.TR808_HIHAT_CLOSED,
            VoiceType.TR808_HIHAT_OPEN,
            VoiceType.TR808_TOM_LOW,
            VoiceType.TR808_TOM_MID,
            VoiceType.TR808_TOM_HI,
            VoiceType.TR808_CYMBAL,
            VoiceType.TR808_COWBELL,
        ],
        "TR-909": [
            VoiceType.TR909_KICK,
            VoiceType.TR909_SNARE,
            VoiceType.TR909_CLAP,
            VoiceType.TR909_HIHAT_CLOSED,
            VoiceType.TR909_HIHAT_OPEN,
            VoiceType.TR909_RIDE,
            VoiceType.TR909_CRASH,
        ],
        "TR-707": [
            VoiceType.TR707_KICK,
            VoiceType.TR707_SNARE,
            VoiceType.TR707_HIHAT,
            VoiceType.TR707_TOM,
        ],
        "LinnDrum": [
            VoiceType.LINN_KICK,
            VoiceType.LINN_SNARE,
            VoiceType.LINN_HIHAT,
            VoiceType.LINN_CLAP,
        ],
        "Glitch": [
            VoiceType.GLITCH_KICK,
            VoiceType.GLITCH_SNARE,
            VoiceType.GLITCH_HIHAT,
            VoiceType.GLITCH_PERC,
        ],
    }

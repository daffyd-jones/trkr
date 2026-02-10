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
from collections import deque


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


@dataclass
class EffectParams:
    """Base class for effect parameters"""
    enabled: bool = False
    wet_mix: float = 0.5  # 0.0 to 1.0


@dataclass
class ChorusParams(EffectParams):
    """Chorus effect parameters"""
    rate: float = 1.5  # Hz
    depth: float = 0.02  # seconds
    feedback: float = 0.1


@dataclass
class DelayParams(EffectParams):
    """Delay effect parameters"""
    time: float = 0.3  # seconds
    feedback: float = 0.4
    cross_feedback: float = 0.0


@dataclass
class ReverbParams(EffectParams):
    """Reverb effect parameters"""
    room_size: float = 0.5  # 0.0 to 1.0
    damping: float = 0.5  # 0.0 to 1.0
    width: float = 1.0  # 0.0 to 1.0


@dataclass
class CompressionParams(EffectParams):
    """Compression effect parameters"""
    threshold: float = -20.0  # dB
    ratio: float = 4.0  # 1.0 to 20.0
    attack: float = 0.005  # seconds
    release: float = 0.1  # seconds
    makeup_gain: float = 0.0  # dB


@dataclass
class CrushParams(EffectParams):
    """Bit crusher effect parameters"""
    bits: int = 8  # 1 to 16
    downsample: int = 1  # 1 to 32


@dataclass
class EffectsBus:
    """Single effects bus with parameters and state"""
    effect_type: EffectType
    params: EffectParams
    buffer_left: deque = field(default_factory=lambda: deque(maxlen=44100))  # 1 second buffer
    buffer_right: deque = field(default_factory=lambda: deque(maxlen=44100))
    
    def __post_init__(self):
        # Initialize proper parameters based on effect type
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
    """ADSR envelope parameters (all in seconds)"""
    attack: float = 0.01
    decay: float = 0.1
    sustain: float = 0.7  # 0.0 to 1.0
    release: float = 0.2


@dataclass
class Filter:
    """Filter parameters"""
    cutoff: float = 8000.0  # Hz
    resonance: float = 1.0  # Q factor
    filter_type: str = "lowpass"  # lowpass, highpass, bandpass


@dataclass
class DrumFilter:
    """Drum-specific post-generation filter parameters"""
    enabled: bool = False
    cutoff: float = 8000.0  # Hz
    resonance: float = 1.0  # Q factor
    filter_type: str = "lowpass"  # lowpass, highpass, bandpass


@dataclass
class ChannelVoice:
    """Voice configuration for a channel"""
    voice_type: VoiceType = VoiceType.SINE
    adsr: ADSR = field(default_factory=ADSR)
    filter: Filter = field(default_factory=Filter)
    volume: float = 0.8  # 0.0 to 1.0
    pan: float = 0.5  # 0.0 (left) to 1.0 (right)
    detune: float = 0.0  # cents
    
    # Drum-specific parameters
    drum_length_multiplier: float = 1.0  # 0.1 to 2.0, for shortening kick/snare hits
    drum_release_envelope: float = 1.0  # 0.0 to 2.0, release multiplier for drums
    drum_filter: DrumFilter = field(default_factory=DrumFilter)
    
    # Effects sends (0.0 to 1.0 for each effect bus)
    send_chorus: float = 0.0
    send_delay: float = 0.0
    send_reverb: float = 0.0
    send_compression: float = 0.0
    send_crush: float = 0.0
    


class SynthEngine:
    """Main synthesis engine"""

    _COMB_DELAYS_L = [1557, 1617, 1491, 1422]  # samples
    _COMB_DELAYS_R = [1557 + 23, 1617 + 23, 1491 + 23, 1422 + 23]
    _ALLPASS_DELAYS = [225, 556]
    _ALLPASS_GAIN = 0.5
    
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.global_volume = 0.7
        
        # 8 channel voices
        self.channels = [ChannelVoice() for _ in range(8)]
        
        # Effects buses (5 parallel effects)
        self.effects_buses = {
            EffectType.CHORUS: EffectsBus(EffectType.CHORUS, ChorusParams()),
            EffectType.DELAY: EffectsBus(EffectType.DELAY, DelayParams()),
            EffectType.REVERB: EffectsBus(EffectType.REVERB, ReverbParams()),
            EffectType.COMPRESSION: EffectsBus(EffectType.COMPRESSION, CompressionParams()),
            EffectType.CRUSH: EffectsBus(EffectType.CRUSH, CrushParams()),
        }
        
        self._effect_state = {
            EffectType.CHORUS: {'lfo_phase': 0.0},
            EffectType.DELAY: {},   # delay uses the deque buffers directly
            # EffectType.REVERB: {},  # reverb uses the deque buffers directly
            EffectType.REVERB: {
                'comb_buffers': None,   # initialised on first call
                'comb_indices': None,
                'comb_filters': None,   # one-pole LP state per comb
                'allpass_buffers': None,
                'allpass_indices': None,
            },
            EffectType.COMPRESSION: {'env_left': 0.0, 'env_right': 0.0},
            EffectType.CRUSH: {},
        }

        # Active notes per channel
        self.active_notes: Dict[int, List[Dict]] = {i: [] for i in range(8)}
        
        # Audio stream
        self.stream = None
        self.running = False
        
        # Note queue for thread-safe note triggering
        self.note_queue = queue.Queue()
        
    def start(self):
        """Start the audio stream"""
        if self.running:
            return
            
        self.running = True
        self.stream = sd.OutputStream(
            channels=2,  # Stereo
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.stream.start()
        
    def stop(self):
        """Stop the audio stream"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
    def note_on(self, channel: int, note: int, velocity: int):
        """Trigger a note on a channel"""
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
        """Release a note on a channel"""
        if not 0 <= channel < 8:
            return
            
        self.note_queue.put({
            'action': 'note_off',
            'channel': channel,
            'note': note,
            'time': time.time()
        })
        
    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for sounddevice"""
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
        
        # Generate audio
        output = np.zeros((frames, 2), dtype=np.float32)
        
        for ch_idx in range(8):
            channel_out = self._render_channel(ch_idx, frames)
            if channel_out is not None:
                output += channel_out
                
        # Apply global volume
        output *= self.global_volume
        
        # Soft clip
        output = np.tanh(output)
        
        outdata[:] = output
        
    def _trigger_note(self, channel: int, note: int, velocity: float):
        """Add a new active note to a channel"""
        voice = self.channels[channel]
        
        # For drums, pre-render the entire sample
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
                'phase': 0.0,  # This will track phase in radians (0 to 2π)
                'envelope_phase': 0.0,
                'state': 'attack',
                'release_time': None,
                'is_drum': False,
                'filter_state': 0.0  # Each note gets its own filter state
            }
        
        self.active_notes[channel].append(note_data)
        
    def _release_note(self, channel: int, note: int):
        """Release a note (enter release phase)"""
        for note_data in self.active_notes[channel]:
            if note_data['note'] == note and not note_data.get('is_drum'):
                if note_data['state'] != 'release':
                    note_data['state'] = 'release'
                    note_data['release_time'] = time.time()
                
    def _render_channel(self, channel: int, frames: int) -> Optional[np.ndarray]:
        """Render audio for a channel"""
        if not self.active_notes[channel]:
            return None
            
        voice = self.channels[channel]
        output = np.zeros((frames, 2), dtype=np.float32)
        
        notes_to_remove = []
        
        for note_data in self.active_notes[channel]:
            if note_data.get('is_drum'):
                # Play back pre-rendered drum sample
                drum_sample = note_data['drum_sample']
                start_idx = note_data['phase']
                end_idx = min(start_idx + frames, len(drum_sample))
                
                if start_idx >= len(drum_sample):
                    notes_to_remove.append(note_data)
                    continue
                    
                chunk_len = end_idx - start_idx
                signal = drum_sample[start_idx:end_idx]
                
                # Apply channel volume
                signal *= voice.volume
                
                # Apply panning
                left_gain = np.sqrt(1.0 - voice.pan)
                right_gain = np.sqrt(voice.pan)
                
                output[:chunk_len, 0] += signal * left_gain
                output[:chunk_len, 1] += signal * right_gain
                
                note_data['phase'] = end_idx
                
            else:
                # Synth voice - generate oscillator
                signal = self._generate_oscillator(voice.voice_type, note_data, frames, voice.detune)
                
                # Apply envelope
                envelope = self._generate_envelope(note_data, frames, voice.adsr)
                signal *= envelope
                
                # Apply filter with per-note state tracking
                signal, new_filter_state = self._apply_filter(signal, voice.filter, note_data.get('filter_state', 0.0))
                note_data['filter_state'] = new_filter_state
                
                # Apply velocity
                signal *= note_data['velocity']
                
                # Apply channel volume
                signal *= voice.volume
                
                # CRITICAL: Apply anti-click fade AFTER all processing
                # This ensures any filter transients are also smoothed out
                if 'sample_count' not in note_data:
                    note_data['sample_count'] = 0
                
                # Very long exponential fade-in for maximum smoothness
                fade_duration = 512  # ~11.6ms at 44.1kHz
                
                if note_data['sample_count'] < fade_duration:
                    fade_samples = min(fade_duration - note_data['sample_count'], frames)
                    
                    # Exponential curve is smoother than polynomial for audio
                    t = np.linspace(
                        note_data['sample_count'] / fade_duration,
                        min((note_data['sample_count'] + fade_samples) / fade_duration, 1.0),
                        fade_samples
                    )
                    # Exponential fade: 1 - e^(-6*t) approaches 1 smoothly
                    fade_curve = 1.0 - np.exp(-6.0 * t)
                    
                    signal[:fade_samples] *= fade_curve
                
                note_data['sample_count'] += frames
                
                # Apply panning
                left_gain = np.sqrt(1.0 - voice.pan)
                right_gain = np.sqrt(voice.pan)
                
                output[:, 0] += signal * left_gain
                output[:, 1] += signal * right_gain
                
                # Update envelope phase
                note_data['envelope_phase'] += frames / self.sample_rate
                
                # Remove finished notes
                if note_data['state'] == 'release':
                    if note_data['envelope_phase'] > (
                        voice.adsr.attack + voice.adsr.decay + voice.adsr.release
                    ):
                        notes_to_remove.append(note_data)
                    
        # Clean up finished notes
        for note_data in notes_to_remove:
            self.active_notes[channel].remove(note_data)
        
        # Apply effects processing if any sends are active
        if (voice.send_chorus > 0 or voice.send_delay > 0 or 
            voice.send_reverb > 0 or voice.send_compression > 0 or voice.send_crush > 0):
            output = self._apply_effects(output, voice)
            
        return output
        
    # def _apply_effects(self, input_signal: np.ndarray, voice: ChannelVoice) -> np.ndarray:
    #     """Apply all effects based on send amounts"""
    #     if input_signal is None or len(input_signal) == 0:
    #         return input_signal
            
    #     # Split stereo
    #     input_left = input_signal[:, 0]
    #     input_right = input_signal[:, 1]
        
    #     # Start with dry signal
    #     output_left = input_left.copy()
    #     output_right = input_right.copy()
        
    #     # Apply each effect based on send amount
    #     if voice.send_chorus > 0:
    #         chorus_left, chorus_right = self._process_chorus(
    #             input_left, input_right, self.effects_buses[EffectType.CHORUS].params
    #         )
    #         output_left += chorus_left * voice.send_chorus
    #         output_right += chorus_right * voice.send_chorus
            
    #     if voice.send_delay > 0:
    #         delay_left, delay_right = self._process_delay(
    #             input_left, input_right, self.effects_buses[EffectType.DELAY].params
    #         )
    #         output_left += delay_left * voice.send_delay
    #         output_right += delay_right * voice.send_delay
            
    #     if voice.send_reverb > 0:
    #         reverb_left, reverb_right = self._process_reverb(
    #             input_left, input_right, self.effects_buses[EffectType.REVERB].params
    #         )
    #         output_left += reverb_left * voice.send_reverb
    #         output_right += reverb_right * voice.send_reverb
            
    #     if voice.send_compression > 0:
    #         comp_left, comp_right = self._process_compression(
    #             input_left, input_right, self.effects_buses[EffectType.COMPRESSION].params
    #         )
    #         output_left += comp_left * voice.send_compression
    #         output_right += comp_right * voice.send_compression
            
    #     if voice.send_crush > 0:
    #         crush_left, crush_right = self._process_crush(
    #             input_left, input_right, self.effects_buses[EffectType.CRUSH].params
    #         )
    #         output_left += crush_left * voice.send_crush
    #         output_right += crush_right * voice.send_crush
        
    #     # Combine back to stereo
    #     output = np.column_stack((output_left, output_right))
        
    #     # Prevent clipping
    #     max_val = np.max(np.abs(output))
    #     if max_val > 1.0:
    #         output = output / max_val
            
    #     return output.astype(np.float32)

    # def _apply_effects(self, input_signal: np.ndarray, voice: ChannelVoice) -> np.ndarray:
    #     """Apply all effects based on send amounts"""
    #     if input_signal is None or len(input_signal) == 0:
    #         return input_signal

    #     input_left = input_signal[:, 0].copy()
    #     input_right = input_signal[:, 1].copy()

    #     output_left = input_left.copy()
    #     output_right = input_right.copy()

    #     # Pass the full bus object so processors can read/write persistent state
    #     if voice.send_chorus > 0:
    #         chorus_bus = self.effects_buses[EffectType.CHORUS]
    #         wet_left, wet_right = self._process_chorus(
    #             input_left, input_right, chorus_bus
    #         )
    #         output_left = output_left + wet_left * voice.send_chorus
    #         output_right = output_right + wet_right * voice.send_chorus

    #     if voice.send_delay > 0:
    #         delay_bus = self.effects_buses[EffectType.DELAY]
    #         wet_left, wet_right = self._process_delay(
    #             input_left, input_right, delay_bus
    #         )
    #         output_left = output_left + wet_left * voice.send_delay
    #         output_right = output_right + wet_right * voice.send_delay

    #     if voice.send_reverb > 0:
    #         reverb_bus = self.effects_buses[EffectType.REVERB]
    #         wet_left, wet_right = self._process_reverb(
    #             input_left, input_right, reverb_bus
    #         )
    #         output_left = output_left + wet_left * voice.send_reverb
    #         output_right = output_right + wet_right * voice.send_reverb

    #     if voice.send_compression > 0:
    #         comp_bus = self.effects_buses[EffectType.COMPRESSION]
    #         wet_left, wet_right = self._process_compression(
    #             input_left, input_right, comp_bus
    #         )
    #         output_left = output_left + wet_left * voice.send_compression
    #         output_right = output_right + wet_right * voice.send_compression

    #     if voice.send_crush > 0:
    #         crush_bus = self.effects_buses[EffectType.CRUSH]
    #         wet_left, wet_right = self._process_crush(
    #             input_left, input_right, crush_bus
    #         )
    #         output_left = output_left + wet_left * voice.send_crush
    #         output_right = output_right + wet_right * voice.send_crush

    #     output = np.column_stack((output_left, output_right))
    #     max_val = np.max(np.abs(output))
    #     if max_val > 1.0:
    #         output /= max_val

    #     return output.astype(np.float32)

    def _apply_effects(self, input_signal: np.ndarray, voice: ChannelVoice) -> np.ndarray:
        if input_signal is None or len(input_signal) == 0:
            return input_signal
    
        input_left  = input_signal[:, 0].copy()
        input_right = input_signal[:, 1].copy()
    
        output_left  = input_left.copy()
        output_right = input_right.copy()
    
        chorus_bus = self.effects_buses[EffectType.CHORUS]
        if voice.send_chorus > 0 and chorus_bus.params.enabled:
            wet_l, wet_r = self._process_chorus(input_left, input_right, chorus_bus)
            output_left  += wet_l * voice.send_chorus
            output_right += wet_r * voice.send_chorus
    
        delay_bus = self.effects_buses[EffectType.DELAY]
        if voice.send_delay > 0 and delay_bus.params.enabled:
            wet_l, wet_r = self._process_delay(input_left, input_right, delay_bus)
            output_left  += wet_l * voice.send_delay
            output_right += wet_r * voice.send_delay
    
        reverb_bus = self.effects_buses[EffectType.REVERB]
        if voice.send_reverb > 0 and reverb_bus.params.enabled:
            wet_l, wet_r = self._process_reverb(input_left, input_right, reverb_bus)
            output_left  += wet_l * voice.send_reverb
            output_right += wet_r * voice.send_reverb
    
        comp_bus = self.effects_buses[EffectType.COMPRESSION]
        if voice.send_compression > 0 and comp_bus.params.enabled:
            wet_l, wet_r = self._process_compression(input_left, input_right, comp_bus)
            output_left  += wet_l * voice.send_compression
            output_right += wet_r * voice.send_compression
    
        crush_bus = self.effects_buses[EffectType.CRUSH]
        if voice.send_crush > 0 and crush_bus.params.enabled:
            wet_l, wet_r = self._process_crush(input_left, input_right, crush_bus)
            output_left  += wet_l * voice.send_crush
            output_right += wet_r * voice.send_crush
    
        output  = np.column_stack((output_left, output_right))
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output /= max_val
    
        return output.astype(np.float32)
        
    def _generate_oscillator(self, voice_type: VoiceType, note_data: Dict, 
                            frames: int, detune: float) -> np.ndarray:
        """Generate oscillator waveform for synth voices with proper phase continuity"""
        # Calculate frequency with detune
        freq = 440.0 * (2.0 ** ((note_data['note'] - 69) / 12.0))
        freq *= (2.0 ** (detune / 1200.0))
        
        # Phase increment per sample
        phase_inc = 2.0 * np.pi * freq / self.sample_rate
        
        # Generate phase array starting from current phase
        phases = note_data['phase'] + np.arange(frames) * phase_inc
        
        # Update stored phase with wrap-around to prevent overflow
        note_data['phase'] = (note_data['phase'] + frames * phase_inc) % (2.0 * np.pi)
        
        # Generate waveform
        if voice_type == VoiceType.SINE:
            waveform = np.sin(phases).astype(np.float32)
            
        elif voice_type == VoiceType.SQUARE:
            waveform = np.sign(np.sin(phases)).astype(np.float32)
            
        elif voice_type == VoiceType.SAW:
            # Sawtooth: -1 to 1 ramp, starts at 0
            # Offset by 0.5 cycles so it starts at zero crossing
            t = ((phases / (2.0 * np.pi)) + 0.5) % 1.0  # 0 to 1, offset
            waveform = (2.0 * t - 1.0).astype(np.float32)
            
        elif voice_type == VoiceType.TRIANGLE:
            # Triangle: starts at 0, goes to 1, back through 0 to -1, back to 0
            # Offset by 0.75 cycles so it starts at zero crossing
            t = ((phases / (2.0 * np.pi)) + 0.75) % 1.0  # 0 to 1, offset
            waveform = (2.0 * np.abs(2.0 * t - 1.0) - 1.0).astype(np.float32)
            
        elif voice_type == VoiceType.NOISE:
            waveform = (np.random.uniform(-1.0, 1.0, frames)).astype(np.float32)
        else:
            waveform = np.zeros(frames, dtype=np.float32)
        
        return waveform
        
    def _is_drum_voice(self, voice_type: VoiceType) -> bool:
        """Check if voice type is a drum sound"""
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
    
    # ──────────────────────────────────────────────────────────────────
    # DRUM SAMPLE RENDERING (inspired by reference materials)
    # ──────────────────────────────────────────────────────────────────
    
    def _render_drum_sample(self, voice_type: VoiceType, note: int, velocity: float, adsr: ADSR, voice: ChannelVoice) -> np.ndarray:
        """Render a complete drum sample with ADSR envelope applied"""
        # Determine base sample length based on drum type
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
        
        # Apply drum length multiplier for kick and snare shortening
        if 'KICK' in voice_type.value or 'SNARE' in voice_type.value:
            length = int(base_length * voice.drum_length_multiplier)
        else:
            length = base_length
        
        # Generate raw drum sound
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
        
        # Apply ADSR envelope for expression
        drum = self._apply_adsr_to_drum(drum, velocity, adsr, voice)
        
        return drum
    
    def _apply_adsr_to_drum(self, drum: np.ndarray, velocity: float, adsr: ADSR, voice: ChannelVoice) -> np.ndarray:
        """Apply ADSR envelope to drum sample for expression control"""
        length = len(drum)
        envelope = np.ones(length, dtype=np.float32)
        
        attack_samples = int(adsr.attack * self.sample_rate)
        release_samples = int(adsr.release * self.sample_rate * voice.drum_release_envelope)
        
        # Attack phase
        if attack_samples > 0 and attack_samples < length:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release phase (from end) - use drum release envelope multiplier
        if release_samples > 0 and release_samples < length:
            release_start = length - release_samples
            envelope[release_start:] = np.linspace(1, 0, release_samples)
        
        # Apply velocity and envelope
        drum = drum * envelope * velocity
        
        # Apply post-generation drum filter if enabled
        if voice.drum_filter.enabled:
            drum = self._apply_drum_filter(drum, voice.drum_filter)
        
        return drum
    
    def _apply_drum_filter(self, drum: np.ndarray, drum_filter: DrumFilter) -> np.ndarray:
        """Apply post-generation filter to drum samples"""
        if not drum_filter.enabled:
            return drum
            
        nyquist = self.sample_rate / 2
        cutoff_norm = drum_filter.cutoff / nyquist
        
        # Ensure cutoff is within valid range
        cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
        
        try:
            if drum_filter.filter_type == "lowpass":
                b, a = signal.butter(2, cutoff_norm, btype='low')
            elif drum_filter.filter_type == "highpass":
                b, a = signal.butter(2, cutoff_norm, btype='high')
            elif drum_filter.filter_type == "bandpass":
                # For bandpass, create a band around the cutoff frequency
                low = max(0.01, cutoff_norm * 0.5)
                high = min(0.99, cutoff_norm * 1.5)
                b, a = signal.butter(2, [low, high], btype='band')
            else:
                return drum
                
            # Apply filter with resonance
            filtered = signal.lfilter(b, a, drum)
            
            # Apply resonance boost if needed (simplified approach)
            if drum_filter.resonance > 1.0:
                # Add some resonance by mixing in a small amount of the filtered signal
                resonance_amount = min(0.3, (drum_filter.resonance - 1.0) * 0.1)
                filtered = filtered * (1.0 + resonance_amount)
                
            return filtered
            
        except Exception:
            # If filtering fails, return original drum
            return drum
    
    # ──────────────────────────────────────────────────────────────────
    # CORE DRUM SYNTHESIS FUNCTIONS (based on references)
    # ──────────────────────────────────────────────────────────────────
    
    def _synth_kick(self, length, start_freq=150, end_freq=40, nonlinearity=1.0):
        """
        Kick drum synthesis with proper pitch sweep
        """
        t = np.arange(length) / self.sample_rate
        
        # Pitch envelope: exponential sweep from start_freq to end_freq
        # Fast sweep in first ~100ms for the "punch"
        freq_env = end_freq + (start_freq - end_freq) * np.exp(-t * 20)
        
        # Generate phase by integrating frequency
        phase = 2 * np.pi * np.cumsum(freq_env) / self.sample_rate
        
        # Sine wave oscillator
        kick = np.sin(phase)
        
        # Amplitude envelope with attack transient for punch
        amp_envelope = np.exp(-t * 4) * (1 + 3 * np.exp(-t * 50))
        
        kick = kick * amp_envelope
        
        # Lowpass filter to smooth
        cutoff = end_freq * 2
        alpha = 1 - np.exp(-2 * np.pi * cutoff / self.sample_rate)
        kick = signal.lfilter([alpha], [1, alpha - 1], kick)
        
        # Nonlinearity for character/saturation
        kick = np.tanh(nonlinearity * kick)
        
        # Normalize
        max_val = np.max(np.abs(kick))
        if max_val > 0:
            kick = kick / max_val * 0.8
        
        return kick.astype(np.float32)
    
    def _synth_tonal_drum(self, length, frequency, nonlinearity=1.0):
        """
        Tonal drum synthesis (based on reference material)
        Used for toms, etc.
        """
        t = np.arange(length) / self.sample_rate
        
        # Pitch envelope with quick drop for attack
        freq_env = frequency * (1 + 1.5 * np.exp(-t * 25))
        
        # Generate phase
        phase = 2 * np.pi * np.cumsum(freq_env) / self.sample_rate
        
        # Sine wave
        drum_data = np.sin(phase)
        
        # Amplitude envelope
        amp_envelope = np.exp(-t * 6) * (1 + 0.5 * np.exp(-t * 40))
        drum_data = drum_data * amp_envelope
        
        # Apply lowpass filter (leaky integrator)
        alpha = 1 - np.exp(-2 * np.pi * frequency / self.sample_rate)
        drum_data = signal.lfilter([alpha], [1, alpha - 1], drum_data)
        
        # Apply nonlinearity for character
        drum_data = np.tanh(nonlinearity * drum_data)
        
        # Normalize
        max_val = np.max(np.abs(drum_data))
        if max_val > 0:
            drum_data = drum_data / max_val * 0.7
        
        return drum_data.astype(np.float32)
    
    def _synth_noise_drum(self, length, center_freq=8000, bandwidth=0.3):
        """
        Noise-based drum (hi-hats, cymbals, etc.)
        Uses bandpass filtering
        """
        # Amplitude envelope
        amp_envelope = np.exp(np.linspace(0, -10, length))
        
        # Generate noise
        noise = np.random.randn(length)
        
        # Bandpass filter
        low_freq = center_freq * (1 - bandwidth)
        high_freq = center_freq * (1 + bandwidth)
        
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low = max(low_freq / nyquist, 0.01)
        high = min(high_freq / nyquist, 0.99)
        
        if low < high:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.lfilter(b, a, noise)
        else:
            filtered = noise
        
        drum_data = amp_envelope * filtered
        
        # Normalize
        max_val = np.max(np.abs(drum_data))
        if max_val > 0:
            drum_data = drum_data / max_val * 0.5
        
        return drum_data.astype(np.float32)
    
    def _synth_808_snare(self, length):
        """808 snare: tone + noise combination"""
        # Tonal component - reduced nonlinearity to avoid resonance
        tone_len = int(length * 0.35)
        t = np.arange(tone_len) / self.sample_rate
        
        # Two sine waves for body
        tone = (np.sin(2 * np.pi * 180 * t) + 0.7 * np.sin(2 * np.pi * 330 * t))
        tone_env = np.exp(-t * 18)
        tone = tone * tone_env
        
        # Gentle saturation
        tone = np.tanh(tone * 1.5)
        
        # Noise component - wider bandwidth, less resonant
        noise_len = int(length * 0.35)
        noise = np.random.randn(noise_len)
        
        # Wider bandpass filter for less resonance
        b, a = signal.butter(2, [1500 / (self.sample_rate / 2), 5000 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 12)
        filtered = filtered * noise_env
        
        # Combine (40% tone, 60% noise)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.4 * tone[:min_len] + 0.6 * filtered[:min_len]
        
        # Normalize to medium level
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.5
        
        return combined
    
    def _synth_909_snare(self, length):
        """909 snare: sharper than 808"""
        # Tonal component - crisp but not overly resonant
        tone_len = int(length * 0.3)
        t = np.arange(tone_len) / self.sample_rate
        
        # Two sine waves
        tone = (np.sin(2 * np.pi * 200 * t) + 0.8 * np.sin(2 * np.pi * 350 * t))
        tone_env = np.exp(-t * 22)
        tone = tone * tone_env
        
        # Moderate saturation
        tone = np.tanh(tone * 2.0)
        
        # Noise component - brighter, crisper
        noise_len = int(length * 0.3)
        noise = np.random.randn(noise_len)
        
        # Brighter bandpass
        b, a = signal.butter(2, [2000 / (self.sample_rate / 2), 6000 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 15)
        filtered = filtered * noise_env
        
        # Combine (35% tone, 65% noise)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.65 * filtered[:min_len]
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.5
        
        return combined
    
    def _synth_707_snare(self, length):
        """707 snare: more electronic"""
        # Tonal component
        tone_len = int(length * 0.25)
        t = np.arange(tone_len) / self.sample_rate
        
        tone = np.sin(2 * np.pi * 220 * t)
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        
        # Light saturation
        tone = np.tanh(tone * 1.2)
        
        # Noise component
        noise_len = int(length * 0.25)
        noise = np.random.randn(noise_len)
        
        # Mid-range bandpass
        b, a = signal.butter(2, [1200 / (self.sample_rate / 2), 4500 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 18)
        filtered = filtered * noise_env
        
        # Combine (30% tone, 70% noise)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.3 * tone[:min_len] + 0.7 * filtered[:min_len]
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.48
        
        return combined
    
    def _synth_linn_snare(self, length):
        """Linn snare: bright and crisp"""
        # Tonal component
        tone_len = int(length * 0.32)
        t = np.arange(tone_len) / self.sample_rate
        
        # Two sine waves for fullness
        tone = (np.sin(2 * np.pi * 240 * t) + 0.7 * np.sin(2 * np.pi * 380 * t))
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        
        # Moderate saturation
        tone = np.tanh(tone * 1.8)
        
        # Noise component
        noise_len = int(length * 0.32)
        noise = np.random.randn(noise_len)
        
        # Bright but not too narrow
        b, a = signal.butter(2, [1800 / (self.sample_rate / 2), 5500 / (self.sample_rate / 2)], btype='band')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 14)
        filtered = filtered * noise_env
        
        # Combine (35% tone, 65% noise)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.65 * filtered[:min_len]
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.52
        
        return combined
    
    def _synth_clap(self, length):
        """Hand clap: multiple noise bursts"""
        clap = np.zeros(length, dtype=np.float32)
        
        # Multiple bursts with decreasing amplitude
        burst_times = [0, 0.03, 0.06]
        burst_length = int(self.sample_rate * 0.05)
        
        for i, offset in enumerate(burst_times):
            offset_samples = int(offset * self.sample_rate)
            if offset_samples + burst_length > length:
                break
                
            # Generate filtered noise burst
            noise = np.random.randn(burst_length)
            
            # Bandpass filter
            b, a = signal.butter(4, [500 / (self.sample_rate / 2), 2500 / (self.sample_rate / 2)], btype='band')
            filtered = signal.lfilter(b, a, noise)
            
            # Envelope
            env = np.exp(np.linspace(0, -15, burst_length))
            
            # Decreasing amplitude for each burst
            amp = 1.0 - i * 0.2
            
            clap[offset_samples:offset_samples + burst_length] += filtered * env * amp
        
        # Normalize - slightly louder
        max_val = np.max(np.abs(clap))
        if max_val > 0:
            clap = clap / max_val * 0.75
        
        return clap
    
    def _synth_fm_cowbell(self, length):
        """FM synthesis cowbell (based on Chowning's techniques)"""
        # FM parameters for metallic sound
        carrier_freq = 540
        mod_freq = 800
        mod_index = 1.5
        
        t = np.arange(length) / self.sample_rate
        
        # Modulator
        modulator = mod_index * np.sin(2 * np.pi * mod_freq * t)
        
        # Carrier with FM
        carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)
        
        # Envelope
        envelope = np.exp(-t * 6)
        
        cowbell = carrier * envelope
        
        # Normalize
        max_val = np.max(np.abs(cowbell))
        if max_val > 0:
            cowbell = cowbell / max_val * 0.5
        
        return cowbell.astype(np.float32)
    
    def _synth_glitch_kick(self, length):
        """Glitchy kick using square wave"""
        # Pitch envelope
        freq = 80 * np.exp(np.linspace(0, -3, length))
        
        # Square wave
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        square = np.sign(np.sin(phase))
        
        # Envelope
        envelope = np.exp(np.linspace(0, -8, length))
        
        kick = square * envelope
        
        # Bit crushing
        kick = np.round(kick * 4) / 4
        
        return (kick * 0.7).astype(np.float32)
    
    def _synth_glitch_snare(self, length):
        """Glitchy snare with bit reduction"""
        # Add a tonal component for body
        tone_len = int(length * 0.4)
        t = np.arange(tone_len) / self.sample_rate
        
        # Square wave tone
        tone_freq = 150 * np.exp(-t * 15)  # Pitch sweep
        phase = 2 * np.pi * np.cumsum(tone_freq) / self.sample_rate
        tone = np.sign(np.sin(phase))
        tone_env = np.exp(-t * 18)
        tone = tone * tone_env
        
        # Bit crush the tone
        tone = np.round(tone * 6) / 6
        
        # Noise component
        noise_len = int(length * 0.4)
        noise = np.random.randn(noise_len)
        
        # Bit crush the noise
        noise = np.round(noise * 8) / 8
        
        # Envelope for noise
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 20)
        noise = noise * noise_env
        
        # Combine (30% tone, 70% noise)
        min_len = min(len(tone), len(noise))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.3 * tone[:min_len] + 0.7 * noise[:min_len]
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.6
        
        return combined
    
    def _synth_glitch_hihat(self, length):
        """Glitchy hi-hat"""
        noise = np.random.randn(length)
        
        # Bit crush
        noise = np.round(noise * 4) / 4
        
        # Envelope
        envelope = np.exp(np.linspace(0, -20, length))
        
        hihat = noise * envelope
        
        return (hihat * 0.4).astype(np.float32)
    
    def _synth_glitch_perc(self, length):
        """Glitchy percussion"""
        # Pitch envelope
        freq = 440 * np.exp(np.linspace(0, -4, length))
        
        # Triangle wave
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        tri = 2.0 * np.abs(2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0) - 1.0
        
        # Bit crush
        tri = np.round(tri * 6) / 6
        
        # Envelope
        envelope = np.exp(np.linspace(0, -10, length))
        
        perc = tri * envelope
        
        return (perc * 0.6).astype(np.float32)
    
    # ──────────────────────────────────────────────────────────────────
    # EFFECTS PROCESSING
    # ──────────────────────────────────────────────────────────────────
    
    # def _process_chorus(self, input_left: np.ndarray, input_right: np.ndarray, params: ChorusParams) -> tuple:
    #     """Process chorus effect with simple, stable implementation"""
    #     if not params.enabled:
    #         return input_left, input_right
            
    #     length = len(input_left)
    #     output_left = np.zeros_like(input_left)
    #     output_right = np.zeros_like(input_right)
        
    #     # Simple chorus with minimal buffering
    #     delay_samples = int(params.depth * self.sample_rate)
    #     if delay_samples < 1:
    #         delay_samples = 1
            
    #     # Create LFO for modulation
    #     t = np.arange(length) / self.sample_rate
    #     lfo = np.sin(2 * np.pi * params.rate * t)
        
    #     # Simple delay buffer using numpy arrays
    #     for i in range(length):
    #         # Calculate modulated delay
    #         mod_delay = delay_samples * (0.5 + 0.5 * lfo[i])
    #         mod_delay_samples = int(mod_delay)
            
    #         if i >= mod_delay_samples:
    #             # Simple delayed sample
    #             delayed_left = input_left[i - mod_delay_samples]
    #             delayed_right = input_right[i - mod_delay_samples]
    #         else:
    #             delayed_left = 0
    #             delayed_right = 0
            
    #         # Mix dry and wet signals
    #         output_left[i] = input_left[i] + delayed_left * params.wet_mix * 0.3
    #         output_right[i] = input_right[i] + delayed_right * params.wet_mix * 0.3
        
    #     return output_left, output_right

    def _process_chorus(self, input_left: np.ndarray, input_right: np.ndarray,
                        bus: EffectsBus) -> tuple:
        """
        Chorus effect with persistent LFO phase and cross-buffer delay history.
        Returns wet-only signal (no dry component).
        """
        params = bus.params
        length = len(input_left)

        # --- Persistent delay history ---
        # Extend the history deques if needed to cover maximum delay
        max_delay_samples = int((params.depth + 0.005) * self.sample_rate) + 4
        if bus.buffer_left.maxlen < max_delay_samples:
            # Recreate with larger maxlen, preserving contents
            old_l = list(bus.buffer_left)
            old_r = list(bus.buffer_right)
            bus.buffer_left = deque(old_l, maxlen=max_delay_samples * 2)
            bus.buffer_right = deque(old_r, maxlen=max_delay_samples * 2)

        # Pad history if it is shorter than max_delay_samples on first run
        while len(bus.buffer_left) < max_delay_samples:
            bus.buffer_left.appendleft(0.0)
            bus.buffer_right.appendleft(0.0)

        state = self._effect_state[EffectType.CHORUS]
        lfo_phase = state['lfo_phase']

        output_left = np.zeros(length, dtype=np.float32)
        output_right = np.zeros(length, dtype=np.float32)

        # Build a working copy of the history + current input so we can
        # index into it with simple integer offsets.
        # history[-1] is the most recent past sample.
        history_l = list(bus.buffer_left)  # oldest … newest
        history_r = list(bus.buffer_right)

        phase_inc = 2.0 * np.pi * params.rate / self.sample_rate
        # Right channel LFO is 90° offset for stereo spread
        lfo_phase_r = lfo_phase + (np.pi / 2.0)

        for i in range(length):
            # LFO value in [-1, 1]
            lfo_l = np.sin(lfo_phase)
            lfo_r = np.sin(lfo_phase_r)

            # Modulated delay in samples
            # Centre delay is depth/2, modulation swings ±depth/2
            centre = params.depth * self.sample_rate * 0.5
            half = centre
            delay_l = centre + half * lfo_l   # oscillates 0 … depth*sr
            delay_r = centre + half * lfo_r

            # Integer and fractional parts for linear interpolation
            d_int_l = int(delay_l)
            d_frac_l = delay_l - d_int_l
            d_int_r = int(delay_r)
            d_frac_r = delay_r - d_int_r

            hist_len = len(history_l)

            # Index into combined history + current-buffer-so-far
            # Positive i means we have i samples of the current buffer available
            def read_sample(hist, cur_buf, pos_from_now):
                """Read a sample that is pos_from_now samples in the past.
                pos_from_now=1 means one sample ago."""
                if pos_from_now <= i:
                    # Available in current buffer
                    idx = i - pos_from_now
                    return float(cur_buf[idx])
                else:
                    # Must read from history
                    hist_idx = hist_len - (pos_from_now - i)
                    if 0 <= hist_idx < hist_len:
                        return float(hist[hist_idx])
                    return 0.0

            # Current buffer accumulates output_left up to i (dry input used)
            # We read from input (dry) for the delay line, not from output,
            # to avoid feedback instability in the chorus.
            s0_l = read_sample(history_l, input_left, d_int_l)
            s1_l = read_sample(history_l, input_left, d_int_l + 1)
            delayed_l = s0_l + d_frac_l * (s1_l - s0_l)

            s0_r = read_sample(history_r, input_right, d_int_r)
            s1_r = read_sample(history_r, input_right, d_int_r + 1)
            delayed_r = s0_r + d_frac_r * (s1_r - s0_r)

            # Optional feedback (subtle, keep < 0.5 to avoid instability)
            feedback_l = delayed_l * params.feedback
            feedback_r = delayed_r * params.feedback

            output_left[i] = delayed_l + feedback_l
            output_right[i] = delayed_r + feedback_r

            lfo_phase = (lfo_phase + phase_inc) % (2.0 * np.pi)
            lfo_phase_r = (lfo_phase_r + phase_inc) % (2.0 * np.pi)

        # Update persistent state
        state['lfo_phase'] = lfo_phase

        # Push current input samples into history for next buffer
        for s in input_left:
            bus.buffer_left.append(float(s))
        for s in input_right:
            bus.buffer_right.append(float(s))

        return output_left, output_right
    
    # def _process_delay(self, input_left: np.ndarray, input_right: np.ndarray, params: DelayParams) -> tuple:
    #     """Process delay effect with simple, stable implementation"""
    #     if not params.enabled:
    #         return input_left, input_right
            
    #     length = len(input_left)
    #     delay_samples = int(params.time * self.sample_rate)
    #     if delay_samples < 1:
    #         delay_samples = 1
            
    #     output_left = np.zeros_like(input_left)
    #     output_right = np.zeros_like(input_right)
        
    #     # Simple delay using numpy arrays
    #     for i in range(length):
    #         # Get delayed sample
    #         if i >= delay_samples:
    #             delayed_left = output_left[i - delay_samples]  # Use output for feedback
    #             delayed_right = output_right[i - delay_samples]
    #         else:
    #             delayed_left = 0
    #             delayed_right = 0
            
    #         # Apply feedback
    #         feedback_left = delayed_left * params.feedback * 0.3
    #         feedback_right = delayed_right * params.feedback * 0.3
            
    #         # Mix dry signal with delayed+feedback signal
    #         output_left[i] = input_left[i] + (delayed_left + feedback_left) * params.wet_mix * 0.5
    #         output_right[i] = input_right[i] + (delayed_right + feedback_right) * params.wet_mix * 0.5
        
    #     return output_left, output_right

    def _process_delay(self, input_left: np.ndarray, input_right: np.ndarray,
                       bus: EffectsBus) -> tuple:
        """
        Delay with persistent circular buffer for feedback across audio blocks.
        Returns wet-only signal (echo tail, no dry component).
        """
        params = bus.params
        length = len(input_left)

        delay_samples = int(params.time * self.sample_rate)
        delay_samples = max(1, delay_samples)

        # Ensure the deque buffers are large enough to hold the full delay time
        required_size = delay_samples + length + 4
        if bus.buffer_left.maxlen < required_size:
            old_l = list(bus.buffer_left)
            old_r = list(bus.buffer_right)
            bus.buffer_left = deque(old_l, maxlen=required_size * 2)
            bus.buffer_right = deque(old_r, maxlen=required_size * 2)

        # Pre-fill with silence if history is too short
        while len(bus.buffer_left) < delay_samples:
            bus.buffer_left.appendleft(0.0)
            bus.buffer_right.appendleft(0.0)

        hist_l = list(bus.buffer_left)   # index 0 = oldest, -1 = most recent
        hist_r = list(bus.buffer_right)
        hist_len_l = len(hist_l)
        hist_len_r = len(hist_r)

        output_left = np.zeros(length, dtype=np.float32)
        output_right = np.zeros(length, dtype=np.float32)

        # We build a working buffer that is history + current output so that
        # feedback reads the delay line output (not dry input).
        # This is the standard comb-filter / feedback delay topology.
        work_l = list(hist_l)
        work_r = list(hist_r)

        for i in range(length):
            # Read from delay line: delay_samples behind the current write head
            read_pos_l = len(work_l) - delay_samples
            read_pos_r = len(work_r) - delay_samples

            echo_l = work_l[read_pos_l] if read_pos_l >= 0 else 0.0
            echo_r = work_r[read_pos_r] if read_pos_r >= 0 else 0.0

            # Write head: input + feedback from the delay line output
            # Clamp feedback to < 1.0 to prevent runaway
            fb = min(params.feedback, 0.95)
            new_l = float(input_left[i]) + echo_l * fb
            new_r = float(input_right[i]) + echo_r * fb

            work_l.append(new_l)
            work_r.append(new_r)

            # Wet output is the echo only (no dry)
            output_left[i] = echo_l
            output_right[i] = echo_r

        # Persist only the last `delay_samples` samples as history
        keep = delay_samples + 4
        new_hist_l = work_l[-keep:]
        new_hist_r = work_r[-keep:]

        bus.buffer_left.clear()
        bus.buffer_right.clear()
        for s in new_hist_l:
            bus.buffer_left.append(s)
        for s in new_hist_r:
            bus.buffer_right.append(s)

        return output_left, output_right
    
    # def _process_reverb(self, input_left: np.ndarray, input_right: np.ndarray, params: ReverbParams) -> tuple:
    #     """Process reverb effect with simple, stable implementation"""
    #     if not params.enabled:
    #         return input_left, input_right
            
    #     length = len(input_left)
    #     output_left = np.zeros_like(input_left)
    #     output_right = np.zeros_like(input_right)
        
    #     # Simple reverb with multiple delay taps
    #     delays = [0.03, 0.07, 0.15]  # seconds - reduced taps
    #     gains = [0.5, 0.3, 0.15]  # reduced gains
        
    #     # Process each delay tap
    #     for delay, gain in zip(delays, gains):
    #         delay_samples = int(delay * self.sample_rate)
    #         if delay_samples < 1:
    #             delay_samples = 1
                
    #         tap_left = np.zeros_like(input_left)
    #         tap_right = np.zeros_like(input_right)
            
    #         for i in range(length):
    #             if i >= delay_samples:
    #                 tap_left[i] = input_left[i - delay_samples] * gain * params.room_size * 0.2
    #                 tap_right[i] = input_right[i - delay_samples] * gain * params.room_size * 0.2
            
    #         output_left += tap_left
    #         output_right += tap_right
        
    #     # Apply damping
    #     damping_factor = 1.0 - (params.damping * 0.3)  # Reduced damping
    #     output_left *= damping_factor
    #     output_right *= damping_factor
        
    #     # Mix with dry signal
    #     output_left = input_left + output_left * params.wet_mix * 0.3
    #     output_right = input_right + output_right * params.wet_mix * 0.3
        
    #     return output_left, output_right

    # def _reverb_init_state(self, params: ReverbParams) -> dict:
    #     """Allocate reverb filter state buffers."""
    #     n_combs = len(self._COMB_DELAYS_L)
    #     n_allpass = len(self._ALLPASS_DELAYS)

    #     return {
    #         'comb_buf_l': [np.zeros(d, dtype=np.float64) for d in self._COMB_DELAYS_L],
    #         'comb_buf_r': [np.zeros(d, dtype=np.float64) for d in self._COMB_DELAYS_R],
    #         'comb_idx_l': [0] * n_combs,
    #         'comb_idx_r': [0] * n_combs,
    #         'comb_flt_l': [0.0] * n_combs,  # one-pole LP state per comb, left
    #         'comb_flt_r': [0.0] * n_combs,  # one-pole LP state per comb, right
    #         'ap_buf_l':   [np.zeros(d, dtype=np.float64) for d in self._ALLPASS_DELAYS],
    #         'ap_buf_r':   [np.zeros(d, dtype=np.float64) for d in self._ALLPASS_DELAYS],
    #         'ap_idx_l':   [0] * n_allpass,
    #         'ap_idx_r':   [0] * n_allpass,
    #     }

    # def _process_reverb(self, input_left: np.ndarray, input_right: np.ndarray,
    #                     bus: EffectsBus) -> tuple:
    #     """
    #     Freeverb-inspired reverb with feedback comb filters and allpass diffusers.
    #     Returns wet-only output.
    #     """
    #     params = bus.params
    #     state = self._effect_state[EffectType.REVERB]

    #     # Lazy initialisation of filter buffers
    #     if state.get('comb_buf_l') is None:
    #         state.update(self._reverb_init_state(params))

    #     length = len(input_left)
    #     out_l = np.zeros(length, dtype=np.float32)
    #     out_r = np.zeros(length, dtype=np.float32)

    #     # room_size controls feedback coefficient (0.0–1.0 → 0.7–0.98)
    #     feedback = 0.7 + params.room_size * 0.28
    #     feedback = min(feedback, 0.98)  # never exactly 1 (unstable)

    #     # damping controls one-pole LP cutoff inside the comb loop
    #     # damping=0 → no filtering (bright); damping=1 → heavy LP (dark)
    #     damp1 = params.damping * 0.4
    #     damp2 = 1.0 - damp1

    #     n_combs = len(self._COMB_DELAYS_L)
    #     n_allpass = len(self._ALLPASS_DELAYS)

    #     comb_buf_l = state['comb_buf_l']
    #     comb_buf_r = state['comb_buf_r']
    #     comb_idx_l = state['comb_idx_l']
    #     comb_idx_r = state['comb_idx_r']
    #     comb_flt_l = state['comb_flt_l']
    #     comb_flt_r = state['comb_flt_r']
    #     ap_buf_l   = state['ap_buf_l']
    #     ap_buf_r   = state['ap_buf_r']
    #     ap_idx_l   = state['ap_idx_l']
    #     ap_idx_r   = state['ap_idx_r']

    #     # Mix stereo input to mono for comb input (standard Freeverb approach)
    #     # Scale by 0.015 (Freeverb's fixedgain) to prevent accumulation overload
    #     input_gain = 0.015
    #     mono = (input_left + input_right) * input_gain

    #     for i in range(length):
    #         x = float(mono[i])
    #         sum_l = 0.0
    #         sum_r = 0.0

    #         # --- Parallel feedback comb filters ---
    #         for c in range(n_combs):
    #             # Left comb
    #             buf_out_l = comb_buf_l[c][comb_idx_l[c]]
    #             # One-pole damping filter inside the loop
    #             comb_flt_l[c] = buf_out_l * damp2 + comb_flt_l[c] * damp1
    #             comb_buf_l[c][comb_idx_l[c]] = x + comb_flt_l[c] * feedback
    #             sum_l += buf_out_l
    #             comb_idx_l[c] = (comb_idx_l[c] + 1) % len(comb_buf_l[c])

    #             # Right comb (slightly longer buffer for stereo width)
    #             buf_out_r = comb_buf_r[c][comb_idx_r[c]]
    #             comb_flt_r[c] = buf_out_r * damp2 + comb_flt_r[c] * damp1
    #             comb_buf_r[c][comb_idx_r[c]] = x + comb_flt_r[c] * feedback
    #             sum_r += buf_out_r
    #             comb_idx_r[c] = (comb_idx_r[c] + 1) % len(comb_buf_r[c])

    #         # --- Series allpass diffusers ---
    #         for a in range(n_allpass):
    #             # Left allpass
    #             buf_out_l = ap_buf_l[a][ap_idx_l[a]]
    #             ap_buf_l[a][ap_idx_l[a]] = sum_l + buf_out_l * _ALLPASS_GAIN
    #             sum_l = buf_out_l - sum_l * _ALLPASS_GAIN
    #             ap_idx_l[a] = (ap_idx_l[a] + 1) % len(ap_buf_l[a])

    #             # Right allpass
    #             buf_out_r = ap_buf_r[a][ap_idx_r[a]]
    #             ap_buf_r[a][ap_idx_r[a]] = sum_r + buf_out_r * _ALLPASS_GAIN
    #             sum_r = buf_out_r - sum_r * _ALLPASS_GAIN
    #             ap_idx_r[a] = (ap_idx_r[a] + 1) % len(ap_buf_r[a])

    #         out_l[i] = float(sum_l)
    #         out_r[i] = float(sum_r)

    #     # Width control: blend between mono and stereo wet signal
    #     # width=1.0 → full stereo; width=0.0 → mono wet
    #     w = params.width
    #     width_scale = w / 2.0 + 0.5
    #     mono_wet = (out_l + out_r) * 0.5
    #     out_l = mono_wet * (1.0 - width_scale) + out_l * width_scale
    #     out_r = mono_wet * (1.0 - width_scale) + out_r * width_scale

    #     return out_l.astype(np.float32), out_r.astype(np.float32)

    def _reverb_init_state(self) -> dict:
        """Allocate reverb filter state buffers."""
        return {
            'comb_buf_l': [np.zeros(d, dtype=np.float64)
                           for d in self._COMB_DELAYS_L],
            'comb_buf_r': [np.zeros(d, dtype=np.float64)
                           for d in self._COMB_DELAYS_R],
            'comb_idx_l': [0] * len(self._COMB_DELAYS_L),
            'comb_idx_r': [0] * len(self._COMB_DELAYS_R),
            'comb_flt_l': [0.0] * len(self._COMB_DELAYS_L),
            'comb_flt_r': [0.0] * len(self._COMB_DELAYS_R),
            'ap_buf_l':   [np.zeros(d, dtype=np.float64)
                           for d in self._ALLPASS_DELAYS],
            'ap_buf_r':   [np.zeros(d, dtype=np.float64)
                           for d in self._ALLPASS_DELAYS],
            'ap_idx_l':   [0] * len(self._ALLPASS_DELAYS),
            'ap_idx_r':   [0] * len(self._ALLPASS_DELAYS),
            'initialised': True,
        }

    def _process_reverb(self, input_left: np.ndarray, input_right: np.ndarray,
                        bus: EffectsBus) -> tuple:
        """
        Freeverb-inspired reverb with feedback comb filters and allpass diffusers.
        Returns wet-only output.
        """
        params = bus.params
        state  = self._effect_state[EffectType.REVERB]

        if not state.get('initialised'):
            state.update(self._reverb_init_state())

        length = len(input_left)
        out_l  = np.zeros(length, dtype=np.float32)
        out_r  = np.zeros(length, dtype=np.float32)

        feedback = 0.7 + params.room_size * 0.28
        feedback = min(feedback, 0.98)

        damp1 = params.damping * 0.4
        damp2 = 1.0 - damp1

        n_combs   = len(self._COMB_DELAYS_L)
        n_allpass = len(self._ALLPASS_DELAYS)

        comb_buf_l = state['comb_buf_l']
        comb_buf_r = state['comb_buf_r']
        comb_idx_l = state['comb_idx_l']
        comb_idx_r = state['comb_idx_r']
        comb_flt_l = state['comb_flt_l']
        comb_flt_r = state['comb_flt_r']
        ap_buf_l   = state['ap_buf_l']
        ap_buf_r   = state['ap_buf_r']
        ap_idx_l   = state['ap_idx_l']
        ap_idx_r   = state['ap_idx_r']

        input_gain = 0.015
        mono = (input_left + input_right) * input_gain

        for i in range(length):
            x     = float(mono[i])
            sum_l = 0.0
            sum_r = 0.0

            # Parallel feedback comb filters
            for c in range(n_combs):
                buf_out_l           = comb_buf_l[c][comb_idx_l[c]]
                comb_flt_l[c]       = buf_out_l * damp2 + comb_flt_l[c] * damp1
                comb_buf_l[c][comb_idx_l[c]] = x + comb_flt_l[c] * feedback
                sum_l              += buf_out_l
                comb_idx_l[c]       = (comb_idx_l[c] + 1) % len(comb_buf_l[c])

                buf_out_r           = comb_buf_r[c][comb_idx_r[c]]
                comb_flt_r[c]       = buf_out_r * damp2 + comb_flt_r[c] * damp1
                comb_buf_r[c][comb_idx_r[c]] = x + comb_flt_r[c] * feedback
                sum_r              += buf_out_r
                comb_idx_r[c]       = (comb_idx_r[c] + 1) % len(comb_buf_r[c])

            # Series allpass diffusers
            for a in range(n_allpass):
                buf_out_l              = ap_buf_l[a][ap_idx_l[a]]
                ap_buf_l[a][ap_idx_l[a]] = sum_l + buf_out_l * self._ALLPASS_GAIN
                sum_l                  = buf_out_l - sum_l * self._ALLPASS_GAIN
                ap_idx_l[a]            = (ap_idx_l[a] + 1) % len(ap_buf_l[a])

                buf_out_r              = ap_buf_r[a][ap_idx_r[a]]
                ap_buf_r[a][ap_idx_r[a]] = sum_r + buf_out_r * self._ALLPASS_GAIN
                sum_r                  = buf_out_r - sum_r * self._ALLPASS_GAIN
                ap_idx_r[a]            = (ap_idx_r[a] + 1) % len(ap_buf_r[a])

            out_l[i] = float(sum_l)
            out_r[i] = float(sum_r)

        # Stereo width blend
        w           = params.width
        width_scale = w / 2.0 + 0.5
        mono_wet    = (out_l + out_r) * 0.5
        out_l       = mono_wet * (1.0 - width_scale) + out_l * width_scale
        out_r       = mono_wet * (1.0 - width_scale) + out_r * width_scale

        return out_l.astype(np.float32), out_r.astype(np.float32)

    # def _process_compression(self, input_left: np.ndarray, input_right: np.ndarray, params: CompressionParams) -> tuple:
    #     """Process compression effect with improved implementation"""
    #     if not params.enabled:
    #         return input_left, input_right
            
    #     # Convert threshold from dB to linear
    #     threshold_linear = 10 ** (params.threshold / 20.0)
        
    #     # Attack and release time constants
    #     attack_coeff = np.exp(-1.0 / (params.attack * self.sample_rate))
    #     release_coeff = np.exp(-1.0 / (params.release * self.sample_rate))
        
    #     output_left = np.zeros_like(input_left)
    #     output_right = np.zeros_like(input_right)
        
    #     # Compression state variables
    #     envelope_left = 0
    #     envelope_right = 0
    #     gain_left = 1.0
    #     gain_right = 1.0
        
    #     for i in range(len(input_left)):
    #         # Get input levels
    #         input_level_left = abs(input_left[i])
    #         input_level_right = abs(input_right[i])
            
    #         # Update envelope followers
    #         if input_level_left > envelope_left:
    #             envelope_left = attack_coeff * envelope_left + (1 - attack_coeff) * input_level_left
    #         else:
    #             envelope_left = release_coeff * envelope_left + (1 - release_coeff) * input_level_left
                
    #         if input_level_right > envelope_right:
    #             envelope_right = attack_coeff * envelope_right + (1 - attack_coeff) * input_level_right
    #         else:
    #             envelope_right = release_coeff * envelope_right + (1 - release_coeff) * input_level_right
            
    #         # Calculate gain reduction
    #         if envelope_left > threshold_linear:
    #             gain_left = threshold_linear / envelope_left
    #             # Apply ratio
    #             gain_left = threshold_linear + (envelope_left - threshold_linear) / params.ratio
    #             gain_left = gain_left / envelope_left
    #         else:
    #             gain_left = 1.0
                
    #         if envelope_right > threshold_linear:
    #             gain_right = threshold_linear / envelope_right
    #             # Apply ratio
    #             gain_right = threshold_linear + (envelope_right - threshold_linear) / params.ratio
    #             gain_right = gain_right / envelope_right
    #         else:
    #             gain_right = 1.0
            
    #         # Apply compression
    #         output_left[i] = input_left[i] * gain_left
    #         output_right[i] = input_right[i] * gain_right
        
    #     # Apply makeup gain
    #     makeup_linear = 10 ** (params.makeup_gain / 20.0)
    #     output_left *= makeup_linear
    #     output_right *= makeup_linear
        
    #     # Apply wet/dry mix
    #     output_left = output_left * params.wet_mix + input_left * (1 - params.wet_mix)
    #     output_right = output_right * params.wet_mix + input_right * (1 - params.wet_mix)
        
    #     return output_left, output_right

    # def _process_compression(self, input_left: np.ndarray, input_right: np.ndarray,
    #                          bus: EffectsBus) -> tuple:
    #     """
    #     Feed-forward RMS compressor with persistent envelope state.
    #     Returns the gain-reduced signal (wet only, for parallel blend in _apply_effects).
    #     """
    #     params = bus.params
    #     state = self._effect_state[EffectType.COMPRESSION]

    #     length = len(input_left)
    #     output_left  = np.zeros(length, dtype=np.float32)
    #     output_right = np.zeros(length, dtype=np.float32)

    #     # Time constants — convert seconds to per-sample coefficients
    #     attack_coeff  = np.exp(-1.0 / max(params.attack  * self.sample_rate, 1))
    #     release_coeff = np.exp(-1.0 / max(params.release * self.sample_rate, 1))

    #     # dB conversion helpers
    #     def lin_to_db(x):
    #         return 20.0 * np.log10(max(x, 1e-9))

    #     def db_to_lin(x):
    #         return 10.0 ** (x / 20.0)

    #     # Threshold in dB
    #     threshold_db = params.threshold

    #     # Persistent state
    #     env_l = state.get('env_left',  0.0)
    #     env_r = state.get('env_right', 0.0)

    #     makeup = db_to_lin(params.makeup_gain)

    #     for i in range(length):
    #         in_l = float(input_left[i])
    #         in_r = float(input_right[i])

    #         # Peak envelope follower (using absolute value)
    #         lvl_l = abs(in_l)
    #         lvl_r = abs(in_r)

    #         # Attack on the way up, release on the way down
    #         coeff_l = attack_coeff  if lvl_l > env_l else release_coeff
    #         coeff_r = attack_coeff  if lvl_r > env_r else release_coeff
    #         env_l   = coeff_l * env_l + (1.0 - coeff_l) * lvl_l
    #         env_r   = coeff_r * env_r + (1.0 - coeff_r) * lvl_r

    #         # --- Gain computation in dB domain ---
    #         # Left
    #         env_db_l = lin_to_db(env_l)
    #         if env_db_l > threshold_db:
    #             # How many dB above threshold?
    #             over_l = env_db_l - threshold_db
    #             # Gain reduction: compress excess by ratio
    #             gain_db_l = threshold_db + over_l / params.ratio - env_db_l
    #             # gain_db_l is negative (reduction)
    #         else:
    #             gain_db_l = 0.0
    #         gain_l = db_to_lin(gain_db_l) * makeup

    #         # Right
    #         env_db_r = lin_to_db(env_r)
    #         if env_db_r > threshold_db:
    #             over_r = env_db_r - threshold_db
    #             gain_db_r = threshold_db + over_r / params.ratio - env_db_r
    #         else:
    #             gain_db_r = 0.0
    #         gain_r = db_to_lin(gain_db_r) * makeup

    #         output_left[i]  = in_l * gain_l
    #         output_right[i] = in_r * gain_r

    #     # Save persistent envelope state
    #     state['env_left']  = env_l
    #     state['env_right'] = env_r

    #     return output_left, output_right

    def _process_compression(self, input_left: np.ndarray, input_right: np.ndarray,
                              bus: EffectsBus) -> tuple:
        """
        Feed-forward peak compressor with linked stereo, soft knee, auto makeup gain,
        and persistent envelope state.
        Returns wet-only gain-reduced signal.
        """
        params = bus.params
        state  = self._effect_state[EffectType.COMPRESSION]

        length = len(input_left)
        output_left  = np.zeros(length, dtype=np.float32)
        output_right = np.zeros(length, dtype=np.float32)

        # --- Time constants ---
        attack_coeff  = np.exp(-1.0 / max(params.attack  * self.sample_rate, 1.0))
        release_coeff = np.exp(-1.0 / max(params.release * self.sample_rate, 1.0))

        # --- dB helpers ---
        def lin_to_db(x):
            return 20.0 * np.log10(max(abs(x), 1e-9))

        def db_to_lin(x):
            return 10.0 ** (x / 20.0)

        threshold_db = params.threshold   # e.g. -20 dB
        ratio        = max(params.ratio, 1.0)
        knee_db      = 6.0                # soft knee width in dB (fixed, feels natural)
        knee_half    = knee_db / 2.0

        # --- Auto makeup gain ---
        # Estimate the gain reduction at the threshold point and compensate.
        # This brings the compressed signal back to approximately the same
        # loudness as the input, making the parallel blend audible.
        # Formula: makeup ≈ -threshold * (1 - 1/ratio) / 2  (heuristic, works well)
        auto_makeup_db  = -threshold_db * (1.0 - 1.0 / ratio) * 0.5
        manual_makeup   = db_to_lin(params.makeup_gain)
        total_makeup    = db_to_lin(auto_makeup_db) * manual_makeup

        # --- Persistent state ---
        env = state.get('env_linked', 0.0)   # single linked envelope

        for i in range(length):
            in_l = float(input_left[i])
            in_r = float(input_right[i])

            # Linked stereo: drive with the louder channel
            lvl = max(abs(in_l), abs(in_r))

            # Peak envelope follower
            coeff = attack_coeff if lvl > env else release_coeff
            env   = coeff * env + (1.0 - coeff) * lvl

            # --- Gain computation in dB with soft knee ---
            env_db = lin_to_db(env)
            diff   = env_db - threshold_db   # how far above threshold

            if diff < -knee_half:
                # Below knee — no gain reduction
                gain_db = 0.0
            elif diff <= knee_half:
                # Inside soft knee — gradual onset
                # Quadratic interpolation across the knee region
                gain_db = (1.0 / ratio - 1.0) * ((diff + knee_half) ** 2) / (2.0 * knee_db)
            else:
                # Above knee — full ratio compression
                gain_db = threshold_db + diff / ratio - env_db

            gain = db_to_lin(gain_db) * total_makeup

            output_left[i]  = in_l * gain
            output_right[i] = in_r * gain

        # Save persistent state
        state['env_linked'] = env

        return output_left, output_right

    # def _process_crush(self, input_left: np.ndarray, input_right: np.ndarray, params: CrushParams) -> tuple:
    #     """Process bit crusher effect"""
    #     if not params.enabled:
    #         return input_left, input_right
            
    #     # Bit reduction
    #     max_val = (2 ** (params.bits - 1)) - 1
        
    #     # Quantize
    #     crushed_left = np.round(input_left * max_val) / max_val
    #     crushed_right = np.round(input_right * max_val) / max_val
        
    #     # Downsample
    #     if params.downsample > 1:
    #         crushed_left = crushed_left[::params.downsample]
    #         crushed_right = crushed_right[::params.downsample]
            
    #         # Stretch back to original length
    #         crushed_left = np.repeat(crushed_left, params.downsample)[:len(input_left)]
    #         crushed_right = np.repeat(crushed_right, params.downsample)[:len(input_right)]
        
    #     # Apply wet/dry mix
    #     output_left = crushed_left * params.wet_mix + input_left * (1 - params.wet_mix)
    #     output_right = crushed_right * params.wet_mix + input_right * (1 - params.wet_mix)
        
    #     return output_left, output_right

    def _process_crush(self, input_left: np.ndarray, input_right: np.ndarray,
                       bus: EffectsBus) -> tuple:
        """
        Bit crusher and sample rate reducer.
        Returns wet-only crushed signal.
        """
        params = bus.params
        length = len(input_left)
    
        # --- Bit reduction ---
        # bits=16 → essentially no change; bits=1 → 1-bit (sign only)
        levels = float(2 ** (params.bits - 1))
        crushed_left  = np.round(input_left  * levels) / levels
        crushed_right = np.round(input_right * levels) / levels
    
        # --- Sample rate reduction (zero-order hold) ---
        if params.downsample > 1:
            ds = int(params.downsample)
            # Build output sample-by-sample holding each sampled value
            # This is cleaner than the slice+repeat approach for non-divisible lengths
            held_l = np.zeros(length, dtype=np.float32)
            held_r = np.zeros(length, dtype=np.float32)
            current_l = 0.0
            current_r = 0.0
            for i in range(length):
                if i % ds == 0:
                    current_l = float(crushed_left[i])
                    current_r = float(crushed_right[i])
                held_l[i] = current_l
                held_r[i] = current_r
            crushed_left  = held_l
            crushed_right = held_r
    
        return crushed_left, crushed_right
    
    # ──────────────────────────────────────────────────────────────────
    # SYNTH VOICE HELPERS
    # ──────────────────────────────────────────────────────────────────
    
    def _generate_envelope(self, note_data: Dict, frames: int, 
                          adsr: ADSR) -> np.ndarray:
        """Generate ADSR envelope for synth voices"""
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
        """Apply simple lowpass filter with state continuity
        
        Returns:
            tuple: (filtered_signal, new_filter_state)
        """
        if filter_params.cutoff >= self.sample_rate / 2:
            return signal_in, signal_in[-1] if len(signal_in) > 0 else 0.0
            
        # One-pole lowpass with state tracking
        rc = 1.0 / (2.0 * np.pi * filter_params.cutoff)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)
        
        filtered = np.zeros_like(signal_in)
        
        # Initialize with previous state for continuity
        filtered[0] = prev_state + alpha * (signal_in[0] - prev_state)
        
        for i in range(1, len(signal_in)):
            filtered[i] = filtered[i-1] + alpha * (signal_in[i] - filtered[i-1])
        
        # Return filtered signal and final state for next buffer
        final_state = filtered[-1] if len(filtered) > 0 else prev_state
        
        return filtered, final_state


def get_voice_categories():
    """Return voice types organized by category"""
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

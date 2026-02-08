#!/usr/bin/env python3
"""
Improved Synth Engine for Terminal MIDI Phrase Tracker
Based on classic drum synthesis techniques:
- Proper resonant filtering
- Tone + noise combinations
- FM synthesis for metallic sounds
- ADSR envelope control for expression
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
class ChannelVoice:
    """Voice configuration for a channel"""
    voice_type: VoiceType = VoiceType.SINE
    adsr: ADSR = field(default_factory=ADSR)
    filter: Filter = field(default_factory=Filter)
    volume: float = 0.8  # 0.0 to 1.0
    pan: float = 0.5  # 0.0 (left) to 1.0 (right)
    detune: float = 0.0  # cents
    

class SynthEngine:
    """Main synthesis engine"""
    
    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.global_volume = 0.7
        
        # 8 channel voices
        self.channels = [ChannelVoice() for _ in range(8)]
        
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
            drum_sample = self._render_drum_sample(voice.voice_type, note, velocity, voice.adsr)
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
                'is_drum': False
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
                
                # Apply filter
                signal = self._apply_filter(signal, voice.filter)
                
                # Apply velocity
                signal *= note_data['velocity']
                
                # Apply channel volume
                signal *= voice.volume
                
                # Apply panning
                left_gain = np.sqrt(1.0 - voice.pan)
                right_gain = np.sqrt(voice.pan)
                
                output[:, 0] += signal * left_gain
                output[:, 1] += signal * right_gain
                
                # Update phase
                note_data['phase'] += frames
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
            
        return output
        
    def _generate_oscillator(self, voice_type: VoiceType, note_data: Dict, 
                            frames: int, detune: float) -> np.ndarray:
        """Generate oscillator waveform for synth voices"""
        # Calculate frequency with detune
        freq = 440.0 * (2.0 ** ((note_data['note'] - 69) / 12.0))
        freq *= (2.0 ** (detune / 1200.0))
        
        # Generate time array
        t = (np.arange(frames) + note_data['phase']) / self.sample_rate
        phase = 2.0 * np.pi * freq * t
        
        if voice_type == VoiceType.SINE:
            return np.sin(phase).astype(np.float32)
            
        elif voice_type == VoiceType.SQUARE:
            return np.sign(np.sin(phase)).astype(np.float32)
            
        elif voice_type == VoiceType.SAW:
            return (2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0).astype(np.float32)
            
        elif voice_type == VoiceType.TRIANGLE:
            return (2.0 * np.abs(2.0 * (phase / (2.0 * np.pi) % 1.0) - 1.0) - 1.0).astype(np.float32)
            
        elif voice_type == VoiceType.NOISE:
            return (np.random.uniform(-1.0, 1.0, frames)).astype(np.float32)
            
        return np.zeros(frames, dtype=np.float32)
        
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
    
    def _render_drum_sample(self, voice_type: VoiceType, note: int, velocity: float, adsr: ADSR) -> np.ndarray:
        """Render a complete drum sample with ADSR envelope applied"""
        # Determine sample length based on drum type
        if 'KICK' in voice_type.value or 'TOM' in voice_type.value:
            length = int(self.sample_rate * 0.8)
        elif 'SNARE' in voice_type.value or 'CLAP' in voice_type.value:
            length = int(self.sample_rate * 0.4)
        elif 'HIHAT' in voice_type.value and 'CLOSED' in voice_type.value:
            length = int(self.sample_rate * 0.15)
        elif 'HIHAT' in voice_type.value and 'OPEN' in voice_type.value:
            length = int(self.sample_rate * 0.6)
        elif 'CYMBAL' in voice_type.value or 'CRASH' in voice_type.value or 'RIDE' in voice_type.value:
            length = int(self.sample_rate * 1.5)
        else:
            length = int(self.sample_rate * 0.5)
        
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
            drum = self._synth_tonal_drum(length, 80, 8.0)
        elif voice_type == VoiceType.TR808_TOM_MID:
            drum = self._synth_tonal_drum(length, 110, 4.0)
        elif voice_type == VoiceType.TR808_TOM_HI:
            drum = self._synth_tonal_drum(length, 155, 4.0)
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
            drum = self._synth_tonal_drum(length, 90, 4.0)
            
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
        drum = self._apply_adsr_to_drum(drum, velocity, adsr)
        
        return drum
    
    def _apply_adsr_to_drum(self, drum: np.ndarray, velocity: float, adsr: ADSR) -> np.ndarray:
        """Apply ADSR envelope to drum sample for expression control"""
        length = len(drum)
        envelope = np.ones(length, dtype=np.float32)
        
        attack_samples = int(adsr.attack * self.sample_rate)
        release_samples = int(adsr.release * self.sample_rate)
        
        # Attack phase
        if attack_samples > 0 and attack_samples < length:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Release phase (from end)
        if release_samples > 0 and release_samples < length:
            release_start = length - release_samples
            envelope[release_start:] = np.linspace(1, 0, release_samples)
        
        # Apply velocity
        drum = drum * envelope * velocity
        
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
        # Tonal component - lower pitched
        tone_len = int(length * 0.35)
        t = np.arange(tone_len) / self.sample_rate
        
        # Lower frequencies for warmth
        tone = (np.sin(2 * np.pi * 150 * t) + 0.7 * np.sin(2 * np.pi * 280 * t))
        tone_env = np.exp(-t * 18)
        tone = tone * tone_env
        
        # Very gentle saturation
        tone = np.tanh(tone * 0.8)
        
        # Noise component - use highpass instead of bandpass to avoid resonance
        noise_len = int(length * 0.35)
        noise = np.random.randn(noise_len)
        
        # Simple highpass filter - no resonant peak
        b, a = signal.butter(1, 1200 / (self.sample_rate / 2), btype='high')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 12)
        filtered = filtered * noise_env
        
        # Impact transient (short high-freq click)
        impact_len = int(self.sample_rate * 0.002)  # 2ms
        impact = np.random.randn(impact_len) * np.exp(-np.arange(impact_len) / self.sample_rate * 500)
        
        # Combine (40% tone, 55% noise, 5% impact)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.4 * tone[:min_len] + 0.55 * filtered[:min_len]
        combined[:len(impact)] += 0.05 * impact
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.5
        
        return combined
    
    def _synth_909_snare(self, length):
        """909 snare: sharper than 808"""
        # Tonal component - lower pitched
        tone_len = int(length * 0.3)
        t = np.arange(tone_len) / self.sample_rate
        
        # Lower frequencies
        tone = (np.sin(2 * np.pi * 165 * t) + 0.8 * np.sin(2 * np.pi * 300 * t))
        tone_env = np.exp(-t * 22)
        tone = tone * tone_env
        
        # Light saturation
        tone = np.tanh(tone * 1.0)
        
        # Noise component - highpass for crisp character without resonance
        noise_len = int(length * 0.3)
        noise = np.random.randn(noise_len)
        
        # Simple highpass
        b, a = signal.butter(1, 1500 / (self.sample_rate / 2), btype='high')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 15)
        filtered = filtered * noise_env
        
        # Impact transient (short, bright click)
        impact_len = int(self.sample_rate * 0.0015)  # 1.5ms
        impact = np.random.randn(impact_len) * np.exp(-np.arange(impact_len) / self.sample_rate * 600)
        
        # Combine (35% tone, 57% noise, 8% impact)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.57 * filtered[:min_len]
        combined[:len(impact)] += 0.08 * impact
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.52
        
        return combined
    
    def _synth_707_snare(self, length):
        """707 snare: more electronic"""
        # Tonal component - lower pitched
        tone_len = int(length * 0.25)
        t = np.arange(tone_len) / self.sample_rate
        
        tone = np.sin(2 * np.pi * 180 * t)
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        
        # Very light saturation
        tone = np.tanh(tone * 0.6)
        
        # Noise component - simple highpass
        noise_len = int(length * 0.25)
        noise = np.random.randn(noise_len)
        
        # Simple highpass, no resonance
        b, a = signal.butter(1, 1000 / (self.sample_rate / 2), btype='high')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 18)
        filtered = filtered * noise_env
        
        # Impact transient
        impact_len = int(self.sample_rate * 0.002)
        impact = np.random.randn(impact_len) * np.exp(-np.arange(impact_len) / self.sample_rate * 500)
        
        # Combine (30% tone, 64% noise, 6% impact)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.3 * tone[:min_len] + 0.64 * filtered[:min_len]
        combined[:len(impact)] += 0.06 * impact
        
        # Normalize
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined = combined / max_val * 0.48
        
        return combined
    
    def _synth_linn_snare(self, length):
        """Linn snare: bright and crisp"""
        # Tonal component - lower pitched
        tone_len = int(length * 0.32)
        t = np.arange(tone_len) / self.sample_rate
        
        # Lower frequencies for warmth
        tone = (np.sin(2 * np.pi * 190 * t) + 0.7 * np.sin(2 * np.pi * 320 * t))
        tone_env = np.exp(-t * 20)
        tone = tone * tone_env
        
        # Light saturation
        tone = np.tanh(tone * 0.9)
        
        # Noise component - highpass for brightness without resonance
        noise_len = int(length * 0.32)
        noise = np.random.randn(noise_len)
        
        # Simple highpass
        b, a = signal.butter(1, 1300 / (self.sample_rate / 2), btype='high')
        filtered = signal.lfilter(b, a, noise)
        
        noise_env = np.exp(-np.arange(noise_len) / self.sample_rate * 14)
        filtered = filtered * noise_env
        
        # Impact transient
        impact_len = int(self.sample_rate * 0.0018)
        impact = np.random.randn(impact_len) * np.exp(-np.arange(impact_len) / self.sample_rate * 550)
        
        # Combine (35% tone, 58% noise, 7% impact)
        min_len = min(len(tone), len(filtered))
        combined = np.zeros(length, dtype=np.float32)
        combined[:min_len] = 0.35 * tone[:min_len] + 0.58 * filtered[:min_len]
        combined[:len(impact)] += 0.07 * impact
        
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
        
    def _apply_filter(self, signal_in: np.ndarray, filter_params: Filter) -> np.ndarray:
        """Apply simple lowpass filter"""
        if filter_params.cutoff >= self.sample_rate / 2:
            return signal_in
            
        # One-pole lowpass
        rc = 1.0 / (2.0 * np.pi * filter_params.cutoff)
        dt = 1.0 / self.sample_rate
        alpha = dt / (rc + dt)
        
        filtered = np.zeros_like(signal_in)
        filtered[0] = signal_in[0]
        
        for i in range(1, len(signal_in)):
            filtered[i] = filtered[i-1] + alpha * (signal_in[i] - filtered[i-1])
            
        return filtered


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

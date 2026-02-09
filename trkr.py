#!/usr/bin/env python3
"""
TRKR MIDI TRACKER

"""

import sys
import threading
import time
import random
import json
import os
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import mido
from mido import Message
from blessed import Terminal

# Import synth engine
try:
    from synth_engine import SynthEngine, VoiceType, get_voice_categories
    SYNTH_AVAILABLE = True
except ImportError:
    SYNTH_AVAILABLE = False
    print("Warning: synth_engine not available. Install numpy, sounddevice, scipy")


@dataclass
class PhraseStep:
    note: Optional[int] = None
    note_length: str = "gate"  # gate, fill, .25, .50, .75, or 1-64
    velocity: int = 100
    probability: int = 100
    condition: str = "1/1"


@dataclass
class Phrase:
    length: int = 16
    default_note_length: str = "gate"  # phrase-wide default
    steps: List[PhraseStep] = field(
        default_factory=lambda: [PhraseStep() for _ in range(16)]
    )


def midi_to_note(midi_number):
    """Convert MIDI number to note representation using flats."""
    if midi_number is None:
        return "---"
    if not 0 <= midi_number <= 127:
        raise ValueError("MIDI number must be between 0 and 127")

    note_names = [
        "C", "Db", "D", "Eb", "E", "F",
        "Gb", "G", "Ab", "A", "Bb", "B",
    ]
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"


class TRKR:
    def __init__(self, use_synth=False):
        self.term = Terminal()
        self.phrases = {i: Phrase() for i in range(128)}
        self.arrangement = [[None for _ in range(8)] for _ in range(64)]
        self.current_notes = [None] * 8
        self.active_note_timers = {}  # Track active note-off timers
        self.current_phrase_num = 0
        self.cursor_row = 0
        self.cursor_col = 0
        self.view = "arrangement"
        self.phrase_cursor = 0
        self.phrase_field = 0  # 0=note, 1=length, 2=vel, 3=prob, 4=cond
        self.phrase_page = 0
        self.phrase_header_field = 0  # 0=length, 1=default_len, 2=page, 3=offset
        self.length_options = [16, 32, 48, 64]
        self.note_length_options = ["gate", "fill", ".25", ".50", ".75"] + [str(i) for i in range(1, 65)]
        self.bar_tick = 0
        self.playing = False
        self.play_mode = "pattern"  # "pattern" or "song"
        self.current_row = 0
        self.current_steps = [0] * 8
        self.next_row = None
        self.pending_stop = False
        self.condition_counters = {}
        self.tempo = 120
        self.playback_thread = None
        self.stop_playback = False

        # Synth engine
        self.use_synth = use_synth and SYNTH_AVAILABLE
        self.synth_engine = None
        self.synth_view_mode = "channels"  # channels, voice, adsr, filter, drums, effects
        self.synth_cursor_channel = 0
        self.synth_cursor_param = 0
        self.synth_effect_param_idx = 0  # For individual effect parameters
        self.synth_voice_category = 0
        self.synth_voice_index = 0
        
        if self.use_synth:
            self.synth_engine = SynthEngine()
            self.synth_engine.start()

        # Initialize MIDI
        try:
            self.midi_out = mido.open_output()
        except Exception:
            self.midi_out = None

        self.condition_options = [
            "1/1", "1/2", "2/2", "1/3", "2/3", "3/3",
            "1/4", "2/4", "3/4", "4/4",
            "1/5", "2/5", "3/5", "4/5", "5/5",
            "1/6", "2/6", "3/6", "4/6", "5/6", "6/6",
            "1/7", "2/7", "3/7", "4/7", "5/7", "6/7", "7/7",
            "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8",
        ]

    # ── save/load functionality ────────────────────────────────

    def save_project(self, filename):
        """Save the current project to a JSON file."""
        project_data = {
            "tempo": self.tempo,
            "arrangement": self.arrangement,
            "phrases": {}
        }
        
        # Save synth settings if enabled
        if self.use_synth and self.synth_engine:
            project_data["synth"] = {
                "global_volume": self.synth_engine.global_volume,
                "channels": [],
                "effects_buses": {}
            }
            for ch in self.synth_engine.channels:
                project_data["synth"]["channels"].append({
                    "voice_type": ch.voice_type.name,
                    "volume": ch.volume,
                    "pan": ch.pan,
                    "detune": ch.detune,
                    "adsr": {
                        "attack": ch.adsr.attack,
                        "decay": ch.adsr.decay,
                        "sustain": ch.adsr.sustain,
                        "release": ch.adsr.release,
                    },
                    "filter": {
                        "cutoff": ch.filter.cutoff,
                        "resonance": ch.filter.resonance,
                        "filter_type": ch.filter.filter_type,
                    },
                    "drum_length_multiplier": ch.drum_length_multiplier,
                    "drum_release_envelope": ch.drum_release_envelope,
                    "drum_filter": {
                        "enabled": ch.drum_filter.enabled,
                        "cutoff": ch.drum_filter.cutoff,
                        "resonance": ch.drum_filter.resonance,
                        "filter_type": ch.drum_filter.filter_type,
                    },
                    "effects_sends": {
                        "chorus": ch.send_chorus,
                        "delay": ch.send_delay,
                        "reverb": ch.send_reverb,
                        "compression": ch.send_compression,
                        "crush": ch.send_crush,
                    }
                })
            
            # Save effects bus parameters
            for effect_type, bus in self.synth_engine.effects_buses.items():
                effect_data = {
                    "enabled": bus.params.enabled,
                    "wet_mix": bus.params.wet_mix
                }
                
                if effect_type.name == "CHORUS":
                    effect_data.update({
                        "rate": bus.params.rate,
                        "depth": bus.params.depth,
                        "feedback": bus.params.feedback
                    })
                elif effect_type.name == "DELAY":
                    effect_data.update({
                        "time": bus.params.time,
                        "feedback": bus.params.feedback,
                        "cross_feedback": bus.params.cross_feedback
                    })
                elif effect_type.name == "REVERB":
                    effect_data.update({
                        "room_size": bus.params.room_size,
                        "damping": bus.params.damping,
                        "width": bus.params.width
                    })
                elif effect_type.name == "COMPRESSION":
                    effect_data.update({
                        "threshold": bus.params.threshold,
                        "ratio": bus.params.ratio,
                        "attack": bus.params.attack,
                        "release": bus.params.release,
                        "makeup_gain": bus.params.makeup_gain
                    })
                elif effect_type.name == "CRUSH":
                    effect_data.update({
                        "bits": bus.params.bits,
                        "downsample": bus.params.downsample
                    })
                
                project_data["synth"]["effects_buses"][effect_type.name] = effect_data
        
        # Convert phrases to serializable format
        for phrase_num, phrase in self.phrases.items():
            project_data["phrases"][str(phrase_num)] = {
                "length": phrase.length,
                "default_note_length": phrase.default_note_length,
                "steps": [
                    {
                        "note": step.note,
                        "note_length": step.note_length,
                        "velocity": step.velocity,
                        "probability": step.probability,
                        "condition": step.condition
                    }
                    for step in phrase.steps
                ]
            }
        
        try:
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=2)
            return True
        except Exception as e:
            return False

    def load_project(self, filename):
        """Load a project from a JSON file."""
        try:
            with open(filename, 'r') as f:
                project_data = json.load(f)
            
            # Load tempo
            self.tempo = project_data.get("tempo", 120)
            
            # Load arrangement
            self.arrangement = project_data.get("arrangement", [[None for _ in range(8)] for _ in range(64)])
            
            # Load synth settings if available
            if self.use_synth and self.synth_engine and "synth" in project_data:
                synth_data = project_data["synth"]
                self.synth_engine.global_volume = synth_data.get("global_volume", 0.8)
                
                for i, ch_data in enumerate(synth_data.get("channels", [])):
                    if i < 8:
                        ch = self.synth_engine.channels[i]
                        ch.voice_type = VoiceType[ch_data.get("voice_type", "SINE")]
                        ch.volume = ch_data.get("volume", 0.8)
                        ch.pan = ch_data.get("pan", 0.5)
                        ch.detune = ch_data.get("detune", 0.0)
                        
                        adsr_data = ch_data.get("adsr", {})
                        ch.adsr.attack = adsr_data.get("attack", 0.01)
                        ch.adsr.decay = adsr_data.get("decay", 0.1)
                        ch.adsr.sustain = adsr_data.get("sustain", 0.7)
                        ch.adsr.release = adsr_data.get("release", 0.2)
                        
                        filter_data = ch_data.get("filter", {})
                        ch.filter.cutoff = filter_data.get("cutoff", 8000.0)
                        ch.filter.resonance = filter_data.get("resonance", 1.0)
                        ch.filter.filter_type = filter_data.get("filter_type", "lowpass")
                        
                        # Load drum-specific parameters
                        ch.drum_length_multiplier = ch_data.get("drum_length_multiplier", 1.0)
                        ch.drum_release_envelope = ch_data.get("drum_release_envelope", 1.0)
                        
                        drum_filter_data = ch_data.get("drum_filter", {})
                        ch.drum_filter.enabled = drum_filter_data.get("enabled", False)
                        ch.drum_filter.cutoff = drum_filter_data.get("cutoff", 8000.0)
                        ch.drum_filter.resonance = drum_filter_data.get("resonance", 1.0)
                        ch.drum_filter.filter_type = drum_filter_data.get("filter_type", "lowpass")
                        
                        # Load effects sends
                        effects_sends_data = ch_data.get("effects_sends", {})
                        ch.send_chorus = effects_sends_data.get("chorus", 0.0)
                        ch.send_delay = effects_sends_data.get("delay", 0.0)
                        ch.send_reverb = effects_sends_data.get("reverb", 0.0)
                        ch.send_compression = effects_sends_data.get("compression", 0.0)
                        ch.send_crush = effects_sends_data.get("crush", 0.0)
            
            # Load effects bus parameters
            if "effects_buses" in synth_data:
                for effect_name, effect_data in synth_data["effects_buses"].items():
                    if effect_name in [et.name for et in self.synth_engine.effects_buses.keys()]:
                        effect_type = getattr(self.synth_engine, 'EffectType.' + effect_name)
                        if effect_type in self.synth_engine.effects_buses:
                            bus = self.synth_engine.effects_buses[effect_type]
                            bus.params.enabled = effect_data.get("enabled", False)
                            bus.params.wet_mix = effect_data.get("wet_mix", 0.5)
                            
                            if effect_name == "CHORUS":
                                if hasattr(bus.params, 'rate'):
                                    bus.params.rate = effect_data.get("rate", 1.5)
                                if hasattr(bus.params, 'depth'):
                                    bus.params.depth = effect_data.get("depth", 0.02)
                                if hasattr(bus.params, 'feedback'):
                                    bus.params.feedback = effect_data.get("feedback", 0.1)
                            elif effect_name == "DELAY":
                                if hasattr(bus.params, 'time'):
                                    bus.params.time = effect_data.get("time", 0.3)
                                if hasattr(bus.params, 'feedback'):
                                    bus.params.feedback = effect_data.get("feedback", 0.4)
                                if hasattr(bus.params, 'cross_feedback'):
                                    bus.params.cross_feedback = effect_data.get("cross_feedback", 0.0)
                            elif effect_name == "REVERB":
                                if hasattr(bus.params, 'room_size'):
                                    bus.params.room_size = effect_data.get("room_size", 0.5)
                                if hasattr(bus.params, 'damping'):
                                    bus.params.damping = effect_data.get("damping", 0.5)
                                if hasattr(bus.params, 'width'):
                                    bus.params.width = effect_data.get("width", 1.0)
                            elif effect_name == "COMPRESSION":
                                if hasattr(bus.params, 'threshold'):
                                    bus.params.threshold = effect_data.get("threshold", -20.0)
                                if hasattr(bus.params, 'ratio'):
                                    bus.params.ratio = effect_data.get("ratio", 4.0)
                                if hasattr(bus.params, 'attack'):
                                    bus.params.attack = effect_data.get("attack", 0.005)
                                if hasattr(bus.params, 'release'):
                                    bus.params.release = effect_data.get("release", 0.1)
                                if hasattr(bus.params, 'makeup_gain'):
                                    bus.params.makeup_gain = effect_data.get("makeup_gain", 0.0)
                            elif effect_name == "CRUSH":
                                if hasattr(bus.params, 'bits'):
                                    bus.params.bits = effect_data.get("bits", 8)
                                if hasattr(bus.params, 'downsample'):
                                    bus.params.downsample = effect_data.get("downsample", 1)
            
            # Load phrases
            phrases_data = project_data.get("phrases", {})
            for phrase_str, phrase_data in phrases_data.items():
                phrase_num = int(phrase_str)
                phrase = Phrase(
                    length=phrase_data.get("length", 16),
                    default_note_length=phrase_data.get("default_note_length", "gate")
                )
                phrase.steps = []
                
                for step_data in phrase_data.get("steps", []):
                    step = PhraseStep(
                        note=step_data.get("note"),
                        note_length=step_data.get("note_length", "gate"),
                        velocity=step_data.get("velocity", 100),
                        probability=step_data.get("probability", 100),
                        condition=step_data.get("condition", "1/1")
                    )
                    phrase.steps.append(step)
                
                self.phrases[phrase_num] = phrase
            
            return True
        except Exception as e:
            return False

    def file_browser(self, mode="save"):
        """File browser for saving or loading projects."""
        t = self.term
        current_dir = os.getcwd()
        selected_idx = 0
        filename_input = ""
        input_mode = False
        
        while True:
            h, w = t.height, t.width
            buf = self._clear_screen()
            
            # Header
            buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
            title = f" {mode.upper()} PROJECT " if mode == "save" else " LOAD PROJECT "
            buf.append(t.move_xy(2, 1) + t.bold_cyan(title))
            buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
            
            # Current directory
            buf.append(t.move_xy(2, 4) + t.bold(f"Directory: {current_dir}"))
            buf.append(t.move_xy(0, 5) + "─" * (w - 1))
            
            # Get files and directories
            try:
                items = []
                if current_dir != "/":
                    items.append(("..", "directory"))
                
                for item in sorted(os.listdir(current_dir)):
                    item_path = os.path.join(current_dir, item)
                    if os.path.isdir(item_path):
                        items.append((item, "directory"))
                    elif item.endswith('.json'):
                        items.append((item, "file"))
                
                # Display items
                for i, (name, item_type) in enumerate(items):
                    y = 7 + i
                    if y >= h - 8:
                        break
                    
                    max_len = w - 10
                    display = name if len(name) <= max_len else name[:max_len-3] + "..."
                    
                    if i == selected_idx and not input_mode:
                        prefix = "► " if item_type == "directory" else "► "
                        buf.append(t.move_xy(4, y) + t.bold_reverse(f"{prefix}{display}"))
                    else:
                        prefix = "📁 " if item_type == "directory" else "📄 "
                        buf.append(t.move_xy(4, y) + f"{prefix}{display}")
                
                # Filename input for save mode
                if mode == "save":
                    input_y = min(h - 8, 7 + len(items))
                    buf.append(t.move_xy(0, input_y) + "─" * (w - 1))
                    buf.append(t.move_xy(2, input_y + 1) + t.bold("Filename: "))
                    if input_mode:
                        buf.append(t.move_xy(12, input_y + 1) + t.reverse(filename_input + "_"))
                    else:
                        buf.append(t.move_xy(12, input_y + 1) + filename_input)
                
            except Exception:
                buf.append(t.move_xy(2, 7) + t.red("Error reading directory"))
            
            # Footer
            footer_y = h - 5
            buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
            
            if mode == "save":
                controls = [
                    "↑/↓: Navigate | ENTER: Select/Save | TAB: Edit Filename | ESC: Cancel",
                    "TAB: Edit filename when not in input mode"
                ]
            else:
                controls = [
                    "↑/↓: Navigate | ENTER: Select | ESC: Cancel"
                ]
            
            for i, ctrl in enumerate(controls):
                buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
            
            self._flush(buf)
            
            key = t.inkey(timeout=None)
            
            if input_mode:
                if key.name == "KEY_ENTER" or key in ("\n", "\r"):
                    if filename_input.strip():
                        return os.path.join(current_dir, filename_input if filename_input.endswith('.json') else filename_input + '.json')
                elif key.name == "KEY_ESCAPE":
                    input_mode = False
                elif self._is_backspace(key):
                    filename_input = filename_input[:-1]
                elif len(str(key)) == 1 and len(filename_input) < 50:
                    filename_input += str(key)
            else:
                if key.name == "KEY_UP":
                    selected_idx = max(0, selected_idx - 1)
                elif key.name == "KEY_DOWN":
                    selected_idx = min(len(items) - 1, selected_idx + 1)
                elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                    if selected_idx < len(items):
                        name, item_type = items[selected_idx]
                        if item_type == "directory":
                            if name == "..":
                                current_dir = os.path.dirname(current_dir)
                            else:
                                current_dir = os.path.join(current_dir, name)
                            selected_idx = 0
                        elif mode == "load":
                            return os.path.join(current_dir, name)
                elif key.name == "KEY_TAB" and mode == "save":
                    input_mode = True
                elif key.name == "KEY_ESCAPE":
                    return None

    def offset_phrase(self, phrase_num, offset):
        """Offset a phrase by the specified amount (positive = forward, negative = backward)."""
        if phrase_num not in self.phrases:
            return
        
        phrase = self.phrases[phrase_num]
        if len(phrase.steps) <= 1:
            return
        
        # Normalize offset to phrase length
        offset = offset % len(phrase.steps)
        if offset == 0:
            return
        
        # Perform the cyclic shift
        steps = phrase.steps
        if offset > 0:
            # Shift forward: move steps from end to beginning
            phrase.steps = steps[-offset:] + steps[:-offset]
        else:
            # Shift backward: move steps from beginning to end
            offset = abs(offset)
            phrase.steps = steps[offset:] + steps[:offset]

    def esc_menu(self):
        """Main ESC menu with MIDI and save/load submenus."""
        t = self.term
        current_menu = "main"  # "main", "midi", "save_load"
        selected_idx = 0
        
        while True:
            h, w = t.height, t.width
            buf = self._clear_screen()
            
            # Header
            buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
            buf.append(t.move_xy(2, 1) + t.bold_cyan(" MENU "))
            buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
            
            if current_menu == "main":
                menu_items = [
                    "MIDI Settings",
                    "Save/Load Project",
                    "Back to Tracker"
                ]
                
                buf.append(t.move_xy(2, 4) + t.bold("Main Menu:"))
                buf.append(t.move_xy(0, 5) + "─" * (w - 1))
                
                for i, item in enumerate(menu_items):
                    y = 7 + i
                    if i == selected_idx:
                        buf.append(t.move_xy(4, y) + t.bold_reverse(f"► {item}"))
                    else:
                        buf.append(t.move_xy(4, y) + f"  {item}")
                
                controls = "↑/↓: Navigate | ENTER: Select | ESC: Back to Tracker"
                
            elif current_menu == "midi":
                menu_items = [
                    "Select MIDI Port",
                    "Back to Main Menu"
                ]
                
                buf.append(t.move_xy(2, 4) + t.bold("MIDI Settings:"))
                buf.append(t.move_xy(0, 5) + "─" * (w - 1))
                
                # Show current MIDI port
                current_port = self.midi_out.name if self.midi_out else "None"
                buf.append(t.move_xy(4, 7) + f"Current Port: {t.cyan(current_port)}")
                buf.append(t.move_xy(0, 8) + "─" * (w - 1))
                
                for i, item in enumerate(menu_items):
                    y = 10 + i
                    if i == selected_idx:
                        buf.append(t.move_xy(4, y) + t.bold_reverse(f"► {item}"))
                    else:
                        buf.append(t.move_xy(4, y) + f"  {item}")
                
                controls = "↑/↓: Navigate | ENTER: Select | ESC: Back to Main Menu"
                
            elif current_menu == "save_load":
                menu_items = [
                    "Save Project",
                    "Load Project",
                    "Back to Main Menu"
                ]
                
                buf.append(t.move_xy(2, 4) + t.bold("Save/Load Project:"))
                buf.append(t.move_xy(0, 5) + "─" * (w - 1))
                
                for i, item in enumerate(menu_items):
                    y = 7 + i
                    if i == selected_idx:
                        buf.append(t.move_xy(4, y) + t.bold_reverse(f"► {item}"))
                    else:
                        buf.append(t.move_xy(4, y) + f"  {item}")
                
                controls = "↑/↓: Navigate | ENTER: Select | ESC: Back to Main Menu"
            
            # Footer
            footer_y = h - 3
            buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
            buf.append(t.move_xy(2, footer_y + 1) + t.magenta(controls))
            
            self._flush(buf)
            
            key = t.inkey(timeout=None)
            
            if key.name == "KEY_UP":
                selected_idx = max(0, selected_idx - 1)
            elif key.name == "KEY_DOWN":
                if current_menu == "main":
                    selected_idx = min(2, selected_idx + 1)
                elif current_menu == "midi":
                    selected_idx = min(1, selected_idx + 1)
                elif current_menu == "save_load":
                    selected_idx = min(2, selected_idx + 1)
            elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                if current_menu == "main":
                    if selected_idx == 0:  # MIDI Settings
                        current_menu = "midi"
                        selected_idx = 0
                    elif selected_idx == 1:  # Save/Load
                        current_menu = "save_load"
                        selected_idx = 0
                    elif selected_idx == 2:  # Back to Tracker
                        return
                elif current_menu == "midi":
                    if selected_idx == 0:  # Select MIDI Port
                        selected_port = self.select_midi_port()
                        if selected_port:
                            self.change_midi_port(selected_port)
                    elif selected_idx == 1:  # Back to Main
                        current_menu = "main"
                        selected_idx = 0
                elif current_menu == "save_load":
                    if selected_idx == 0:  # Save Project
                        filename = self.file_browser("save")
                        if filename:
                            if self.save_project(filename):
                                # Show success message briefly
                                buf = self._clear_screen()
                                buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
                                buf.append(t.move_xy(2, 1) + t.bold_green("SUCCESS"))
                                buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
                                buf.append(t.move_xy(2, 5) + t.green(f"Project saved to: {filename}"))
                                buf.append(t.move_xy(2, 7) + "Press any key to continue...")
                                self._flush(buf)
                                t.inkey(timeout=None)
                    elif selected_idx == 1:  # Load Project
                        filename = self.file_browser("load")
                        if filename:
                            if self.load_project(filename):
                                # Show success message briefly
                                buf = self._clear_screen()
                                buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
                                buf.append(t.move_xy(2, 1) + t.bold_green("SUCCESS"))
                                buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
                                buf.append(t.move_xy(2, 5) + t.green(f"Project loaded from: {filename}"))
                                buf.append(t.move_xy(2, 7) + "Press any key to continue...")
                                self._flush(buf)
                                t.inkey(timeout=None)
                    elif selected_idx == 2:  # Back to Main
                        current_menu = "main"
                        selected_idx = 1
            elif key.name == "KEY_ESCAPE":
                if current_menu == "main":
                    return
                else:
                    current_menu = "main"
                    selected_idx = 0

    # ── helpers ───────────────────────────────────────────────

    def _flush(self, buf):
        """Write the entire frame buffer to stdout in one call."""
        sys.stdout.write("".join(buf))
        sys.stdout.flush()

    def _clear_screen(self):
        """Return a buffer that homes the cursor then overwrites every
        screen position with spaces.  Because this is part of the same
        write as the content that follows, the terminal never shows a
        blank frame."""
        t = self.term
        w = t.width
        buf = [t.home]
        blank = " " * w
        for y in range(t.height):
            buf.append(t.move_xy(0, y) + blank)
        return buf

    @staticmethod
    def _is_backspace(key):
        return (
            key.name == "KEY_BACKSPACE"
            or key.name == "KEY_DELETE"
            or key in ("\x7f", "\x08")
        )

    def _is_shift_left(self, key):
        """Detect Shift+Left across terminals, with [ as fallback."""
        if key.name == "KEY_SLEFT":
            return True
        if key.code is not None and key.code == getattr(
            self.term, "KEY_SLEFT", -1
        ):
            return True
        if str(key) == "\x1b[1;2D":  # xterm raw sequence
            return True
        if str(key) == "[":
            return True
        return False

    def _is_shift_right(self, key):
        """Detect Shift+Right across terminals, with ] as fallback."""
        if key.name == "KEY_SRIGHT":
            return True
        if key.code is not None and key.code == getattr(
            self.term, "KEY_SRIGHT", -1
        ):
            return True
        if str(key) == "\x1b[1;2C":  # xterm raw sequence
            return True
        if str(key) == "]":
            return True
        return False

    def _is_shift_up(self, key):
        """Detect Shift+Up across terminals, with + as fallback."""
        if key.name == "KEY_SR":  # Shift+Up in many terminals
            return True
        if key.code is not None and key.code == getattr(
            self.term, "KEY_SR", -1
        ):
            return True
        if str(key) == "\x1b[1;2A":  # xterm raw sequence
            return True
        if str(key) == "+":  # fallback key
            return True
        return False

    def _is_shift_down(self, key):
        """Detect Shift+Down across terminals, with - as fallback."""
        if key.name == "KEY_SF":  # Shift+Down in many terminals
            return True
        if key.code is not None and key.code == getattr(
            self.term, "KEY_SF", -1
        ):
            return True
        if str(key) == "\x1b[1;2B":  # xterm raw sequence
            return True
        if str(key) == "-":  # fallback key
            return True
        return False

    
    def _set_phrase_length(self, phrase, new_length):
        """Extend or shrink a phrase. New pages are copies of page 1."""
        old_length = phrase.length
        if new_length == old_length:
            return
        if new_length > old_length:
            first_page = phrase.steps[:16]
            while len(phrase.steps) < new_length:
                for src in first_page:
                    phrase.steps.append(PhraseStep(
                        note=src.note,
                        note_length=src.note_length,
                        velocity=src.velocity,
                        probability=src.probability,
                        condition=src.condition,
                    ))
        else:
            phrase.steps = phrase.steps[:new_length]
        phrase.length = new_length

    def _get_max_phrase_length(self, row):
        """Return the longest phrase length assigned to a row."""
        max_len = 16
        for ch in range(8):
            phrase_num = self.arrangement[row][ch]
            if phrase_num is not None:
                max_len = max(max_len, self.phrases[phrase_num].length)
        return max_len

    # ── MIDI / playback ──────────────────────────────────────

    def get_current_note(self, channel):
        note_num = self.current_notes[channel]
        if note_num is not None:
            return midi_to_note(note_num)
        return None

    def should_trigger(self, step, step_key):
        if random.random() * 100 > step.probability:
            return False
        if step.condition == "1/1":
            return True
        num, denom = map(int, step.condition.split("/"))
        key = f"{step_key}_{step.condition}"
        count = self.condition_counters.get(key, 0) + 1
        self.condition_counters[key] = count % denom
        return count % denom == num - 1

    def calculate_note_length(self, note_length, phrase, step_idx):
        """Calculate note duration in seconds based on note_length parameter."""
        step_time = 60 / self.tempo / 4  # Duration of one step
        
        if note_length == "gate":
            return 0.05
        elif note_length == "fill":
            # Will be handled specially - return None
            return None
        elif note_length == ".25":
            return step_time * 0.25
        elif note_length == ".50":
            return step_time * 0.50
        elif note_length == ".75":
            return step_time * 0.75
        else:
            # Integer step counts
            try:
                steps = int(note_length)
                return step_time * steps
            except ValueError:
                return 0.05  # Default to gate on error

    def send_note_off(self, channel, note):
        """Send note off to both synth and MIDI"""
        if self.use_synth and self.synth_engine:
            self.synth_engine.note_off(channel, note)
        
        if self.midi_out:
            try:
                self.midi_out.send(
                    Message("note_off", channel=channel, note=note)
                )
            except Exception:
                pass

    def send_midi(self, channel, note, velocity, note_length, phrase, step_idx):
        """Send MIDI note with proper note length handling"""
        # Cancel any existing timer for this channel
        timer_key = f"{channel}_{note}"
        if timer_key in self.active_note_timers:
            self.active_note_timers[timer_key].cancel()
            del self.active_note_timers[timer_key]
        
        # Send note on
        if self.use_synth and self.synth_engine:
            self.synth_engine.note_on(channel, note, velocity)
        
        if self.midi_out:
            try:
                self.midi_out.send(
                    Message("note_on", channel=channel,
                            note=note, velocity=velocity)
                )
            except Exception:
                pass
        
        # Handle note off based on length
        duration = self.calculate_note_length(note_length, phrase, step_idx)
        
        if duration is None:  # "fill" mode
            # Note will be turned off when next note plays or at end of phrase
            # Store the note so we can turn it off later
            pass
        else:
            # Schedule note off
            timer = threading.Timer(
                duration,
                lambda: self.send_note_off(channel, note)
            )
            timer.start()
            self.active_note_timers[timer_key] = timer

    def playback_loop(self):
        step_time = 60 / self.tempo / 4
        last_step_time = time.time()
        self.bar_tick = 0
        max_length = self._get_max_phrase_length(self.current_row)

        while not self.stop_playback:
            current_time = time.time()

            if current_time - last_step_time >= step_time:
                last_step_time = current_time

                for channel in range(8):
                    phrase_num = self.arrangement[self.current_row][channel]

                    if phrase_num is not None:
                        phrase = self.phrases[phrase_num]
                        step_idx = self.current_steps[channel]

                        # Safety clamp if phrase was resized during playback
                        if step_idx >= phrase.length:
                            step_idx = step_idx % phrase.length
                            self.current_steps[channel] = step_idx

                        step = phrase.steps[step_idx]

                        if step.note is not None:
                            step_key = (
                                f"{self.current_row}_{channel}_{step_idx}"
                            )
                            if self.should_trigger(step, step_key):
                                # Turn off previous note if in "fill" mode
                                prev_note = self.current_notes[channel]
                                if prev_note is not None:
                                    timer_key = f"{channel}_{prev_note}"
                                    if timer_key in self.active_note_timers:
                                        self.active_note_timers[timer_key].cancel()
                                        del self.active_note_timers[timer_key]
                                    self.send_note_off(channel, prev_note)
                                
                                self.current_notes[channel] = step.note
                                
                                # Determine note length (use step length or phrase default)
                                effective_length = step.note_length
                                if effective_length == "gate" and phrase.default_note_length != "gate":
                                    effective_length = phrase.default_note_length
                                
                                self.send_midi(
                                    channel, step.note, step.velocity,
                                    effective_length, phrase, step_idx
                                )

                        # Wrap at this phrase's own length (short phrases loop)
                        self.current_steps[channel] = (
                            (step_idx + 1) % phrase.length
                        )

                self.bar_tick += 1

                # Bar boundary reached when the longest phrase completes
                if self.bar_tick >= max_length:
                    self.bar_tick = 0
                    self.current_steps = [0] * 8

                    if self.pending_stop and self.play_mode == "pattern":
                        self.playing = False
                        self.pending_stop = False
                        break

                    if self.next_row is not None:
                        self.current_row = self.next_row
                        self.next_row = None
                        max_length = self._get_max_phrase_length(
                            self.current_row
                        )
                    elif self.play_mode == "song":
                        next_row = self.current_row + 1
                        if next_row >= 64 or all(
                            p is None
                            for p in self.arrangement[next_row]
                        ):
                            self.current_row = 0
                        else:
                            self.current_row = next_row
                        max_length = self._get_max_phrase_length(
                            self.current_row
                        )

            time.sleep(0.001)

        self.playing = False

    def start_playback(self, row):
        if self.playing:
            self.next_row = row
        else:
            self.current_row = row
            self.current_steps = [0] * 8
            self.bar_tick = 0
            self.playing = True
            self.pending_stop = False
            self.stop_playback = False
            self.playback_thread = threading.Thread(
                target=self.playback_loop, daemon=True
            )
            self.playback_thread.start()

    def stop_playback_func(self):
        if self.play_mode == "pattern":
            self.pending_stop = True
        else:
            self.playing = False
            self.stop_playback = True
            if self.playback_thread:
                self.playback_thread.join(timeout=1.0)

    def toggle_play_mode(self):
        was_playing = self.playing
        if was_playing:
            self.stop_playback_func()
            if self.playback_thread:
                self.playback_thread.join(timeout=1.0)
        self.play_mode = "song" if self.play_mode == "pattern" else "pattern"
        self.playing = False
        self.pending_stop = False
        self.next_row = None

    # ── drawing ──────────────────────────────────────────────

    def draw_arrangement(self):
        t = self.term
        h, w = t.height, t.width
        buf = self._clear_screen()

        # ── header ──
        buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))

        title = " MIDI PHRASE TRACKER - ARRANGEMENT "
        buf.append(t.move_xy(2, 1) + t.bold_cyan(title))

        play_status = (
            "PLAYING"
            if self.playing
            else ("STOPPING..." if self.pending_stop else "STOPPED")
        )
        
        synth_status = " | SYNTH:ON" if self.use_synth else ""
        
        status = (
            f"PHRASE:{self.current_phrase_num:03d} | "
            f"MODE:{self.play_mode.upper()} | "
            f"{play_status} | TEMPO:{self.tempo} | "
            f"ROW:{self.current_row:02d}{synth_status}"
        )
        buf.append(
            t.move_xy(max(0, w - len(status) - 2), 1) + t.yellow(status)
        )

        buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))

        # ── column headers ──
        headers = (
            "ROW │ CH1  CH2  CH3  CH4  "
            "CH5  CH6  CH7  CH8 │ CURRENT NOTES"
        )
        buf.append(t.move_xy(2, 3) + t.bold(headers))
        buf.append(t.move_xy(0, 4) + "─" * (w - 1))

        # ── arrangement grid ──
        start_row = max(0, self.cursor_row - 10)
        notes_x = 9 + (8 * 5) + 2  # column for current-notes display

        for i in range(start_row, min(64, start_row + h - 10)):
            y = 5 + (i - start_row)
            if y >= h - 5:
                break

            # row number
            row_text = f"{i:02d}  │ "
            if i == self.current_row and self.playing:
                buf.append(t.move_xy(2, y) + t.bold_green(row_text))
            else:
                buf.append(t.move_xy(2, y) + t.green(row_text))

            # channels
            for ch in range(8):
                x = 9 + (ch * 5)
                phrase_num = self.arrangement[i][ch]
                text = (
                    f"{phrase_num:03d}" if phrase_num is not None else "---"
                )

                if i == self.cursor_row and ch == self.cursor_col:
                    buf.append(t.move_xy(x, y) + t.reverse(text))
                elif (
                    i == self.current_row
                    and self.playing
                    and phrase_num is not None
                ):
                    buf.append(t.move_xy(x, y) + t.black_on_green(text))
                else:
                    buf.append(t.move_xy(x, y) + text)

            # vertical separator for notes column
            buf.append(t.move_xy(notes_x - 2, y) + "│")

        # ── current notes (fixed position, right of grid) ──
        for pair in range(4):
            ch1, ch2 = pair * 2, pair * 2 + 1
            n1 = f"{midi_to_note(self.current_notes[ch1]):<4}"
            n2 = f"{midi_to_note(self.current_notes[ch2]):<4}"
            buf.append(t.move_xy(notes_x, 5 + pair) + f"{n1}| {n2}")

        # ── footer ──
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))

        if self.play_mode == "pattern":
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase "
                "| SHIFT+←→ or []:Change Phrase# | BKSP:Remove",
                "SPACE:Play Row | TAB:Toggle Mode | .:Stop | T:Tempo | Q:Quit"
            ]
        else:
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase "
                "| SHIFT+←→ or []:Change Phrase# | BKSP:Remove",
                "SPACE:Play/Stop Song | TAB:Toggle Mode "
                "| T:Tempo | Q:Quit"
            ]
        
        if self.use_synth:
            controls.append("S:Synth Engine | ESC:Menu")
        else:
            controls.append("ESC:Menu")

        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, h - 4 + i) + t.magenta(ctrl))

        self._flush(buf)

    def draw_phrase(self):
        t = self.term
        h, w = t.height, t.width
        phrase = self.phrases[self.current_phrase_num]
        max_pages = phrase.length // 16
        page_start = self.phrase_page * 16
        buf = self._clear_screen()

        # ── header ──
        buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))

        title = f" PHRASE {self.current_phrase_num:03d} EDITOR "
        buf.append(t.move_xy(2, 1) + t.bold_cyan(title))

        # Length selector
        len_x = 24
        length_label = "LENGTH:"
        length_val = f"{phrase.length:2d}"
        if self.phrase_cursor == -1 and self.phrase_header_field == 0:
            buf.append(
                t.move_xy(len_x, 1)
                + t.bold(length_label)
                + t.reverse(length_val)
            )
        else:
            buf.append(
                t.move_xy(len_x, 1)
                + t.bold(length_label)
                + length_val
            )

        # Default note length selector
        deflen_x = 36
        deflen_label = "DEF.LEN:"
        deflen_val = f"{phrase.default_note_length:>4s}"
        if self.phrase_cursor == -1 and self.phrase_header_field == 1:
            buf.append(
                t.move_xy(deflen_x, 1)
                + t.bold(deflen_label)
                + t.reverse(deflen_val)
            )
        else:
            buf.append(
                t.move_xy(deflen_x, 1)
                + t.bold(deflen_label)
                + deflen_val
            )

        # Page selector
        page_x = 51
        page_label = "PAGE:"
        page_val = f"{self.phrase_page + 1}/{max_pages}"
        if self.phrase_cursor == -1 and self.phrase_header_field == 2:
            buf.append(
                t.move_xy(page_x, 1)
                + t.bold(page_label)
                + t.reverse(page_val)
            )
        else:
            buf.append(
                t.move_xy(page_x, 1)
                + t.bold(page_label)
                + page_val
            )

        # Offset selector
        offset_x = 63
        offset_label = "OFFSET:"
        offset_val = "  0"
        if self.phrase_cursor == -1 and self.phrase_header_field == 3:
            buf.append(
                t.move_xy(offset_x, 1)
                + t.bold(offset_label)
                + t.reverse(offset_val)
            )
        else:
            buf.append(
                t.move_xy(offset_x, 1)
                + t.bold(offset_label)
                + offset_val
            )

        buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))

        # Column headers
        headers = "STEP │ NOTE   LEN   VEL  PROB%  COND"
        buf.append(t.move_xy(2, 3) + t.bold(headers))
        buf.append(t.move_xy(0, 4) + "─" * (w - 1))

        # ── steps for current page ──
        for i in range(16):
            y = 5 + i
            step_idx = page_start + i

            if step_idx >= phrase.length:
                break

            step = phrase.steps[step_idx]

            # Check if this step is currently playing
            playing_here = False
            if self.playing:
                for ch in range(8):
                    if (
                        self.arrangement[self.current_row][ch]
                        == self.current_phrase_num
                        and self.current_steps[ch] == step_idx
                    ):
                        playing_here = True
                        break

            row_fmt = t.black_on_green if playing_here else str

            # Step number (actual index across all pages)
            step_label = f" {step_idx:02d}  │ "
            if playing_here:
                buf.append(
                    t.move_xy(2, y)
                    + t.bold(t.black_on_green(step_label))
                )
            else:
                buf.append(t.move_xy(2, y) + t.bold(step_label))

            # Note
            note_raw = (
                midi_to_note(step.note)
                if step.note is not None
                else "---"
            )
            note_text = f"{note_raw:<4}"
            if i == self.phrase_cursor and self.phrase_field == 0:
                buf.append(t.move_xy(10, y) + t.reverse(note_text))
            else:
                buf.append(t.move_xy(10, y) + row_fmt(note_text))

            # Note Length
            len_text = f"{step.note_length:>4s}"
            if i == self.phrase_cursor and self.phrase_field == 1:
                buf.append(t.move_xy(16, y) + t.reverse(len_text))
            else:
                buf.append(t.move_xy(16, y) + row_fmt(len_text))

            # Velocity
            vel_text = f"{step.velocity:3d}"
            if i == self.phrase_cursor and self.phrase_field == 2:
                buf.append(t.move_xy(22, y) + t.reverse(vel_text))
            else:
                buf.append(t.move_xy(22, y) + row_fmt(vel_text))

            # Probability
            prob_text = f"{step.probability:3d}"
            if i == self.phrase_cursor and self.phrase_field == 3:
                buf.append(t.move_xy(28, y) + t.reverse(prob_text))
            else:
                buf.append(t.move_xy(28, y) + row_fmt(prob_text))

            # Condition
            cond_text = f"{step.condition:>4s}"
            if i == self.phrase_cursor and self.phrase_field == 4:
                buf.append(t.move_xy(36, y) + t.reverse(cond_text))
            else:
                buf.append(t.move_xy(36, y) + row_fmt(cond_text))

        # ── footer ──
        buf.append(t.move_xy(0, h - 4) + "─" * (w - 1))
        controls = [
            "↑↓:Navigate Steps | ←→:Navigate Fields "
            "| SHIFT+←→:Adjust Value | SHIFT+↑↓:Octave",
            "BACKSPACE:Clear Note | ESC:Back to Arrangement",
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, h - 3 + i) + t.magenta(ctrl))

        self._flush(buf)

    # ── Synth Engine View ────────────────────────────────────
    
    def draw_synth_engine(self):
        """Draw the synth engine interface"""
        t = self.term
        h, w = t.height, t.width
        buf = self._clear_screen()
        
        # Header
        buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
        title = " SYNTH ENGINE "
        buf.append(t.move_xy(2, 1) + t.bold_cyan(title))
        
        mode_text = f"MODE: {self.synth_view_mode.upper()}"
        buf.append(t.move_xy(w - len(mode_text) - 3, 1) + t.yellow(mode_text))
        buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
        
        if self.synth_view_mode == "channels":
            self._draw_synth_channels(buf, t, h, w)
        elif self.synth_view_mode == "voice":
            self._draw_synth_voice_select(buf, t, h, w)
        elif self.synth_view_mode == "adsr":
            self._draw_synth_adsr(buf, t, h, w)
        elif self.synth_view_mode == "filter":
            self._draw_synth_filter(buf, t, h, w)
        elif self.synth_view_mode == "drums":
            self._draw_synth_drums(buf, t, h, w)
        elif self.synth_view_mode == "effects":
            self._draw_synth_effects(buf, t, h, w)
        
        self._flush(buf)
    
    def _draw_synth_channels(self, buf, t, h, w):
        """Draw channel overview"""
        buf.append(t.move_xy(2, 4) + t.bold("CH │ VOICE              VOL   PAN   DETUNE"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        for ch in range(8):
            y = 6 + ch
            voice = self.synth_engine.channels[ch]
            
            ch_text = f" {ch+1} │"
            voice_text = f"{voice.voice_type.value:<18}"
            vol_text = f"{int(voice.volume * 100):3d}%"
            pan_text = f"{int(voice.pan * 100):3d}%"
            detune_text = f"{int(voice.detune):+4d}c"
            
            if ch == self.synth_cursor_channel:
                buf.append(t.move_xy(2, y) + t.bold_green(ch_text))
                buf.append(t.move_xy(6, y) + t.reverse(voice_text))
                buf.append(t.move_xy(26, y) + t.reverse(vol_text))
                buf.append(t.move_xy(32, y) + t.reverse(pan_text))
                buf.append(t.move_xy(38, y) + t.reverse(detune_text))
            else:
                buf.append(t.move_xy(2, y) + t.bold(ch_text))
                buf.append(t.move_xy(6, y) + voice_text)
                buf.append(t.move_xy(26, y) + vol_text)
                buf.append(t.move_xy(32, y) + pan_text)
                buf.append(t.move_xy(38, y) + detune_text)
        
        # Global volume
        buf.append(t.move_xy(0, 15) + "─" * (w - 1))
        global_vol_text = f"GLOBAL VOLUME: {int(self.synth_engine.global_volume * 100):3d}%"
        buf.append(t.move_xy(2, 16) + t.bold(global_vol_text))
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Select Channel | V: Voice Select | A: ADSR | F: Filter | D: Drums | E: Effects",
            "SHIFT+←/→: Adjust Volume | [/]: Adjust Pan | +/-: Adjust Detune",
            "G+SHIFT+←/→: Global Volume | ESC: Back to Arrangement"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def _draw_synth_voice_select(self, buf, t, h, w):
        """Draw voice selection interface"""
        categories = get_voice_categories()
        cat_names = list(categories.keys())
        current_cat = cat_names[self.synth_voice_category]
        voices = categories[current_cat]
        
        buf.append(t.move_xy(2, 4) + t.bold(f"Channel {self.synth_cursor_channel + 1} - Voice Selection"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        # Category tabs
        tab_y = 6
        tab_x = 2
        for i, cat in enumerate(cat_names):
            if i == self.synth_voice_category:
                buf.append(t.move_xy(tab_x, tab_y) + t.reverse(f" {cat} "))
            else:
                buf.append(t.move_xy(tab_x, tab_y) + f" {cat} ")
            tab_x += len(cat) + 3
        
        buf.append(t.move_xy(0, 7) + "─" * (w - 1))
        
        # Voice list
        for i, voice in enumerate(voices):
            y = 9 + i
            if y >= h - 6:
                break
                
            if i == self.synth_voice_index:
                buf.append(t.move_xy(4, y) + t.bold_reverse(f"► {voice.value}"))
            else:
                buf.append(t.move_xy(4, y) + f"  {voice.value}")
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Navigate Voices | ←/→: Change Category | ENTER: Select Voice",
            "ESC: Back to Channels"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def _draw_synth_adsr(self, buf, t, h, w):
        """Draw ADSR envelope editor"""
        voice = self.synth_engine.channels[self.synth_cursor_channel]
        
        buf.append(t.move_xy(2, 4) + t.bold(f"Channel {self.synth_cursor_channel + 1} - ADSR Envelope"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        params = [
            ("Attack", voice.adsr.attack, "s"),
            ("Decay", voice.adsr.decay, "s"),
            ("Sustain", voice.adsr.sustain, "level"),
            ("Release", voice.adsr.release, "s"),
        ]
        
        for i, (name, value, unit) in enumerate(params):
            y = 7 + i * 2
            
            if unit == "s":
                display = f"{value:.3f}s"
            else:
                display = f"{value:.2f}"
            
            param_text = f"{name:10s}: {display:>8s}"
            
            if i == self.synth_cursor_param:
                buf.append(t.move_xy(4, y) + t.bold_reverse(param_text))
                # Draw bar
                bar_width = min(40, int(value * 100) if unit == "level" else int(value * 20))
                buf.append(t.move_xy(25, y) + t.green("█" * bar_width))
            else:
                buf.append(t.move_xy(4, y) + param_text)
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Navigate Parameters | SHIFT+←/→: Adjust Value",
            "ESC: Back to Channels"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def _draw_synth_filter(self, buf, t, h, w):
        """Draw filter editor"""
        voice = self.synth_engine.channels[self.synth_cursor_channel]
        
        buf.append(t.move_xy(2, 4) + t.bold(f"Channel {self.synth_cursor_channel + 1} - Filter"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        params = [
            ("Cutoff", f"{voice.filter.cutoff:.0f} Hz"),
            ("Resonance", f"{voice.filter.resonance:.2f}"),
            ("Type", voice.filter.filter_type),
        ]
        
        for i, (name, value) in enumerate(params):
            y = 7 + i * 2
            param_text = f"{name:12s}: {value:>12s}"
            
            if i == self.synth_cursor_param:
                buf.append(t.move_xy(4, y) + t.bold_reverse(param_text))
            else:
                buf.append(t.move_xy(4, y) + param_text)
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Navigate Parameters | SHIFT+←/→: Adjust Value",
            "ESC: Back to Channels"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def _draw_synth_drums(self, buf, t, h, w):
        """Draw drum-specific controls editor"""
        voice = self.synth_engine.channels[self.synth_cursor_channel]
        
        buf.append(t.move_xy(2, 4) + t.bold(f"Channel {self.synth_cursor_channel + 1} - Drum Controls"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        # Check if current voice is a drum
        is_drum = self.synth_engine._is_drum_voice(voice.voice_type)
        
        if not is_drum:
            buf.append(t.move_xy(4, 7) + t.yellow("Drum controls only available for drum voices"))
            buf.append(t.move_xy(4, 9) + t.dim(f"Current voice: {voice.voice_type.value}"))
        else:
            params = [
                ("Length Mult", f"{voice.drum_length_multiplier:.2f}", "0.1-2.0"),
                ("Release Env", f"{voice.drum_release_envelope:.2f}", "0.0-2.0"),
                ("Drum Filter", "ON" if voice.drum_filter.enabled else "OFF", "toggle"),
                ("Filter Cutoff", f"{voice.drum_filter.cutoff:.0f} Hz", "20-20000"),
                ("Filter Res", f"{voice.drum_filter.resonance:.2f}", "0.1-10.0"),
                ("Filter Type", voice.drum_filter.filter_type, "low/high/band"),
            ]
            
            for i, (name, value, range_info) in enumerate(params):
                y = 7 + i * 2
                
                if i == 2:  # Filter enabled toggle
                    param_text = f"{name:12s}: {value:>3s} ({range_info})"
                else:
                    param_text = f"{name:12s}: {value:>8s} ({range_info})"
                
                if i == self.synth_cursor_param:
                    buf.append(t.move_xy(4, y) + t.bold_reverse(param_text))
                    # Draw bar for numeric parameters
                    if i in [0, 1, 3, 4]:  # Numeric parameters
                        if i == 0:  # Length multiplier
                            bar_width = min(40, int((voice.drum_length_multiplier - 0.1) / 1.9 * 40))
                        elif i == 1:  # Release envelope
                            bar_width = min(40, int(voice.drum_release_envelope / 2.0 * 40))
                        elif i == 3:  # Filter cutoff (log scale approximation)
                            normalized = (np.log10(max(20, voice.drum_filter.cutoff)) - np.log10(20)) / (np.log10(20000) - np.log10(20))
                            bar_width = min(40, int(normalized * 40))
                        elif i == 4:  # Filter resonance
                            bar_width = min(40, int((voice.drum_filter.resonance - 0.1) / 9.9 * 40))
                        buf.append(t.move_xy(45, y) + t.green("█" * bar_width))
                else:
                    buf.append(t.move_xy(4, y) + param_text)
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Navigate Parameters | SHIFT+←/→: Adjust Value | ENTER: Toggle Filter",
            "ESC: Back to Channels"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def _draw_synth_effects(self, buf, t, h, w):
        """Draw effects sends and parameters editor"""
        voice = self.synth_engine.channels[self.synth_cursor_channel]
        
        buf.append(t.move_xy(2, 4) + t.bold(f"Channel {self.synth_cursor_channel + 1} - Effects Sends"))
        buf.append(t.move_xy(0, 5) + "─" * (w - 1))
        
        # Effect sends
        sends = [
            ("Chorus", voice.send_chorus, "0.0-1.0"),
            ("Delay", voice.send_delay, "0.0-1.0"),
            ("Reverb", voice.send_reverb, "0.0-1.0"),
            ("Compression", voice.send_compression, "0.0-1.0"),
            ("Crush", voice.send_crush, "0.0-1.0"),
        ]
        
        for i, (name, value, range_info) in enumerate(sends):
            y = 7 + i * 2
            param_text = f"{name:12s}: {value:.2f} ({range_info})"
            
            if i == self.synth_cursor_param:
                buf.append(t.move_xy(4, y) + t.bold_reverse(param_text))
                # Draw bar
                bar_width = min(40, int(value * 40))
                buf.append(t.move_xy(45, y) + t.green("█" * bar_width))
            else:
                buf.append(t.move_xy(4, y) + param_text)
        
        # Effect parameters section
        buf.append(t.move_xy(0, 18) + "─" * (w - 1))
        buf.append(t.move_xy(2, 19) + t.bold("Effect Parameters (use ↑/↓ to select effect)"))
        
        # Show parameters for selected effect
        effect_names = ["Chorus", "Delay", "Reverb", "Compression", "Crush"]
        from synth_engine import EffectType
        effect_types = [EffectType.CHORUS, EffectType.DELAY, EffectType.REVERB, EffectType.COMPRESSION, EffectType.CRUSH]
        
        # Determine which effect to show based on cursor position
        if self.synth_cursor_param >= 5:  # Effect parameter editing
            effect_idx = self.synth_cursor_param - 5
            if effect_idx < len(effect_types):
                selected_effect_type = effect_types[effect_idx]
                bus = self.synth_engine.effects_buses[selected_effect_type]
                params = bus.params
                
                buf.append(t.move_xy(4, 21) + t.bold(f"{effect_names[effect_idx]} Effect:"))
                
                # Draw effect-specific parameters with cursor
                if selected_effect_type == EffectType.CHORUS:
                    param_names = ["Enabled", "Rate", "Depth", "Feedback", "Wet Mix"]
                    param_values = [
                        f"{'ON' if params.enabled else 'OFF'}",
                        f"{params.rate:.1f} Hz",
                        f"{params.depth:.3f}s",
                        f"{params.feedback:.2f}",
                        f"{params.wet_mix:.2f}"
                    ]
                elif selected_effect_type == EffectType.DELAY:
                    param_names = ["Enabled", "Time", "Feedback", "Cross Feedback", "Wet Mix"]
                    param_values = [
                        f"{'ON' if params.enabled else 'OFF'}",
                        f"{params.time:.2f}s",
                        f"{params.feedback:.2f}",
                        f"{params.cross_feedback:.2f}",
                        f"{params.wet_mix:.2f}"
                    ]
                elif selected_effect_type == EffectType.REVERB:
                    param_names = ["Enabled", "Room Size", "Damping", "Width", "Wet Mix"]
                    param_values = [
                        f"{'ON' if params.enabled else 'OFF'}",
                        f"{params.room_size:.2f}",
                        f"{params.damping:.2f}",
                        f"{params.width:.2f}",
                        f"{params.wet_mix:.2f}"
                    ]
                elif selected_effect_type == EffectType.COMPRESSION:
                    param_names = ["Enabled", "Threshold", "Ratio", "Attack", "Release", "Makeup", "Wet Mix"]
                    param_values = [
                        f"{'ON' if params.enabled else 'OFF'}",
                        f"{params.threshold:.1f} dB",
                        f"{params.ratio:.1f}:1",
                        f"{params.attack:.3f}s",
                        f"{params.release:.3f}s",
                        f"{params.makeup_gain:.1f} dB",
                        f"{params.wet_mix:.2f}"
                    ]
                elif selected_effect_type == EffectType.CRUSH:
                    param_names = ["Enabled", "Bits", "Downsample", "Wet Mix"]
                    param_values = [
                        f"{'ON' if params.enabled else 'OFF'}",
                        f"{params.bits}",
                        f"{params.downsample}",
                        f"{params.wet_mix:.2f}"
                    ]
                else:
                    param_names = []
                    param_values = []
                
                # Draw parameters with cursor
                for i, (name, value) in enumerate(zip(param_names, param_values)):
                    cursor = "▶ " if i == self.synth_effect_param_idx else "  "
                    buf.append(t.move_xy(6, 22 + i) + cursor + f"{name}: {value}")
        
        # Footer
        footer_y = h - 5
        buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))
        controls = [
            "↑/↓: Navigate Sends | SHIFT+←/→: Adjust Send Amount | TAB: Edit Effect",
            "↑/↓: Navigate Parameters | SHIFT+←/→: Adjust Parameter | ENTER: Toggle Effect | ESC: Back"
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, footer_y + 1 + i) + t.magenta(ctrl))
    
    def handle_synth_input(self, key):
        """Handle synth engine input"""
        if self.synth_view_mode == "channels":
            if key.name == "KEY_UP":
                self.synth_cursor_channel = max(0, self.synth_cursor_channel - 1)
            elif key.name == "KEY_DOWN":
                self.synth_cursor_channel = min(7, self.synth_cursor_channel + 1)
            elif key in ("v", "V"):
                self.synth_view_mode = "voice"
                self.synth_voice_index = 0
            elif key in ("a", "A"):
                self.synth_view_mode = "adsr"
                self.synth_cursor_param = 0
            elif key in ("f", "F"):
                self.synth_view_mode = "filter"
                self.synth_cursor_param = 0
            elif key in ("d", "D"):
                self.synth_view_mode = "drums"
                self.synth_cursor_param = 0
            elif key in ("e", "E"):
                self.synth_view_mode = "effects"
                self.synth_cursor_param = 0
            elif self._is_shift_left(key):
                if key in ("g", "G"):  # Global volume
                    self.synth_engine.global_volume = max(0.0, self.synth_engine.global_volume - 0.05)
                else:
                    voice = self.synth_engine.channels[self.synth_cursor_channel]
                    voice.volume = max(0.0, voice.volume - 0.05)
            elif self._is_shift_right(key):
                if key in ("g", "G"):  # Global volume
                    self.synth_engine.global_volume = min(1.0, self.synth_engine.global_volume + 0.05)
                else:
                    voice = self.synth_engine.channels[self.synth_cursor_channel]
                    voice.volume = min(1.0, voice.volume + 0.05)
            elif str(key) == "[":
                voice = self.synth_engine.channels[self.synth_cursor_channel]
                voice.pan = max(0.0, voice.pan - 0.1)
            elif str(key) == "]":
                voice = self.synth_engine.channels[self.synth_cursor_channel]
                voice.pan = min(1.0, voice.pan + 0.1)
            elif str(key) == "+":
                voice = self.synth_engine.channels[self.synth_cursor_channel]
                voice.detune = min(100, voice.detune + 10)
            elif str(key) == "-":
                voice = self.synth_engine.channels[self.synth_cursor_channel]
                voice.detune = max(-100, voice.detune - 10)
            elif key.name == "KEY_ESCAPE":
                self.view = "arrangement"
                
        elif self.synth_view_mode == "voice":
            categories = get_voice_categories()
            cat_names = list(categories.keys())
            voices = categories[cat_names[self.synth_voice_category]]
            
            if key.name == "KEY_UP":
                self.synth_voice_index = max(0, self.synth_voice_index - 1)
            elif key.name == "KEY_DOWN":
                self.synth_voice_index = min(len(voices) - 1, self.synth_voice_index + 1)
            elif key.name == "KEY_LEFT":
                self.synth_voice_category = max(0, self.synth_voice_category - 1)
                self.synth_voice_index = 0
            elif key.name == "KEY_RIGHT":
                self.synth_voice_category = min(len(cat_names) - 1, self.synth_voice_category + 1)
                self.synth_voice_index = 0
            elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                selected_voice = voices[self.synth_voice_index]
                self.synth_engine.channels[self.synth_cursor_channel].voice_type = selected_voice
                self.synth_view_mode = "channels"
            elif key.name == "KEY_ESCAPE":
                self.synth_view_mode = "channels"
                
        elif self.synth_view_mode == "adsr":
            voice = self.synth_engine.channels[self.synth_cursor_channel]
            
            if key.name == "KEY_UP":
                self.synth_cursor_param = max(0, self.synth_cursor_param - 1)
            elif key.name == "KEY_DOWN":
                self.synth_cursor_param = min(3, self.synth_cursor_param + 1)
            elif self._is_shift_left(key):
                if self.synth_cursor_param == 0:  # Attack
                    voice.adsr.attack = max(0.001, voice.adsr.attack - 0.01)
                elif self.synth_cursor_param == 1:  # Decay
                    voice.adsr.decay = max(0.001, voice.adsr.decay - 0.01)
                elif self.synth_cursor_param == 2:  # Sustain
                    voice.adsr.sustain = max(0.0, voice.adsr.sustain - 0.05)
                elif self.synth_cursor_param == 3:  # Release
                    voice.adsr.release = max(0.001, voice.adsr.release - 0.01)
            elif self._is_shift_right(key):
                if self.synth_cursor_param == 0:  # Attack
                    voice.adsr.attack = min(2.0, voice.adsr.attack + 0.01)
                elif self.synth_cursor_param == 1:  # Decay
                    voice.adsr.decay = min(2.0, voice.adsr.decay + 0.01)
                elif self.synth_cursor_param == 2:  # Sustain
                    voice.adsr.sustain = min(1.0, voice.adsr.sustain + 0.05)
                elif self.synth_cursor_param == 3:  # Release
                    voice.adsr.release = min(5.0, voice.adsr.release + 0.01)
            elif key.name == "KEY_ESCAPE":
                self.synth_view_mode = "channels"
                
        elif self.synth_view_mode == "filter":
            voice = self.synth_engine.channels[self.synth_cursor_channel]
            
            if key.name == "KEY_UP":
                self.synth_cursor_param = max(0, self.synth_cursor_param - 1)
            elif key.name == "KEY_DOWN":
                self.synth_cursor_param = min(2, self.synth_cursor_param + 1)
            elif self._is_shift_left(key):
                if self.synth_cursor_param == 0:  # Cutoff
                    voice.filter.cutoff = max(20.0, voice.filter.cutoff - 100)
                elif self.synth_cursor_param == 1:  # Resonance
                    voice.filter.resonance = max(0.1, voice.filter.resonance - 0.1)
                elif self.synth_cursor_param == 2:  # Type
                    types = ["lowpass", "highpass", "bandpass"]
                    idx = types.index(voice.filter.filter_type)
                    voice.filter.filter_type = types[(idx - 1) % len(types)]
            elif self._is_shift_right(key):
                if self.synth_cursor_param == 0:  # Cutoff
                    voice.filter.cutoff = min(20000.0, voice.filter.cutoff + 100)
                elif self.synth_cursor_param == 1:  # Resonance
                    voice.filter.resonance = min(10.0, voice.filter.resonance + 0.1)
                elif self.synth_cursor_param == 2:  # Type
                    types = ["lowpass", "highpass", "bandpass"]
                    idx = types.index(voice.filter.filter_type)
                    voice.filter.filter_type = types[(idx + 1) % len(types)]
            elif key.name == "KEY_ESCAPE":
                self.synth_view_mode = "channels"
                
        elif self.synth_view_mode == "drums":
            voice = self.synth_engine.channels[self.synth_cursor_channel]
            is_drum = self.synth_engine._is_drum_voice(voice.voice_type)
            
            if not is_drum:
                if key.name == "KEY_ESCAPE":
                    self.synth_view_mode = "channels"
                return
            
            if key.name == "KEY_UP":
                self.synth_cursor_param = max(0, self.synth_cursor_param - 1)
            elif key.name == "KEY_DOWN":
                self.synth_cursor_param = min(5, self.synth_cursor_param + 1)
            elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                if self.synth_cursor_param == 2:  # Filter enabled toggle
                    voice.drum_filter.enabled = not voice.drum_filter.enabled
            elif self._is_shift_left(key):
                if self.synth_cursor_param == 0:  # Length multiplier
                    voice.drum_length_multiplier = max(0.1, voice.drum_length_multiplier - 0.05)
                elif self.synth_cursor_param == 1:  # Release envelope
                    voice.drum_release_envelope = max(0.0, voice.drum_release_envelope - 0.05)
                elif self.synth_cursor_param == 3:  # Filter cutoff
                    voice.drum_filter.cutoff = max(20.0, voice.drum_filter.cutoff - 100)
                elif self.synth_cursor_param == 4:  # Filter resonance
                    voice.drum_filter.resonance = max(0.1, voice.drum_filter.resonance - 0.1)
                elif self.synth_cursor_param == 5:  # Filter type
                    types = ["lowpass", "highpass", "bandpass"]
                    idx = types.index(voice.drum_filter.filter_type)
                    voice.drum_filter.filter_type = types[(idx - 1) % len(types)]
            elif self._is_shift_right(key):
                if self.synth_cursor_param == 0:  # Length multiplier
                    voice.drum_length_multiplier = min(2.0, voice.drum_length_multiplier + 0.05)
                elif self.synth_cursor_param == 1:  # Release envelope
                    voice.drum_release_envelope = min(2.0, voice.drum_release_envelope + 0.05)
                elif self.synth_cursor_param == 3:  # Filter cutoff
                    voice.drum_filter.cutoff = min(20000.0, voice.drum_filter.cutoff + 100)
                elif self.synth_cursor_param == 4:  # Filter resonance
                    voice.drum_filter.resonance = min(10.0, voice.drum_filter.resonance + 0.1)
                elif self.synth_cursor_param == 5:  # Filter type
                    types = ["lowpass", "highpass", "bandpass"]
                    idx = types.index(voice.drum_filter.filter_type)
                    voice.drum_filter.filter_type = types[(idx + 1) % len(types)]
            elif key.name == "KEY_ESCAPE":
                self.synth_view_mode = "channels"
                
        elif self.synth_view_mode == "effects":
            voice = self.synth_engine.channels[self.synth_cursor_channel]
            
            if key.name == "KEY_UP":
                if self.synth_cursor_param < 5:  # Effect sends
                    self.synth_cursor_param = max(0, self.synth_cursor_param - 1)
                else:  # Effect parameters - navigate individual parameters
                    self.synth_effect_param_idx = max(0, self.synth_effect_param_idx - 1)
            elif key.name == "KEY_DOWN":
                if self.synth_cursor_param < 5:  # Effect sends
                    self.synth_cursor_param = min(4, self.synth_cursor_param + 1)  # 5 effect sends
                else:  # Effect parameters - navigate individual parameters
                    # Get max parameters for current effect
                    effect_idx = self.synth_cursor_param - 5
                    if effect_idx == 0:  # Chorus
                        max_params = 5
                    elif effect_idx == 1:  # Delay
                        max_params = 5
                    elif effect_idx == 2:  # Reverb
                        max_params = 5
                    elif effect_idx == 3:  # Compression
                        max_params = 7
                    elif effect_idx == 4:  # Crush
                        max_params = 4
                    else:
                        max_params = 5
                    
                    self.synth_effect_param_idx = min(max_params - 1, self.synth_effect_param_idx + 1)
            elif key.name == "KEY_TAB":
                # Switch between sends and effect parameters
                if self.synth_cursor_param < 5:  # Currently on sends
                    self.synth_cursor_param += 5  # Jump to effect parameters
                    self.synth_effect_param_idx = 0  # Reset parameter cursor
                else:  # Currently on effect parameters
                    self.synth_cursor_param -= 5  # Back to sends
            elif self._is_shift_left(key):
                if self.synth_cursor_param < 5:  # Effect sends
                    if self.synth_cursor_param == 0:
                        voice.send_chorus = max(0.0, voice.send_chorus - 0.05)
                    elif self.synth_cursor_param == 1:
                        voice.send_delay = max(0.0, voice.send_delay - 0.05)
                    elif self.synth_cursor_param == 2:
                        voice.send_reverb = max(0.0, voice.send_reverb - 0.05)
                    elif self.synth_cursor_param == 3:
                        voice.send_compression = max(0.0, voice.send_compression - 0.05)
                    elif self.synth_cursor_param == 4:
                        voice.send_crush = max(0.0, voice.send_crush - 0.05)
                else:  # Effect parameters
                    from synth_engine import EffectType
                    effect_types = [EffectType.CHORUS, EffectType.DELAY, EffectType.REVERB, EffectType.COMPRESSION, EffectType.CRUSH]
                    effect_idx = self.synth_cursor_param - 5
                    if 0 <= effect_idx < len(effect_types):
                        bus = self.synth_engine.effects_buses[effect_types[effect_idx]]
                        params = bus.params
                        
                        # Adjust individual parameter based on cursor position
                        if effect_types[effect_idx] == EffectType.CHORUS:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Rate
                                params.rate = max(0.1, params.rate - 0.1)
                            elif self.synth_effect_param_idx == 2:  # Depth
                                params.depth = max(0.001, params.depth - 0.001)
                            elif self.synth_effect_param_idx == 3:  # Feedback
                                params.feedback = max(0.0, params.feedback - 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = max(0.0, params.wet_mix - 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.DELAY:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Time
                                params.time = max(0.01, params.time - 0.01)
                            elif self.synth_effect_param_idx == 2:  # Feedback
                                params.feedback = max(0.0, params.feedback - 0.05)
                            elif self.synth_effect_param_idx == 3:  # Cross Feedback
                                params.cross_feedback = max(0.0, params.cross_feedback - 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = max(0.0, params.wet_mix - 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.REVERB:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Room Size
                                params.room_size = max(0.0, params.room_size - 0.05)
                            elif self.synth_effect_param_idx == 2:  # Damping
                                params.damping = max(0.0, params.damping - 0.05)
                            elif self.synth_effect_param_idx == 3:  # Width
                                params.width = max(0.0, params.width - 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = max(0.0, params.wet_mix - 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.COMPRESSION:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Threshold
                                params.threshold = max(-60.0, params.threshold - 1.0)
                            elif self.synth_effect_param_idx == 2:  # Ratio
                                params.ratio = max(1.0, params.ratio - 0.5)
                            elif self.synth_effect_param_idx == 3:  # Attack
                                params.attack = max(0.001, params.attack - 0.001)
                            elif self.synth_effect_param_idx == 4:  # Release
                                params.release = max(0.01, params.release - 0.01)
                            elif self.synth_effect_param_idx == 5:  # Makeup Gain
                                params.makeup_gain = max(-20.0, params.makeup_gain - 1.0)
                            elif self.synth_effect_param_idx == 6:  # Wet Mix
                                params.wet_mix = max(0.0, params.wet_mix - 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.CRUSH:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Bits
                                params.bits = max(1, params.bits - 1)
                            elif self.synth_effect_param_idx == 2:  # Downsample
                                params.downsample = max(1, params.downsample - 1)
                            elif self.synth_effect_param_idx == 3:  # Wet Mix
                                params.wet_mix = max(0.0, params.wet_mix - 0.05)
                                
            elif self._is_shift_right(key):
                if self.synth_cursor_param < 5:  # Effect sends
                    if self.synth_cursor_param == 0:
                        voice.send_chorus = min(1.0, voice.send_chorus + 0.05)
                    elif self.synth_cursor_param == 1:
                        voice.send_delay = min(1.0, voice.send_delay + 0.05)
                    elif self.synth_cursor_param == 2:
                        voice.send_reverb = min(1.0, voice.send_reverb + 0.05)
                    elif self.synth_cursor_param == 3:
                        voice.send_compression = min(1.0, voice.send_compression + 0.05)
                    elif self.synth_cursor_param == 4:
                        voice.send_crush = min(1.0, voice.send_crush + 0.05)
                else:  # Effect parameters
                    from synth_engine import EffectType
                    effect_types = [EffectType.CHORUS, EffectType.DELAY, EffectType.REVERB, EffectType.COMPRESSION, EffectType.CRUSH]
                    effect_idx = self.synth_cursor_param - 5
                    if 0 <= effect_idx < len(effect_types):
                        bus = self.synth_engine.effects_buses[effect_types[effect_idx]]
                        params = bus.params
                        
                        # Adjust individual parameter based on cursor position
                        if effect_types[effect_idx] == EffectType.CHORUS:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Rate
                                params.rate = min(10.0, params.rate + 0.1)
                            elif self.synth_effect_param_idx == 2:  # Depth
                                params.depth = min(0.1, params.depth + 0.001)
                            elif self.synth_effect_param_idx == 3:  # Feedback
                                params.feedback = min(1.0, params.feedback + 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = min(1.0, params.wet_mix + 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.DELAY:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Time
                                params.time = min(2.0, params.time + 0.01)
                            elif self.synth_effect_param_idx == 2:  # Feedback
                                params.feedback = min(0.95, params.feedback + 0.05)
                            elif self.synth_effect_param_idx == 3:  # Cross Feedback
                                params.cross_feedback = min(1.0, params.cross_feedback + 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = min(1.0, params.wet_mix + 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.REVERB:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Room Size
                                params.room_size = min(1.0, params.room_size + 0.05)
                            elif self.synth_effect_param_idx == 2:  # Damping
                                params.damping = min(1.0, params.damping + 0.05)
                            elif self.synth_effect_param_idx == 3:  # Width
                                params.width = min(1.0, params.width + 0.05)
                            elif self.synth_effect_param_idx == 4:  # Wet Mix
                                params.wet_mix = min(1.0, params.wet_mix + 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.COMPRESSION:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Threshold
                                params.threshold = min(0.0, params.threshold + 1.0)
                            elif self.synth_effect_param_idx == 2:  # Ratio
                                params.ratio = min(20.0, params.ratio + 0.5)
                            elif self.synth_effect_param_idx == 3:  # Attack
                                params.attack = min(0.1, params.attack + 0.001)
                            elif self.synth_effect_param_idx == 4:  # Release
                                params.release = min(2.0, params.release + 0.01)
                            elif self.synth_effect_param_idx == 5:  # Makeup Gain
                                params.makeup_gain = min(20.0, params.makeup_gain + 1.0)
                            elif self.synth_effect_param_idx == 6:  # Wet Mix
                                params.wet_mix = min(1.0, params.wet_mix + 0.05)
                                
                        elif effect_types[effect_idx] == EffectType.CRUSH:
                            if self.synth_effect_param_idx == 0:  # Enabled
                                params.enabled = not params.enabled
                            elif self.synth_effect_param_idx == 1:  # Bits
                                params.bits = min(16, params.bits + 1)
                            elif self.synth_effect_param_idx == 2:  # Downsample
                                params.downsample = min(32, params.downsample + 1)
                            elif self.synth_effect_param_idx == 3:  # Wet Mix
                                params.wet_mix = min(1.0, params.wet_mix + 0.05)
                                
            elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                if self.synth_cursor_param < 5:  # Effect sends
                    return  # No toggle for sends
                else:  # Effect parameters
                    from synth_engine import EffectType
                    effect_types = [EffectType.CHORUS, EffectType.DELAY, EffectType.REVERB, EffectType.COMPRESSION, EffectType.CRUSH]
                    effect_idx = self.synth_cursor_param - 5
                    if 0 <= effect_idx < len(effect_types):
                        bus = self.synth_engine.effects_buses[effect_types[effect_idx]]
                        # Toggle enabled state when on first parameter
                        if self.synth_effect_param_idx == 0:
                            bus.params.enabled = not bus.params.enabled
            elif key.name == "KEY_ESCAPE":
                self.synth_view_mode = "channels"

    # ── MIDI port selection ──────────────────────────────────

    def select_midi_port(self):
        t = self.term
        available_ports = mido.get_output_names()

        if not available_ports:
            buf = self._clear_screen()
            w = t.width
            buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
            buf.append(
                t.move_xy(2, 1) + t.bold_cyan(" MIDI PORT SELECTION ")
            )
            buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
            buf.append(
                t.move_xy(2, 5)
                + t.bold_yellow("ERROR: No MIDI output ports found!")
            )
            buf.append(t.move_xy(2, 7) + "Press any key to return...")
            self._flush(buf)
            t.inkey(timeout=None)
            return None

        selected_idx = 0

        while True:
            h, w = t.height, t.width
            buf = self._clear_screen()

            buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
            buf.append(
                t.move_xy(2, 1) + t.bold_cyan(" MIDI PORT SELECTION ")
            )
            buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
            buf.append(
                t.move_xy(2, 4) + t.bold("Select MIDI Output Port:")
            )
            buf.append(t.move_xy(0, 5) + "─" * (w - 1))

            for i, port_name in enumerate(available_ports):
                y = 7 + i
                if y >= h - 6:
                    break

                max_len = w - 10
                display = (
                    port_name
                    if len(port_name) <= max_len
                    else port_name[: max_len - 3] + "..."
                )

                if i == selected_idx:
                    buf.append(
                        t.move_xy(4, y)
                        + t.bold_reverse(f"► {i + 1}. {display}")
                    )
                else:
                    buf.append(
                        t.move_xy(4, y) + f"  {i + 1}. {display}"
                    )

            footer_y = h - 5
            buf.append(t.move_xy(0, footer_y) + "─" * (w - 1))

            current_port = (
                self.midi_out.name if self.midi_out else "None"
            )
            info_lines = [
                "↑/↓: Navigate | ENTER: Select Port | ESC: Cancel",
                f"Current: {current_port}",
            ]
            for i, line in enumerate(info_lines):
                buf.append(
                    t.move_xy(2, footer_y + 1 + i) + t.magenta(line)
                )

            self._flush(buf)

            key = t.inkey(timeout=None)

            if key.name == "KEY_UP":
                selected_idx = (selected_idx - 1) % len(available_ports)
            elif key.name == "KEY_DOWN":
                selected_idx = (selected_idx + 1) % len(available_ports)
            elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                return available_ports[selected_idx]
            elif key.name == "KEY_ESCAPE":
                return None
            elif str(key).isdigit():
                num = int(str(key))
                if 1 <= num <= len(available_ports):
                    return available_ports[num - 1]

    def change_midi_port(self, new_port_name):
        t = self.term
        try:
            if self.midi_out:
                for ch in range(16):
                    self.midi_out.send(
                        mido.Message(
                            "control_change",
                            control=123, value=0, channel=ch,
                        )
                    )
                self.midi_out.close()

            self.midi_out = mido.open_output(new_port_name)
            return True

        except Exception as e:
            h, w = t.height, t.width
            buf = self._clear_screen()
            buf.append(t.move_xy(0, 0) + t.bold("═" * (w - 1)))
            buf.append(t.move_xy(2, 1) + t.bold_yellow(" ERROR "))
            buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))
            buf.append(
                t.move_xy(2, 5)
                + t.yellow(f"Failed to open MIDI port: {new_port_name}")
            )
            buf.append(t.move_xy(2, 6) + t.yellow(f"Error: {e}"))
            buf.append(t.move_xy(2, 8) + "Press any key to return...")
            self._flush(buf)
            t.inkey(timeout=None)
            return False

    # ── tempo input (replaces curses echo/getstr) ────────────

    def get_tempo_input(self):
        t = self.term
        tempo_str = ""
        prompt = "Enter tempo (40-300): "

        while True:
            display = t.move_xy(0, 0) + prompt + tempo_str + t.clear_eol
            sys.stdout.write(display)
            sys.stdout.flush()

            key = t.inkey(timeout=None)

            if key.name == "KEY_ENTER" or key in ("\n", "\r"):
                break
            elif key.name == "KEY_ESCAPE":
                return None
            elif self._is_backspace(key):
                tempo_str = tempo_str[:-1]
            elif str(key).isdigit() and len(tempo_str) < 3:
                tempo_str += str(key)

        try:
            return max(40, min(300, int(tempo_str)))
        except ValueError:
            return None

    # ── input handling ───────────────────────────────────────

    def handle_phrase_input(self, key):
        phrase = self.phrases[self.current_phrase_num]
        max_pages = phrase.length // 16

        # ── header mode (length / default_len / page / offset selectors) ──
        if self.phrase_cursor == -1:
            if key.name == "KEY_DOWN":
                self.phrase_cursor = 0
            elif key.name == "KEY_LEFT":
                self.phrase_header_field = max(
                    0, self.phrase_header_field - 1
                )
            elif key.name == "KEY_RIGHT":
                self.phrase_header_field = min(
                    3, self.phrase_header_field + 1
                )
            elif self._is_shift_right(key):
                if self.phrase_header_field == 0:  # Length
                    idx = self.length_options.index(phrase.length)
                    if idx < len(self.length_options) - 1:
                        new_length = self.length_options[idx + 1]
                        self._set_phrase_length(phrase, new_length)
                elif self.phrase_header_field == 1:  # Default note length
                    idx = self.note_length_options.index(phrase.default_note_length)
                    if idx < len(self.note_length_options) - 1:
                        phrase.default_note_length = self.note_length_options[idx + 1]
                elif self.phrase_header_field == 2:  # Page
                    new_max = phrase.length // 16
                    if self.phrase_page < new_max - 1:
                        self.phrase_page += 1
                elif self.phrase_header_field == 3:  # Offset
                    self.offset_phrase(self.current_phrase_num, 1)
            elif self._is_shift_left(key):
                if self.phrase_header_field == 0:  # Length
                    idx = self.length_options.index(phrase.length)
                    if idx > 0:
                        new_length = self.length_options[idx - 1]
                        self._set_phrase_length(phrase, new_length)
                        # Clamp page if it's now out of range
                        new_max = new_length // 16
                        if self.phrase_page >= new_max:
                            self.phrase_page = new_max - 1
                elif self.phrase_header_field == 1:  # Default note length
                    idx = self.note_length_options.index(phrase.default_note_length)
                    if idx > 0:
                        phrase.default_note_length = self.note_length_options[idx - 1]
                elif self.phrase_header_field == 2:  # Page
                    if self.phrase_page > 0:
                        self.phrase_page -= 1
                elif self.phrase_header_field == 3:  # Offset
                    self.offset_phrase(self.current_phrase_num, -1)
            elif key.name == "KEY_ESCAPE":
                self.view = "arrangement"
            return

        # ── step mode ──
        step_idx = self.phrase_page * 16 + self.phrase_cursor
        # Safety clamp in case length was reduced externally
        if step_idx >= len(phrase.steps):
            self.phrase_page = 0
            self.phrase_cursor = 0
            step_idx = 0
        step = phrase.steps[step_idx]

        if key.name == "KEY_UP":
            if self.phrase_cursor > 0:
                self.phrase_cursor -= 1
            else:
                self.phrase_cursor = -1  # Move to header
        elif key.name == "KEY_DOWN":
            self.phrase_cursor = min(15, self.phrase_cursor + 1)
        elif key.name == "KEY_LEFT":
            self.phrase_field = max(0, self.phrase_field - 1)
        elif key.name == "KEY_RIGHT":
            self.phrase_field = min(4, self.phrase_field + 1)
        elif self._is_shift_right(key):
            if self.phrase_field == 0:  # Note
                step.note = min(127, (step.note or 60) + 1)
            elif self.phrase_field == 1:  # Note length
                idx = self.note_length_options.index(step.note_length)
                step.note_length = self.note_length_options[
                    (idx + 1) % len(self.note_length_options)
                ]
            elif self.phrase_field == 2:  # Velocity
                step.velocity = min(127, step.velocity + 1)
            elif self.phrase_field == 3:  # Probability
                step.probability = min(100, step.probability + 10)
            elif self.phrase_field == 4:  # Condition
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[
                    (idx + 1) % len(self.condition_options)
                ]
        elif self._is_shift_left(key):
            if self.phrase_field == 0:  # Note
                step.note = max(0, (step.note or 60) - 1)
            elif self.phrase_field == 1:  # Note length
                idx = self.note_length_options.index(step.note_length)
                step.note_length = self.note_length_options[
                    (idx - 1) % len(self.note_length_options)
                ]
            elif self.phrase_field == 2:  # Velocity
                step.velocity = max(0, step.velocity - 1)
            elif self.phrase_field == 3:  # Probability
                step.probability = max(0, step.probability - 10)
            elif self.phrase_field == 4:  # Condition
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[
                    (idx - 1) % len(self.condition_options)
                ]
        elif self._is_shift_up(key):
            if self.phrase_field == 0:  # Note
                step.note = min(127, (step.note or 60) + 12)  # +1 octave
        elif self._is_shift_down(key):
            if self.phrase_field == 0:  # Note
                step.note = max(0, (step.note or 60) - 12)
        elif self._is_backspace(key):
            if self.phrase_field == 0:  # Note
                step.note = None
        elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
            if self.phrase_field == 0:  # Note
                step.note = 60
        elif key.name == "KEY_ESCAPE":
            self.view = "arrangement"

    # ── main loop ────────────────────────────────────────────

    def run(self):
        t = self.term

        with t.fullscreen(), t.cbreak(), t.hidden_cursor():
            while True:
                if self.view == "arrangement":
                    self.draw_arrangement()
                elif self.view == "phrase":
                    self.draw_phrase()
                elif self.view == "synth":
                    self.draw_synth_engine()

                key = t.inkey(timeout=0.02)

                if not key:
                    continue

                if self.view == "phrase":
                    self.handle_phrase_input(key)
                    continue
                
                if self.view == "synth":
                    self.handle_synth_input(key)
                    continue

                # ── arrangement view controls ──
                if key.name == "KEY_UP":
                    self.cursor_row = max(0, self.cursor_row - 1)

                elif key.name == "KEY_DOWN":
                    self.cursor_row = min(63, self.cursor_row + 1)

                elif key.name == "KEY_LEFT":
                    self.cursor_col = max(0, self.cursor_col - 1)

                elif key.name == "KEY_RIGHT":
                    self.cursor_col = min(7, self.cursor_col + 1)

                elif self._is_shift_left(key):
                    self.current_phrase_num = max(
                        0, self.current_phrase_num - 1
                    )
                    self.arrangement[self.cursor_row][
                        self.cursor_col
                    ] = self.current_phrase_num

                elif self._is_shift_right(key):
                    self.current_phrase_num = min(
                        127, self.current_phrase_num + 1
                    )
                    self.arrangement[self.cursor_row][
                        self.cursor_col
                    ] = self.current_phrase_num

                elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
                    existing = self.arrangement[self.cursor_row][
                        self.cursor_col
                    ]
                    if existing is not None:
                        self.current_phrase_num = existing
                        self.view = "phrase"
                        self.phrase_cursor = 0
                        self.phrase_field = 0
                        self.phrase_page = 0
                    else:
                        self.arrangement[self.cursor_row][
                            self.cursor_col
                        ] = self.current_phrase_num

                elif self._is_backspace(key):
                    self.arrangement[self.cursor_row][
                        self.cursor_col
                    ] = None

                elif key == " ":
                    if self.play_mode == "pattern":
                        self.start_playback(self.cursor_row)
                    else:
                        if self.playing:
                            self.stop_playback_func()
                        else:
                            self.start_playback(self.cursor_row)

                elif key == ".":
                    self.stop_playback_func()

                elif key.name == "KEY_TAB" or key == "\t":
                    self.toggle_play_mode()

                elif key in ("t", "T"):
                    new_tempo = self.get_tempo_input()
                    if new_tempo is not None:
                        self.tempo = new_tempo
                
                elif key in ("s", "S") and self.use_synth:
                    self.view = "synth"
                    self.synth_view_mode = "channels"

                elif key in ("q", "Q"):
                    if self.playing:
                        self.stop_playback_func()
                        if self.playback_thread:
                            self.playback_thread.join(timeout=1.0)
                    if self.use_synth and self.synth_engine:
                        self.synth_engine.stop()
                    break

                elif key.name == "KEY_ESCAPE":
                    self.esc_menu()


def main():
    parser = argparse.ArgumentParser(description='Terminal MIDI Phrase Tracker')
    parser.add_argument('--synth', action='store_true', 
                       help='Enable synth engine on startup')
    args = parser.parse_args()
    
    if args.synth and not SYNTH_AVAILABLE:
        print("Error: Synth engine requires numpy, sounddevice, and scipy")
        print("Install with: pip install numpy sounddevice scipy")
        sys.exit(1)
    
    tracker = TRKR(use_synth=args.synth)
    tracker.run()


if __name__ == "__main__":
    main()

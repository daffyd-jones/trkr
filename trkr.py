#!/usr/bin/env python3
"""
Terminal MIDI Phrase Tracker
Requirements: pip install mido python-rtmidi
"""

import curses
import threading
import time
import random
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import mido
from mido import Message

@dataclass
class PhraseStep:
    note: Optional[int] = None
    velocity: int = 100
    probability: int = 100
    condition: str = "1/1"

@dataclass
class Phrase:
    steps: List[PhraseStep] = field(default_factory=lambda: [PhraseStep() for _ in range(16)])

def midi_to_note(midi_number):
    if midi_number is None:
        return "---"
    """Convert MIDI number to note representation using flats."""
    if not 0 <= midi_number <= 127:
        raise ValueError("MIDI number must be between 0 and 127")

    # Note names using flats
    note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]

    return f"{note}{octave}"

class MidiTracker:
    def __init__(self):
        self.phrases = {i: Phrase() for i in range(128)}
        self.arrangement = [[None for _ in range(8)] for _ in range(64)]
        self.current_notes = [None] * 8 
        self.current_phrase_num = 0
        self.cursor_row = 0
        self.cursor_col = 0
        self.view = "arrangement"  # "arrangement" or "phrase"
        self.phrase_cursor = 0
        self.phrase_field = 0  # 0=note, 1=vel, 2=prob, 3=cond
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
        
        # Initialize MIDI
        try:
            self.midi_out = mido.open_output()
        except:
            self.midi_out = None
        
        self.condition_options = [
            "1/1", "1/2", "2/2", "1/3", "2/3", "3/3", "1/4", "2/4", "3/4", "4/4",
            "1/5", "2/5", "3/5", "4/5", "5/5", "1/6", "2/6", "3/6", "4/6", "5/6", "6/6",
            "1/7", "2/7", "3/7", "4/7", "5/7", "6/7", "7/7", "1/8", "2/8", "3/8", "4/8",
            "5/8", "6/8", "7/8", "8/8"
        ]

    def save_arrangement(self, filename):
        """Save the current arrangement, phrases, and settings to a JSON file."""
        save_data = {
            'version': '1.0',
            'tempo': self.tempo,
            'play_mode': self.play_mode,
            'current_phrase_num': self.current_phrase_num,
            'arrangement': self.arrangement,
            'phrases': {},
            'condition_options': self.condition_options
        }
        
        # Convert phrases to serializable format
        for phrase_num, phrase in self.phrases.items():
            save_data['phrases'][str(phrase_num)] = {
                'steps': [asdict(step) for step in phrase.steps]
            }
        
        try:
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            return True
        except Exception as e:
            return False
    
    def load_arrangement(self, filename):
        """Load arrangement, phrases, and settings from a JSON file."""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
            
            # Load basic settings
            self.tempo = save_data.get('tempo', 120)
            self.play_mode = save_data.get('play_mode', 'pattern')
            self.current_phrase_num = save_data.get('current_phrase_num', 0)
            
            # Load arrangement
            self.arrangement = save_data.get('arrangement', [[None for _ in range(8)] for _ in range(64)])
            
            # Load phrases
            phrases_data = save_data.get('phrases', {})
            for phrase_num_str, phrase_data in phrases_data.items():
                phrase_num = int(phrase_num_str)
                steps_data = phrase_data.get('steps', [])
                
                # Create Phrase object with loaded steps
                steps = []
                for step_data in steps_data:
                    step = PhraseStep(
                        note=step_data.get('note'),
                        velocity=step_data.get('velocity', 100),
                        probability=step_data.get('probability', 100),
                        condition=step_data.get('condition', '1/1')
                    )
                    steps.append(step)
                
                # Ensure we have exactly 16 steps
                while len(steps) < 16:
                    steps.append(PhraseStep())
                self.phrases[phrase_num] = Phrase(steps=steps[:16])
            
            return True
        except Exception as e:
            return False
    
    def get_save_files(self):
        """Get list of existing save files in current directory."""
        save_files = []
        for file in os.listdir('.'):
            if file.endswith('.trkr'):
                save_files.append(file)
        return sorted(save_files)

    def get_current_note(self, channel):
        """Get the current note name for a channel."""
        note_num = self.current_notes[channel]
        # print(f"{note_num}")
        if note_num is not None:
            return midi_to_note(note_num)
        return None
    
    def should_trigger(self, step, step_key):
        if random.random() * 100 > step.probability:
            return False
        
        if step.condition == "1/1":
            return True
        
        num, denom = map(int, step.condition.split('/'))
        key = f"{step_key}_{step.condition}"
        count = self.condition_counters.get(key, 0) + 1
        self.condition_counters[key] = count % denom
        
        return count % denom == num - 1
    
    def send_midi(self, channel, note, velocity):
        # Use the selected MIDI output (self.output if available, otherwise self.midi_out)
        midi_output = getattr(self, 'output', None) or self.midi_out
        if midi_output:
            try:
                midi_output.send(Message('note_on', channel=channel, note=note, velocity=velocity))
                # Schedule note off after 50ms
                threading.Timer(0.05, lambda: midi_output.send(
                    Message('note_off', channel=channel, note=note)
                )).start()
            except:
                pass
    
    def playback_loop(self):
        step_time = (60 / self.tempo / 4)
        last_step_time = time.time()
        
        while not self.stop_playback:
            current_time = time.time()
            
            if current_time - last_step_time >= step_time:
                last_step_time = current_time
                
                # Process current steps for all channels
                for channel in range(8):
                    phrase_num = self.arrangement[self.current_row][channel]
                    
                    if phrase_num is not None:
                        phrase = self.phrases[phrase_num]
                        step_idx = self.current_steps[channel]
                        step = phrase.steps[step_idx]
                        
                        if step.note is not None:
                            step_key = f"{self.current_row}_{channel}_{step_idx}"
                            if self.should_trigger(step, step_key):
                                self.current_notes[channel] = step.note
                                self.send_midi(channel, step.note, step.velocity)
                        # else:
                            # self.current_notes[channel] = None
                        self.current_steps[channel] = (step_idx + 1) % 16
                
                # Check if we've completed a bar (all steps back to 0)
                if all(s == 0 for s in self.current_steps):
                    # Handle pending stop in pattern mode
                    if self.pending_stop and self.play_mode == "pattern":
                        self.playing = False
                        self.pending_stop = False
                        break
                    
                    # Handle next row change
                    if self.next_row is not None:
                        self.current_row = self.next_row
                        self.next_row = None
                    elif self.play_mode == "song":
                        # Advance to next row in song mode
                        next_row = self.current_row + 1
                        if next_row >= 64 or all(p is None for p in self.arrangement[next_row]):
                            self.current_row = 0
                        else:
                            self.current_row = next_row
            
            time.sleep(0.001)
        
        self.playing = False
    
    def start_playback(self, row):
        """Start playback from a specific row, waiting for next bar if needed"""
        if self.playing:
            # If already playing, schedule the change for the next bar
            self.next_row = row
        else:
            # Start fresh
            self.current_row = row
            self.current_steps = [0] * 8
            self.playing = True
            self.pending_stop = False
            self.stop_playback = False
            self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
            self.playback_thread.start()
    
    def stop_playback_func(self):
        """Stop playback, waiting for current bar to finish in pattern mode"""
        if self.play_mode == "pattern":
            self.pending_stop = True
        else:
            self.playing = False
            self.stop_playback = True
            if self.playback_thread:
                self.playback_thread.join(timeout=1.0)
    
    def toggle_play_mode(self):
        """Toggle between pattern and song mode"""
        was_playing = self.playing
        if was_playing:
            self.stop_playback_func()
            if self.playback_thread:
                self.playback_thread.join(timeout=1.0)
        
        self.play_mode = "song" if self.play_mode == "pattern" else "pattern"
        self.playing = False
        self.pending_stop = False
        self.next_row = None
    
    # def draw_arrangement(self, stdscr):
    #     height, width = stdscr.getmaxyx()
    #     stdscr.clear()
        
    #     # Header
    #     stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
    #     title = f" MIDI PHRASE TRACKER - ARRANGEMENT "
    #     stdscr.addstr(1, 2, title, curses.A_BOLD | curses.color_pair(1))
        
    #     play_status = "PLAYING" if self.playing else ("STOPPING..." if self.pending_stop else "STOPPED")
    #     status = f"PHRASE:{self.current_phrase_num:03d} | MODE:{self.play_mode.upper()} | "
    #     status += f"{play_status} | TEMPO:{self.tempo} | ROW:{self.current_row:02d}"
    #     stdscr.addstr(1, width - len(status) - 2, status, curses.color_pair(2))
    #     stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        
    #     # Column headers
    #     headers = "ROW │ CH1  CH2  CH3  CH4  CH5  CH6  CH7  CH8"
    #     stdscr.addstr(3, 2, headers, curses.A_BOLD)
    #     stdscr.addstr(4, 0, "─" * (width - 1))
        
    #     # Arrangement grid
    #     start_row = max(0, self.cursor_row - 10)
    #     for i in range(start_row, min(64, start_row + height - 10)):
    #         y = 5 + (i - start_row)
    #         if y >= height - 5:
    #             break
            
    #         # Row number
    #         row_attr = curses.A_BOLD if i == self.current_row and self.playing else 0
    #         stdscr.addstr(y, 2, f"{i:02d}  │ ", row_attr | curses.color_pair(3))
            
    #         # Channels
    #         for ch in range(8):
    #             x = 9 + (ch * 5)
    #             phrase_num = self.arrangement[i][ch]
                
    #             text = f"{phrase_num:03d}" if phrase_num is not None else "---"
                
    #             attr = 0
    #             if i == self.cursor_row and ch == self.cursor_col:
    #                 attr = curses.A_REVERSE
    #             elif i == self.current_row and self.playing and phrase_num is not None:
    #                 attr = curses.color_pair(4)
                
    #             stdscr.addstr(y, x, text, attr)
        
    #     # Footer with controls
    #     stdscr.addstr(height - 5, 0, "─" * (width - 1))
        
    #     if self.play_mode == "pattern":
    #         controls = [
    #             "ARROWS:Navigate | ENTER:Edit Phrase | SHIFT+←→:Change Phrase# | SHIFT+BKSP:Remove",
    #             "SPACE:Play Row | BKSP:Stop | SHIFT+SPACE:Toggle Mode | T:Tempo | Q:Quit"
    #         ]
    #     else:
    #         controls = [
    #             "ARROWS:Navigate | ENTER:Edit Phrase | SHIFT+←→:Change Phrase# | SHIFT+BKSP:Remove", 
    #             "SPACE:Play/Stop Song | SHIFT+SPACE:Toggle Mode | T:Tempo | Q:Quit"
    #         ]
        
    #     for i, ctrl in enumerate(controls):
    #         stdscr.addstr(height - 4 + i, 2, ctrl, curses.color_pair(5))
        
    #     stdscr.refresh()

    def show_main_menu(self, stdscr):
        """Show the main ESC menu with submenus."""
        menu_options = ["Save/Load", "MIDI Settings", "Resume"]
        selected_idx = 0
        
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            
            # Header
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " MAIN MENU ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
            
            # Menu options
            for i, option in enumerate(menu_options):
                y = 5 + i
                if i == selected_idx:
                    attr = curses.A_REVERSE | curses.A_BOLD
                    prefix = "► "
                else:
                    attr = 0
                    prefix = "  "
                stdscr.addstr(y, 4, f"{prefix}{option}", attr)
            
            # Footer
            footer_y = height - 3
            stdscr.addstr(footer_y, 0, "─" * (width - 1))
            stdscr.addstr(footer_y + 1, 2, "↑/↓: Navigate | ENTER: Select | ESC: Resume", curses.color_pair(5))
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                selected_idx = (selected_idx - 1) % len(menu_options)
            elif key == curses.KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(menu_options)
            elif key == ord('\n'):  # Enter
                if selected_idx == 0:  # Save/Load
                    result = self.show_saveload_menu(stdscr)
                    if result == "quit":
                        return "quit"
                elif selected_idx == 1:  # MIDI Settings
                    self.show_midi_menu(stdscr)
                elif selected_idx == 2:  # Resume
                    break
            elif key == 27:  # ESC
                break
        
        return "resume"
    
    def show_saveload_menu(self, stdscr):
        """Show save/load submenu."""
        menu_options = ["Save Arrangement", "Load Arrangement", "Back"]
        selected_idx = 0
        
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            
            # Header
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " SAVE/LOAD ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
            
            # Menu options
            for i, option in enumerate(menu_options):
                y = 5 + i
                if i == selected_idx:
                    attr = curses.A_REVERSE | curses.A_BOLD
                    prefix = "► "
                else:
                    attr = 0
                    prefix = "  "
                stdscr.addstr(y, 4, f"{prefix}{option}", attr)
            
            # Footer
            footer_y = height - 3
            stdscr.addstr(footer_y, 0, "─" * (width - 1))
            stdscr.addstr(footer_y + 1, 2, "↑/↓: Navigate | ENTER: Select | ESC: Back", curses.color_pair(5))
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                selected_idx = (selected_idx - 1) % len(menu_options)
            elif key == curses.KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(menu_options)
            elif key == ord('\n'):  # Enter
                if selected_idx == 0:  # Save
                    self.show_save_dialog(stdscr)
                elif selected_idx == 1:  # Load
                    result = self.show_load_dialog(stdscr)
                    if result == "quit":
                        return "quit"
                elif selected_idx == 2:  # Back
                    break
            elif key == 27:  # ESC
                break
        
        return "resume"
    
    def show_save_dialog(self, stdscr):
        """Show save file dialog."""
        save_files = self.get_save_files()
        
        # Add "New File..." option at the top
        options = ["New File..."] + save_files + ["Cancel"]
        selected_idx = 0
        filename_input = ""
        input_mode = False
        
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            
            # Header
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " SAVE ARRANGEMENT ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
            
            if input_mode:
                stdscr.addstr(4, 2, "Enter filename (.trkr will be added):", curses.A_BOLD)
                stdscr.addstr(5, 2, filename_input + "_", curses.A_REVERSE)
                stdscr.addstr(7, 2, "ENTER: Save | ESC: Cancel", curses.color_pair(5))
            else:
                stdscr.addstr(4, 2, "Select save slot:", curses.A_BOLD)
                
                # Show options
                for i, option in enumerate(options):
                    y = 6 + i
                    if y >= height - 4:
                        break
                    
                    if i == selected_idx:
                        attr = curses.A_REVERSE | curses.A_BOLD
                        prefix = "► "
                    else:
                        attr = 0
                        prefix = "  "
                    
                    display_name = option
                    if len(display_name) > width - 15:
                        display_name = display_name[:width-18] + "..."
                    
                    stdscr.addstr(y, 4, f"{prefix}{display_name}", attr)
                
                # Footer
                footer_y = height - 3
                stdscr.addstr(footer_y, 0, "─" * (width - 1))
                stdscr.addstr(footer_y + 1, 2, "↑/↓: Navigate | ENTER: Select | ESC: Cancel", curses.color_pair(5))
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if input_mode:
                if key == 27:  # ESC
                    input_mode = False
                    filename_input = ""
                elif key == ord('\n'):  # Enter
                    if filename_input:
                        if not filename_input.endswith('.trkr'):
                            filename_input += '.trkr'
                        if self.save_arrangement(filename_input):
                            # Show success message briefly
                            self.show_message(stdscr, f"Saved to {filename_input}", 2)
                            break
                        else:
                            self.show_message(stdscr, "Failed to save!", 2)
                            input_mode = False
                            filename_input = ""
                elif key == curses.KEY_BACKSPACE or key == 127:
                    filename_input = filename_input[:-1]
                elif 32 <= key <= 126:  # Printable characters
                    if len(filename_input) < 20:
                        filename_input += chr(key)
            else:
                if key == curses.KEY_UP:
                    selected_idx = (selected_idx - 1) % len(options)
                elif key == curses.KEY_DOWN:
                    selected_idx = (selected_idx + 1) % len(options)
                elif key == ord('\n'):  # Enter
                    if selected_idx == 0:  # New File
                        input_mode = True
                        filename_input = ""
                    elif selected_idx == len(options) - 1:  # Cancel
                        break
                    else:  # Existing file
                        filename = save_files[selected_idx - 1]
                        if self.save_arrangement(filename):
                            self.show_message(stdscr, f"Saved to {filename}", 2)
                            break
                        else:
                            self.show_message(stdscr, "Failed to save!", 2)
                elif key == 27:  # ESC
                    break
    
    def show_load_dialog(self, stdscr):
        """Show load file dialog."""
        save_files = self.get_save_files()
        
        if not save_files:
            self.show_message(stdscr, "No save files found!", 2)
            return "resume"
        
        options = save_files + ["Cancel"]
        selected_idx = 0
        
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            
            # Header
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " LOAD ARRANGEMENT ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
            
            stdscr.addstr(4, 2, "Select file to load:", curses.A_BOLD)
            
            # Show options
            for i, option in enumerate(options):
                y = 6 + i
                if y >= height - 4:
                    break
                
                if i == selected_idx:
                    attr = curses.A_REVERSE | curses.A_BOLD
                    prefix = "► "
                else:
                    attr = 0
                    prefix = "  "
                
                display_name = option
                if len(display_name) > width - 15:
                    display_name = display_name[:width-18] + "..."
                
                stdscr.addstr(y, 4, f"{prefix}{display_name}", attr)
            
            # Footer
            footer_y = height - 3
            stdscr.addstr(footer_y, 0, "─" * (width - 1))
            stdscr.addstr(footer_y + 1, 2, "↑/↓: Navigate | ENTER: Load | ESC: Cancel", curses.color_pair(5))
            
            stdscr.refresh()
            
            # Handle input
            key = stdscr.getch()
            
            if key == curses.KEY_UP:
                selected_idx = (selected_idx - 1) % len(options)
            elif key == curses.KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(options)
            elif key == ord('\n'):  # Enter
                if selected_idx == len(options) - 1:  # Cancel
                    break
                else:  # Load file
                    filename = save_files[selected_idx]
                    if self.load_arrangement(filename):
                        self.show_message(stdscr, f"Loaded {filename}", 2)
                        return "quit"  # Signal to restart the interface
                    else:
                        self.show_message(stdscr, "Failed to load!", 2)
            elif key == 27:  # ESC
                break
        
        return "resume"
    
    def show_midi_menu(self, stdscr):
        """Show MIDI settings submenu."""
        selected_port = self.select_midi_port(stdscr)
        if selected_port:
            self.change_midi_port(stdscr, selected_port)
    
    def show_message(self, stdscr, message, duration=2):
        """Show a temporary message."""
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
        stdscr.addstr(1, 2, " MESSAGE ", curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        stdscr.addstr(height//2, (width - len(message))//2, message, curses.A_BOLD)
        stdscr.refresh()
        time.sleep(duration)

    def select_midi_port(self, stdscr):
        import mido
    
        # Get available MIDI output ports
        available_ports = mido.get_output_names()
    
        if not available_ports:
            # No ports available - show error message
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " MIDI PORT SELECTION ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        
            stdscr.addstr(5, 2, "ERROR: No MIDI output ports found!", curses.color_pair(2) | curses.A_BOLD)
            stdscr.addstr(7, 2, "Press any key to return...")
            stdscr.refresh()
            stdscr.getch()
            return None
    
        selected_idx = 0
    
        while True:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
        
            # Header
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " MIDI PORT SELECTION ", curses.A_BOLD | curses.color_pair(1))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        
            # Instructions
            stdscr.addstr(4, 2, "Select MIDI Output Port:", curses.A_BOLD)
            stdscr.addstr(5, 0, "─" * (width - 1))
        
            # Port list
            for i, port_name in enumerate(available_ports):
                y = 7 + i
                if y >= height - 6:
                    break
            
                if i == selected_idx:
                    attr = curses.A_REVERSE | curses.A_BOLD
                    prefix = "► "
                else:
                    attr = 0
                    prefix = "  "
            
                # Truncate port name if too long
                max_port_len = width - 10
                display_name = port_name if len(port_name) <= max_port_len else port_name[:max_port_len-3] + "..."
            
                stdscr.addstr(y, 4, f"{prefix}{i+1}. {display_name}", attr)
        
            # Footer
            footer_y = height - 5
            stdscr.addstr(footer_y, 0, "─" * (width - 1))
        
            controls = [
                "↑/↓: Navigate | ENTER: Select Port | ESC: Cancel",
                f"Current: {self.output.name if hasattr(self, 'output') and self.output else 'None'}"
            ]
        
            for i, ctrl in enumerate(controls):
                stdscr.addstr(footer_y + 1 + i, 2, ctrl, curses.color_pair(5))
        
            stdscr.refresh()
        
            # Handle input
            key = stdscr.getch()
        
            if key == curses.KEY_UP:
                selected_idx = (selected_idx - 1) % len(available_ports)
            elif key == curses.KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(available_ports)
            elif key == ord('\n'):  # Enter key
                return available_ports[selected_idx]
            elif key == 27:  # ESC key
                return None
            elif ord('1') <= key <= ord('9'):  # Number keys for quick selection
                num = key - ord('0')
                if 1 <= num <= len(available_ports):
                    return available_ports[num - 1]


    def change_midi_port(self, stdscr, new_port_name):
        """Change the MIDI output port."""
        import mido
    
        try:
            # Close existing port if open
            if hasattr(self, 'output') and self.output:
                # Send all notes off before closing
                for ch in range(16):
                    self.output.send(mido.Message('control_change', control=123, value=0, channel=ch))
                self.output.close()
            elif self.midi_out:
                # Also close the default midi_out if it exists
                for ch in range(16):
                    self.midi_out.send(mido.Message('control_change', control=123, value=0, channel=ch))
                self.midi_out.close()
        
            # Open new port and assign to both variables for consistency
            self.output = mido.open_output(new_port_name)
            self.midi_out = self.output  # Keep both in sync
        
            # Reset current notes tracking
            # self.current_notes = [None] * 8
        
            return True
        except Exception as e:
            # Show error message
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
            stdscr.addstr(1, 2, " ERROR ", curses.A_BOLD | curses.color_pair(2))
            stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        
            stdscr.addstr(5, 2, f"Failed to open MIDI port: {new_port_name}", curses.color_pair(2))
            stdscr.addstr(6, 2, f"Error: {str(e)}", curses.color_pair(2))
            stdscr.addstr(8, 2, "Press any key to return...")
            stdscr.refresh()
            stdscr.getch()
            return False

    def draw_arrangement(self, stdscr):
        height, width = stdscr.getmaxyx()
        stdscr.clear()
    
        # Header
        stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
        title = f" MIDI PHRASE TRACKER - ARRANGEMENT "
        stdscr.addstr(1, 2, title, curses.A_BOLD | curses.color_pair(1))
    
        play_status = "PLAYING" if self.playing else ("STOPPING..." if self.pending_stop else "STOPPED")
        status = f"PHRASE:{self.current_phrase_num:03d} | MODE:{self.play_mode.upper()} | "
        status += f"{play_status} | TEMPO:{self.tempo} | ROW:{self.current_row:02d}"
        stdscr.addstr(1, width - len(status) - 2, status, curses.color_pair(2))
        stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
    
        # Column headers
        headers = "ROW │ CH1  CH2  CH3  CH4  CH5  CH6  CH7  CH8 │ CURRENT NOTES"
        stdscr.addstr(3, 2, headers, curses.A_BOLD)
        stdscr.addstr(4, 0, "─" * (width - 1))
    
        # Arrangement grid
        start_row = max(0, self.cursor_row - 10)
        for i in range(start_row, min(64, start_row + height - 10)):
            y = 5 + (i - start_row)
            if y >= height - 5:
                break
        
            # Row number
            row_attr = curses.A_BOLD if i == self.current_row and self.playing else 0
            stdscr.addstr(y, 2, f"{i:02d}  │ ", row_attr | curses.color_pair(3))
        
            # Channels
            for ch in range(8):
                x = 9 + (ch * 5)
                phrase_num = self.arrangement[i][ch]
            
                text = f"{phrase_num:03d}" if phrase_num is not None else "---"
            
                attr = 0
                if i == self.cursor_row and ch == self.cursor_col:
                    attr = curses.A_REVERSE
                elif i == self.current_row and self.playing and phrase_num is not None:
                    attr = curses.color_pair(4)
            
                stdscr.addstr(y, x, text, attr)
    
        # Current notes column (aligned to the right of the channels)
        notes_x = 9 + (8 * 5) + 2  # After all 8 channels + separator
    
        # Draw separator
        for i in range(start_row, min(64, start_row + height - 10)):
            y = 5 + (i - start_row)
            if y >= height - 5:
                break
            stdscr.addstr(y, notes_x - 2, "│")
    
        stdscr.addstr(5, notes_x, f"{midi_to_note(self.current_notes[0]):<3}|{midi_to_note(self.current_notes[1]):<3}", attr)
        stdscr.addstr(6, notes_x, f"{midi_to_note(self.current_notes[2]):<3}|{midi_to_note(self.current_notes[3]):<3}", attr)
        stdscr.addstr(7, notes_x, f"{midi_to_note(self.current_notes[4]):<3}|{midi_to_note(self.current_notes[5]):<3}", attr)
        stdscr.addstr(8, notes_x, f"{midi_to_note(self.current_notes[6]):<3}|{midi_to_note(self.current_notes[7]):<3}", attr)
        
        
        # Footer with controls
        stdscr.addstr(height - 5, 0, "─" * (width - 1))
    
        if self.play_mode == "pattern":
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase | SHIFT+←→:Change Phrase# | BKSP:Remove",
                "SPACE:Play Row | TAB:Toggle Mode/Stop Play | T:Tempo | Q:Quit"
            ]
        else:
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase | SHIFT+←→:Change Phrase# | SHIFT+BKSP:Remove", 
                "SPACE:Play/Stop Song | SHIFT+SPACE:Toggle Mode | T:Tempo | Q:Quit"
            ]
    
        for i, ctrl in enumerate(controls):
            stdscr.addstr(height - 4 + i, 2, ctrl, curses.color_pair(5))
    
        stdscr.refresh()
 
    def draw_phrase(self, stdscr):
        height, width = stdscr.getmaxyx()
        stdscr.clear()
        
        phrase = self.phrases[self.current_phrase_num]
        
        # Header
        stdscr.addstr(0, 0, "═" * (width - 1), curses.A_BOLD)
        title = f" PHRASE {self.current_phrase_num:03d} EDITOR "
        stdscr.addstr(1, 2, title, curses.A_BOLD | curses.color_pair(1))
        stdscr.addstr(2, 0, "═" * (width - 1), curses.A_BOLD)
        
        # Column headers
        headers = "STEP │ NOTE   VEL  PROB%  COND"
        stdscr.addstr(3, 2, headers, curses.A_BOLD)
        stdscr.addstr(4, 0, "─" * (width - 1))
        
        # Phrase steps
        for i in range(16):
            y = 5 + i
            step = phrase.steps[i]
            
            # Check if this step is currently playing
            playing_here = False
            if self.playing:
                for ch in range(8):
                    if (self.arrangement[self.current_row][ch] == self.current_phrase_num 
                        and self.current_steps[ch] == i):
                        playing_here = True
                        break
            
            row_attr = curses.color_pair(4) if playing_here else 0
            
            # Step number
            stdscr.addstr(y, 2, f" {i:02d}  │ ", row_attr | curses.A_BOLD)
            
            # Note
            note_text = f"{midi_to_note(step.note)}" if step.note is not None else "---"
            attr = curses.A_REVERSE if i == self.phrase_cursor and self.phrase_field == 0 else row_attr
            stdscr.addstr(y, 10, note_text, attr)
            
            # Velocity
            vel_text = f"{step.velocity:3d}"
            attr = curses.A_REVERSE if i == self.phrase_cursor and self.phrase_field == 1 else row_attr
            stdscr.addstr(y, 16, vel_text, attr)
            
            # Probability
            prob_text = f"{step.probability:3d}"
            attr = curses.A_REVERSE if i == self.phrase_cursor and self.phrase_field == 2 else row_attr
            stdscr.addstr(y, 22, prob_text, attr)
            
            # Condition
            cond_text = f"{step.condition:>4s}"
            attr = curses.A_REVERSE if i == self.phrase_cursor and self.phrase_field == 3 else row_attr
            stdscr.addstr(y, 30, cond_text, attr)
        
        # Footer
        stdscr.addstr(height - 4, 0, "─" * (width - 1))
        controls = [
            "↑↓:Navigate Steps | ←→:Navigate Fields | SHIFT+←→:Adjust Value",
            "BACKSPACE:Clear Note | ESC:Back to Arrangement"
        ]
        for i, ctrl in enumerate(controls):
            stdscr.addstr(height - 3 + i, 2, ctrl, curses.color_pair(5))
        
        stdscr.refresh()
    
    def handle_phrase_input(self, key):
        phrase = self.phrases[self.current_phrase_num]
        step = phrase.steps[self.phrase_cursor]
        
        if key == curses.KEY_UP:
            self.phrase_cursor = max(0, self.phrase_cursor - 1)
        elif key == curses.KEY_DOWN:
            self.phrase_cursor = min(15, self.phrase_cursor + 1)
        elif key == curses.KEY_LEFT:
            self.phrase_field = max(0, self.phrase_field - 1)
        elif key == curses.KEY_RIGHT:
            self.phrase_field = min(3, self.phrase_field + 1)
        elif key == curses.KEY_SRIGHT:
            if self.phrase_field == 0:  # Note
                step.note = min(127, (step.note or 60) + 1)
            elif self.phrase_field == 1:  # Velocity
                step.velocity = min(127, step.velocity + 1)
            elif self.phrase_field == 2:  # Probability
                step.probability = min(100, step.probability + 10)
            elif self.phrase_field == 3:
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[(idx + 1) % len(self.condition_options)]
        elif key == curses.KEY_SLEFT:
            if self.phrase_field == 0:  # Note
                step.note = max(0, (step.note or 60) - 1)
            elif self.phrase_field == 1:  # Velocity
                step.velocity = max(0, step.velocity - 1)
            elif self.phrase_field == 2:  # Probability
                step.probability = max(0, step.probability - 10)
            elif self.phrase_field == 3:
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[(idx - 1) % len(self.condition_options)]
        elif key == curses.KEY_BACKSPACE or key == 127:
            if self.phrase_field == 0:
                step.note = None
        elif key == ord('\n'):
            if self.phrase_field == 0:  # Note
                step.note = 60
        elif key == 27:  # ESC
            self.view = "arrangement"
    
    def run(self, stdscr):
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.nodelay(True)
        stdscr.timeout(20)
        
        while True:
            if self.view == "arrangement":
                self.draw_arrangement(stdscr)
            else:
                self.draw_phrase(stdscr)
            
            key = stdscr.getch()
            
            if key == -1:
                continue
            
            if self.view == "phrase":
                self.handle_phrase_input(key)
                continue
            
            # Arrangement view controls
            if key == curses.KEY_UP:
                self.cursor_row = max(0, self.cursor_row - 1)
            elif key == curses.KEY_DOWN:
                self.cursor_row = min(63, self.cursor_row + 1)
            elif key == curses.KEY_LEFT:
                self.cursor_col = max(0, self.cursor_col - 1)
            elif key == curses.KEY_RIGHT:
                self.cursor_col = min(7, self.cursor_col + 1)
            elif key == curses.KEY_SLEFT:  # Shift+Left
                self.current_phrase_num = max(0, self.current_phrase_num - 1)
                self.arrangement[self.cursor_row][self.cursor_col] = self.current_phrase_num
            elif key == curses.KEY_SRIGHT:  # Shift+Right
                self.current_phrase_num = min(127, self.current_phrase_num + 1)
                self.arrangement[self.cursor_row][self.cursor_col] = self.current_phrase_num
            elif key == ord('\n'):  # Enter
                existing = self.arrangement[self.cursor_row][self.cursor_col]
                if existing is not None:
                    # Edit existing phrase
                    self.current_phrase_num = existing
                    self.view = "phrase"
                    self.phrase_cursor = 0
                    self.phrase_field = 0
                else:
                    # Place current phrase number
                    self.arrangement[self.cursor_row][self.cursor_col] = self.current_phrase_num
            elif key == curses.KEY_BACKSPACE:  
                self.arrangement[self.cursor_row][self.cursor_col] = None
            elif key == ord(' '):
                if self.play_mode == "pattern":
                    # In pattern mode, space starts playing the current row
                    self.start_playback(self.cursor_row)
                else:
                    # In song mode, space toggles play/stop
                    if self.playing:
                        self.stop_playback_func()
                    else:
                        self.start_playback(self.cursor_row)
            elif key == ord('\t'): 
                self.toggle_play_mode()
            elif key == ord('t') or key == ord('T'):
                # Simple tempo adjustment
                stdscr.addstr(0, 0, "Enter tempo (40-300): ")
                curses.echo()
                curses.curs_set(1)
                tempo_str = stdscr.getstr(0, 23, 3).decode('utf-8')
                curses.noecho()
                curses.curs_set(0)
                try:
                    self.tempo = max(40, min(300, int(tempo_str)))
                except:
                    pass
            elif key == ord('q') or key == ord('Q'):
                if self.playing:
                    self.stop_playback_func()
                    if self.playback_thread:
                        self.playback_thread.join(timeout=1.0)
                break
            elif key == 27:  # ESC key
                result = self.show_main_menu(stdscr)
                if result == "quit":
                    if self.playing:
                        self.stop_playback_func()
                        if self.playback_thread:
                            self.playback_thread.join(timeout=1.0)
                    break

def main():
    tracker = MidiTracker()
    curses.wrapper(tracker.run)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Terminal MIDI Phrase Tracker
Requirements: pip install mido python-rtmidi
"""

import curses
import threading
import time
import random
from dataclasses import dataclass, field
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
        if self.midi_out:
            try:
                self.midi_out.send(Message('note_on', channel=channel, note=note, velocity=velocity))
                # Schedule note off after 50ms
                threading.Timer(0.05, lambda: self.midi_out.send(
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
                                self.send_midi(channel, step.note, step.velocity)
                        
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
        headers = "ROW │ CH1  CH2  CH3  CH4  CH5  CH6  CH7  CH8"
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
        
        # Footer with controls
        stdscr.addstr(height - 5, 0, "─" * (width - 1))
        
        if self.play_mode == "pattern":
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase | SHIFT+←→:Change Phrase# | SHIFT+BKSP:Remove",
                "SPACE:Play Row | BKSP:Stop | SHIFT+SPACE:Toggle Mode | T:Tempo | Q:Quit"
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
            "↑↓:Navigate Steps | ←→:Navigate Fields | +/-:Adjust Value | ESC:Back to Arrangement",
            "0-9:Enter Value | BACKSPACE:Clear Note | TAB:Next Condition"
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
        elif key == ord('+') or key == ord('='):
            if self.phrase_field == 0:  # Note
                step.note = min(127, (step.note or 60) + 1)
            elif self.phrase_field == 1:  # Velocity
                step.velocity = min(127, step.velocity + 1)
            elif self.phrase_field == 2:  # Probability
                step.probability = min(100, step.probability + 10)
        elif key == ord('-') or key == ord('_'):
            if self.phrase_field == 0:  # Note
                if step.note is not None:
                    step.note = max(0, step.note - 1)
            elif self.phrase_field == 1:  # Velocity
                step.velocity = max(0, step.velocity - 1)
            elif self.phrase_field == 2:  # Probability
                step.probability = max(0, step.probability - 10)
        elif key == ord('\t'):  # Tab for condition
            if self.phrase_field == 3:
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[(idx + 1) % len(self.condition_options)]
        elif key == curses.KEY_BACKSPACE or key == 127:
            if self.phrase_field == 0:
                step.note = None
        elif key == ord('\n'):
            if self.phrase_field == 0:  # Note
                step.note = 60
        elif ord('0') <= key <= ord('9'):
            # Number input for direct value entry
            digit = key - ord('0')
            if self.phrase_field == 0:  # Note
                step.note = digit if step.note is None else min(127, step.note * 10 + digit)
            elif self.phrase_field == 1:  # Velocity
                step.velocity = min(127, int(str(step.velocity)[-2:] + str(digit)))
            elif self.phrase_field == 2:  # Probability
                step.probability = min(100, int(str(step.probability)[-2:] + str(digit)))
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
        stdscr.timeout(50)
        
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

def main():
    tracker = MidiTracker()
    curses.wrapper(tracker.run)

if __name__ == "__main__":
    main()

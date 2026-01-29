#!/usr/bin/env python3
"""
Terminal MIDI Phrase Tracker
Requirements: pip install mido python-rtmidi blessed
"""

import sys
import threading
import time
import random
from dataclasses import dataclass, field
from typing import Optional, List
import mido
from mido import Message
from blessed import Terminal


@dataclass
class PhraseStep:
    note: Optional[int] = None
    velocity: int = 100
    probability: int = 100
    condition: str = "1/1"


@dataclass
class Phrase:
    length: int = 16
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
    def __init__(self):
        self.term = Terminal()
        self.phrases = {i: Phrase() for i in range(128)}
        self.arrangement = [[None for _ in range(8)] for _ in range(64)]
        self.current_notes = [None] * 8
        self.current_phrase_num = 0
        self.cursor_row = 0
        self.cursor_col = 0
        self.view = "arrangement"
        self.phrase_cursor = 0
        self.phrase_field = 0  # 0=note, 1=vel, 2=prob, 3=cond
        self.phrase_page = 0
        self.phrase_header_field = 0  # 0=length, 1=page
        self.length_options = [16, 32, 48, 64]
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

    # ── MIDI / playback (unchanged) ──────────────────────────

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

    def send_midi(self, channel, note, velocity):
        if self.midi_out:
            try:
                self.midi_out.send(
                    Message("note_on", channel=channel,
                            note=note, velocity=velocity)
                )
                threading.Timer(
                    0.05,
                    lambda: self.midi_out.send(
                        Message("note_off", channel=channel, note=note)
                    ),
                ).start()
            except Exception:
                pass

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
                                self.current_notes[channel] = step.note
                                self.send_midi(
                                    channel, step.note, step.velocity
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
        status = (
            f"PHRASE:{self.current_phrase_num:03d} | "
            f"MODE:{self.play_mode.upper()} | "
            f"{play_status} | TEMPO:{self.tempo} | "
            f"ROW:{self.current_row:02d}"
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
                "SPACE:Play Row | TAB:Toggle Mode"
                ". Stop Playback| T:Tempo | Q:Quit",
            ]
        else:
            controls = [
                "ARROWS:Navigate | ENTER:Edit Phrase "
                "| SHIFT+←→ or []:Change Phrase# | BKSP:Remove",
                "SPACE:Play/Stop Song | TAB:Toggle Mode "
                "| T:Tempo | Q:Quit",
            ]

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

        # Page selector
        page_x = 36
        page_label = "PAGE:"
        page_val = f"{self.phrase_page + 1}/{max_pages}"
        if self.phrase_cursor == -1 and self.phrase_header_field == 1:
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

        buf.append(t.move_xy(0, 2) + t.bold("═" * (w - 1)))

        # Column headers
        headers = "STEP │ NOTE   VEL  PROB%  COND"
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

            # Velocity
            vel_text = f"{step.velocity:3d}"
            if i == self.phrase_cursor and self.phrase_field == 1:
                buf.append(t.move_xy(16, y) + t.reverse(vel_text))
            else:
                buf.append(t.move_xy(16, y) + row_fmt(vel_text))

            # Probability
            prob_text = f"{step.probability:3d}"
            if i == self.phrase_cursor and self.phrase_field == 2:
                buf.append(t.move_xy(22, y) + t.reverse(prob_text))
            else:
                buf.append(t.move_xy(22, y) + row_fmt(prob_text))

            # Condition
            cond_text = f"{step.condition:>4s}"
            if i == self.phrase_cursor and self.phrase_field == 3:
                buf.append(t.move_xy(30, y) + t.reverse(cond_text))
            else:
                buf.append(t.move_xy(30, y) + row_fmt(cond_text))

        # ── footer ──
        buf.append(t.move_xy(0, h - 4) + "─" * (w - 1))
        controls = [
            "↑↓:Navigate Steps | ←→:Navigate Fields "
            "| SHIFT+←→ or []:Adjust Value",
            "BACKSPACE:Clear Note | ESC:Back to Arrangement",
        ]
        for i, ctrl in enumerate(controls):
            buf.append(t.move_xy(2, h - 3 + i) + t.magenta(ctrl))

        self._flush(buf)

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

        # ── header mode (length / page selectors) ──
        if self.phrase_cursor == -1:
            if key.name == "KEY_DOWN":
                self.phrase_cursor = 0
            elif key.name == "KEY_LEFT":
                self.phrase_header_field = max(
                    0, self.phrase_header_field - 1
                )
            elif key.name == "KEY_RIGHT":
                self.phrase_header_field = min(
                    1, self.phrase_header_field + 1
                )
            elif self._is_shift_right(key):
                if self.phrase_header_field == 0:  # Length
                    idx = self.length_options.index(phrase.length)
                    if idx < len(self.length_options) - 1:
                        new_length = self.length_options[idx + 1]
                        self._set_phrase_length(phrase, new_length)
                elif self.phrase_header_field == 1:  # Page
                    new_max = phrase.length // 16
                    if self.phrase_page < new_max - 1:
                        self.phrase_page += 1
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
                elif self.phrase_header_field == 1:  # Page
                    if self.phrase_page > 0:
                        self.phrase_page -= 1
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
            self.phrase_field = min(3, self.phrase_field + 1)
        elif self._is_shift_right(key):
            if self.phrase_field == 0:
                step.note = min(127, (step.note or 60) + 1)
            elif self.phrase_field == 1:
                step.velocity = min(127, step.velocity + 1)
            elif self.phrase_field == 2:
                step.probability = min(100, step.probability + 10)
            elif self.phrase_field == 3:
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[
                    (idx + 1) % len(self.condition_options)
                ]
        elif self._is_shift_left(key):
            if self.phrase_field == 0:
                step.note = max(0, (step.note or 60) - 1)
            elif self.phrase_field == 1:
                step.velocity = max(0, step.velocity - 1)
            elif self.phrase_field == 2:
                step.probability = max(0, step.probability - 10)
            elif self.phrase_field == 3:
                idx = self.condition_options.index(step.condition)
                step.condition = self.condition_options[
                    (idx - 1) % len(self.condition_options)
                ]
        elif self._is_shift_up(key):
            if self.phrase_field == 0:
                step.note = min(127, (step.note or 60) + 12)  # +1 octave
        elif self._is_shift_down(key):
            if self.phrase_field == 0:
                step.note = max(0, (step.note or 60) - 12)
        elif self._is_backspace(key):
            if self.phrase_field == 0:
                step.note = None
        elif key.name == "KEY_ENTER" or key in ("\n", "\r"):
            if self.phrase_field == 0:
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
                else:
                    self.draw_phrase()

                key = t.inkey(timeout=0.02)

                if not key:
                    continue

                if self.view == "phrase":
                    self.handle_phrase_input(key)
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
                        self.phrase_page = 0          # ← add this line
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

                elif key in ("q", "Q"):
                    if self.playing:
                        self.stop_playback_func()
                        if self.playback_thread:
                            self.playback_thread.join(timeout=1.0)
                    break

                elif key.name == "KEY_ESCAPE":
                    selected_port = self.select_midi_port()
                    if selected_port:
                        self.change_midi_port(selected_port)


def main():
    tracker = TRKR()
    tracker.run()


if __name__ == "__main__":
    main()

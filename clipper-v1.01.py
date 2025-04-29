# ==========================================
# clipper-v2.0.py
# Video Player + AI-Based Highlight Detection
# ==========================================

import sys
import os
import whisper
import time
from transformers import pipeline
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QProgressBar, QSlider, QStyle, QMessageBox,
    QLineEdit
)
from PyQt5.QtCore import Qt, QUrl, QTimer, QDir, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from moviepy.editor import VideoFileClip

# Load lightweight text classification model for detecting funny segments
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=2)

class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    live_update = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path

# Inside your TranscriptionWorker class:

    def run(self):
        result = self.model.transcribe(self.audio_path)
        segments = result['segments']
        total_segments = len(segments)

        detected_highlights = []

        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']

            # If segment is long, split into smaller chunks
            if end - start > 10:
                slice_size = 10  # seconds
                num_slices = int((end - start) // slice_size) + 1
                for slice_idx in range(num_slices):
                    slice_start = start + slice_idx * slice_size
                    slice_end = min(slice_start + slice_size, end)
                    slice_text = text  # (basic approach — optional: smarter text splitting later)

                    self.evaluate_segment(slice_start, slice_text, detected_highlights)
            else:
                self.evaluate_segment(start, text, detected_highlights)

            self.live_update.emit(text)
            percent_complete = int((i + 1) / total_segments * 100)
            self.progress.emit(percent_complete)

        self.finished.emit(detected_highlights)

    def evaluate_segment(self, timestamp, text, detected_highlights):
        try:
            prediction = classifier(text)
            top_label = prediction[0][0]['label']
            top_score = prediction[0][0]['score']

            if top_label in ["joy", "surprise"] and top_score > 0.95:
                # Now attach a start and end time
                clip_start = max(0, timestamp - 2)  # back up 2 seconds before detected funny moment
                clip_end = timestamp + 8  # forward 8 seconds after

                detected_highlights.append((clip_start, clip_end, text))
        except Exception:
            pass  # Silently skip bad classifications

class VideoTranscriberEditor(QWidget):
    def __init__(self):
        super().__init__()

        # Load whisper model
        self.model = whisper.load_model("small")
        import torch
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Video editor variables
        self.video_file_path = None
        self.video_clip = None
        self.current_time = 0
        self.clip_start_time = None
        self.clip_end_time = None
        self.is_playing = False

        # Transcription variables
        self.audio_path = None
        self.full_text = ""
        self.pending_segments = []
        self.current_typing_text = ""
        self.current_char_index = 0
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.type_next_character)

        # Detected highlights
        self.highlights = []

        self.setup_window()
        self.setup_media_player()
        self.create_ui_components()
        self.create_layouts()
        self.setup_connections()

        self.apply_dark_theme()

        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)
        self.update_timer.timeout.connect(self.update_playback_position)
    def setup_window(self):
        self.setWindowTitle("Clipper v2.0 - AI Highlight Detection")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(900, 600)

    def setup_media_player(self):
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(360)
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

    def create_ui_components(self):
        # Video player components
        self.load_button = QPushButton("Load Video")

        self.play_button = QPushButton()
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(40, 40)
        self.play_button.setEnabled(False)

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setTracking(True)

        self.time_label = QLabel("00:00:00 / 00:00:00")

        # Clip editing components
        self.start_button = QPushButton("Mark Start")
        self.start_button.setEnabled(False)

        self.end_button = QPushButton("Mark End")
        self.end_button.setEnabled(False)

        self.preview_button = QPushButton("Preview Clip")
        self.preview_button.setEnabled(False)

        self.save_button = QPushButton("Save Clip")
        self.save_button.setEnabled(False)

        self.start_label = QLabel("Manual Start (HH:MM:SS):")
        self.start_entry = QLineEdit()

        self.end_label = QLabel("Manual End (HH:MM:SS):")
        self.end_entry = QLineEdit()

        self.apply_manual_button = QPushButton("Apply Manual Times")
        self.apply_manual_button.setEnabled(False)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.setTracking(True)

        # Transcription components
        self.transcribe_button = QPushButton("Transcribe + Detect Highlights")
        self.transcribe_button.setEnabled(False)

        self.save_transcript_button = QPushButton("Save Transcript")
        self.save_transcript_button.setEnabled(False)

        self.result_textbox = QTextEdit()
        self.result_textbox.setReadOnly(True)
        font = QFont()
        font.setPointSize(11)
        self.result_textbox.setFont(font)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Highlights list
        self.highlights_label = QLabel("AI-DETECTED HIGHLIGHTS LIST")
        self.highlights_textbox = QTextEdit()
        self.highlights_textbox.setReadOnly(True)
        self.highlights_textbox.setFont(font)
        self.highlights_textbox.setMaximumHeight(200)

        # Action buttons for highlights
        self.cut_clip_button = QPushButton("CUT CLIP")
        self.cut_clip_button.setEnabled(False)
        self.save_clip_button = QPushButton("SAVE CLIP")
        self.save_clip_button.setEnabled(False)
        self.reject_button = QPushButton("REJECT")
        self.reject_button.setEnabled(False)

        # Status label
        self.status_label = QLabel("Ready to load video")

    def create_layouts(self):
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_widget)

        playback_layout = QHBoxLayout()
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.time_label)
        playback_layout.addStretch(1)
        playback_layout.addWidget(QLabel("Volume:"))
        playback_layout.addWidget(self.volume_slider)

        slider_layout = QVBoxLayout()
        slider_layout.addLayout(playback_layout)
        slider_layout.addWidget(self.timeline_slider)

        video_playback_container = QVBoxLayout()
        video_playback_container.addLayout(video_layout, 3)
        video_playback_container.addLayout(slider_layout)

        edit_container = QVBoxLayout()

        edit_container.addWidget(self.load_button)

        clip_buttons_layout = QHBoxLayout()
        clip_buttons_layout.addWidget(self.start_button)
        clip_buttons_layout.addWidget(self.end_button)
        edit_container.addLayout(clip_buttons_layout)

        clip_actions_layout = QHBoxLayout()
        clip_actions_layout.addWidget(self.preview_button)
        clip_actions_layout.addWidget(self.save_button)
        edit_container.addLayout(clip_actions_layout)

        manual_start_layout = QHBoxLayout()
        manual_start_layout.addWidget(self.start_label)
        manual_start_layout.addWidget(self.start_entry)
        edit_container.addLayout(manual_start_layout)

        manual_end_layout = QHBoxLayout()
        manual_end_layout.addWidget(self.end_label)
        manual_end_layout.addWidget(self.end_entry)
        edit_container.addLayout(manual_end_layout)

        edit_container.addWidget(self.apply_manual_button)

        edit_container.addSpacing(10)

        edit_container.addWidget(QLabel("Transcription & AI Detection"))

        transcription_buttons = QHBoxLayout()
        transcription_buttons.addWidget(self.transcribe_button)
        transcription_buttons.addWidget(self.save_transcript_button)
        edit_container.addLayout(transcription_buttons)

        edit_container.addWidget(self.progress_bar)
        edit_container.addWidget(self.result_textbox)

        edit_container.addWidget(self.highlights_label)
        edit_container.addWidget(self.highlights_textbox)

        highlight_buttons = QHBoxLayout()
        highlight_buttons.addWidget(self.cut_clip_button)
        highlight_buttons.addWidget(self.save_clip_button)
        highlight_buttons.addWidget(self.reject_button)
        edit_container.addLayout(highlight_buttons)

        edit_container.addStretch(1)
        edit_container.addWidget(self.status_label)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_playback_container, 7)
        main_layout.addLayout(edit_container, 3)

        self.setLayout(main_layout)

    def setup_connections(self):
        self.load_button.clicked.connect(self.load_video)
        self.play_button.clicked.connect(self.toggle_play)
        self.timeline_slider.sliderMoved.connect(self.seek_position)
        self.volume_slider.valueChanged.connect(self.change_volume)

        self.start_button.clicked.connect(self.mark_start)
        self.end_button.clicked.connect(self.mark_end)
        self.preview_button.clicked.connect(self.preview_clip)
        self.save_button.clicked.connect(self.save_clip)
        self.apply_manual_button.clicked.connect(self.set_manual_times)

        self.transcribe_button.clicked.connect(self.transcribe_video)
        self.save_transcript_button.clicked.connect(self.save_transcript)

        self.cut_clip_button.clicked.connect(self.handle_highlight_cut)
        self.save_clip_button.clicked.connect(self.handle_highlight_save)
        self.reject_button.clicked.connect(self.handle_highlight_reject)

        self.highlights_textbox.mouseDoubleClickEvent = self.highlight_double_clicked

    def apply_dark_theme(self):
        flux_stylesheet = """
        QWidget {
            background-color: #2b1d0e; /* dark brown */
            color: #ffae42; /* soft orange text */
        }
        QPushButton {
            background-color: #3c2a17; /* slightly lighter brown */
            color: #ffae42;
            border: 1px solid #5c3b1c;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #5c3b1c; /* hover lighter brown */
        }
        QPushButton:disabled {
            background-color: #2b1d0e;
            color: #7f5a2e;
        }
        QLineEdit, QTextEdit {
            background-color: #3c2a17;
            color: #ffae42;
            border: 1px solid #5c3b1c;
            border-radius: 5px;
        }
        QProgressBar {
            background-color: #3c2a17;
            color: #ffae42;
            border: 1px solid #5c3b1c;
            border-radius: 5px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #ffae42;
            width: 10px;
        }
        QSlider::groove:horizontal {
            border: 1px solid #5c3b1c;
            height: 8px;
            background: #3c2a17;
        }
        QSlider::handle:horizontal {
            background: #ffae42;
            border: 1px solid #5c3b1c;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QLabel {
            color: #ffae42;
        }
        """
        self.setStyleSheet(flux_stylesheet)

    # ===== VIDEO PLAYER FUNCTIONS =====
    def load_video(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Video Files (*.mp4 *.mov *.avi *.mkv)")
        if file_dialog.exec_():
            filepath = file_dialog.selectedFiles()[0]
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "Error", "Selected file does not exist.")
                return

            self.video_file_path = filepath
            self.audio_path = filepath  # Using same file for transcription

            try:
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
                self.video_clip = VideoFileClip(filepath)
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load video: {str(e)}")
                return

            self.status_label.setText(f"Loaded: {os.path.basename(filepath)}")

            self.enable_controls(True)
            self.clip_start_time = None
            self.clip_end_time = None
            self.start_entry.clear()
            self.end_entry.clear()
            self.result_textbox.clear()
            self.highlights_textbox.clear()
            self.highlights.clear()

            self.transcribe_button.setEnabled(True)

    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.pause_icon)
            self.is_playing = True
            self.update_timer.start()
        else:
            self.play_button.setIcon(self.play_icon)
            self.is_playing = False
            self.update_timer.stop()

    def duration_changed(self, duration):
        duration_sec = duration / 1000
        self.timeline_slider.setRange(0, duration)
        self.time_label.setText(f"00:00:00 / {self.format_time(duration_sec)}")

    def update_playback_position(self):
        if self.is_playing:
            position = self.media_player.position()
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(position)
            self.timeline_slider.blockSignals(False)
            current_sec = position / 1000
            duration_sec = self.media_player.duration() / 1000
            self.time_label.setText(f"{self.format_time(current_sec)} / {self.format_time(duration_sec)}")
            self.current_time = current_sec

    def seek_position(self, position):
        self.media_player.setPosition(position)
        self.current_time = position / 1000
        duration_sec = self.media_player.duration() / 1000
        self.time_label.setText(f"{self.format_time(self.current_time)} / {self.format_time(duration_sec)}")

    def change_volume(self, value):
        self.media_player.setVolume(value)

    # ===== CLIP EDITOR FUNCTIONS =====
    def mark_start(self):
        self.clip_start_time = self.current_time
        self.start_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"Start marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()

    def mark_end(self):
        self.clip_end_time = self.current_time
        self.end_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"End marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()

    def set_manual_times(self):
        try:
            start_time = self.parse_time_string(self.start_entry.text())
            end_time = self.parse_time_string(self.end_entry.text())
            if start_time >= 0 and end_time > start_time and end_time <= self.video_clip.duration:
                self.clip_start_time = start_time
                self.clip_end_time = end_time
                self.status_label.setText(f"Manual times set: {self.format_time(start_time)} to {self.format_time(end_time)}")
                self.update_clip_controls()
            else:
                raise ValueError("Invalid time range")
        except ValueError:
            QMessageBox.warning(self, "Invalid Time", "Please enter valid times in HH:MM:SS format.")

    def preview_clip(self):
        if self.validate_clip_times():
            try:
                subclip = self.video_clip.subclip(self.clip_start_time, self.clip_end_time)
                subclip.preview()
            except Exception as e:
                QMessageBox.critical(self, "Preview Error", f"Failed to preview clip: {str(e)}")

    def save_clip(self):
        if self.validate_clip_times():
            try:
                original_filename = os.path.basename(self.video_file_path)
                name_without_ext = os.path.splitext(original_filename)[0]
                default_output = f"{name_without_ext}_clip_{self.format_time(self.clip_start_time)}-{self.format_time(self.clip_end_time)}.mp4"
                output_path, _ = QFileDialog.getSaveFileName(self, "Save Clip", os.path.join(QDir.homePath(), default_output), "Video Files (*.mp4)")
                if output_path:
                    self.status_label.setText("Saving clip... Please wait")
                    QApplication.processEvents()
                    subclip = self.video_clip.subclip(self.clip_start_time, self.clip_end_time)
                    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='medium', threads=4)
                    self.status_label.setText(f"Clip saved to: {os.path.basename(output_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving clip: {str(e)}")

    def validate_clip_times(self):
        if self.clip_start_time is None or self.clip_end_time is None:
            QMessageBox.warning(self, "Missing Time Markers", "Please mark both start and end times.")
            return False
        if self.clip_end_time <= self.clip_start_time:
            QMessageBox.warning(self, "Invalid Time Range", "End time must be after start time.")
            return False
        if self.clip_start_time < 0 or self.clip_end_time > self.video_clip.duration:
            QMessageBox.warning(self, "Out of Range", "Clip times must be within video duration.")
            return False
        return True
    # ===== TRANSCRIPTION + AI DETECTION =====
    def transcribe_video(self):
        if not self.audio_path:
            QMessageBox.warning(self, "No Video", "Please load a video before transcribing.")
            return

        self.status_label.setText("Transcribing and analyzing... please wait.")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_textbox.clear()
        self.highlights_textbox.clear()
        self.highlights.clear()
        QApplication.processEvents()

        self.full_text = ""
        self.pending_segments.clear()

        self.worker = TranscriptionWorker(self.model, self.audio_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.live_update.connect(self.animate_typing)
        self.worker.finished.connect(self.handle_transcription_finished)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def animate_typing(self, new_text):
        if self.typing_timer.isActive():
            self.pending_segments.append(new_text)
        else:
            self.current_typing_text = new_text
            self.current_char_index = 0
            self.typing_timer.start(20)

    def type_next_character(self):
        if self.current_char_index < len(self.current_typing_text):
            self.full_text += self.current_typing_text[self.current_char_index]
            self.result_textbox.setPlainText(self.full_text)
            self.result_textbox.moveCursor(QTextCursor.End)
            self.current_char_index += 1
        else:
            self.typing_timer.stop()
            if self.pending_segments:
                next_text = self.pending_segments.pop(0)
                self.current_typing_text = next_text
                self.current_char_index = 0
                self.typing_timer.start(20)

    def handle_transcription_finished(self, detected_highlights):
        self.status_label.setText("Transcription and AI detection complete.")
        self.progress_bar.setVisible(False)
        self.save_transcript_button.setEnabled(True)

        self.highlights = detected_highlights
        self.update_highlights_display()

        highlight_buttons_enabled = len(self.highlights) > 0
        self.cut_clip_button.setEnabled(highlight_buttons_enabled)
        self.save_clip_button.setEnabled(highlight_buttons_enabled)
        self.reject_button.setEnabled(highlight_buttons_enabled)

    def save_transcript(self):
        if not self.full_text:
            QMessageBox.warning(self, "No Transcript", "There is no transcription to save.")
            return

        file_dialog = QFileDialog(self)
        save_path, _ = file_dialog.getSaveFileName(self, "Save Transcript", "", "Text Files (*.txt)")
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.full_text)
                self.status_label.setText(f"Transcript saved: {os.path.basename(save_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save transcript: {str(e)}")

    # ===== HIGHLIGHTS MANAGEMENT =====
    def update_highlights_display(self):
        self.highlights_textbox.clear()
        highlight_text = ""

        for i, (start_time, end_time, text) in enumerate(self.highlights):
            formatted_start = self.format_time(start_time)
            formatted_end = self.format_time(end_time)
            highlight_text += f"[{formatted_start} - {formatted_end}] {text}\n\n"

        self.highlights_textbox.setPlainText(highlight_text)

    def highlight_double_clicked(self, event):
        cursor = self.highlights_textbox.cursorForPosition(event.pos())
        cursor.select(QTextCursor.LineUnderCursor)
        line = cursor.selectedText()

        if line and "[" in line and "]" in line:
            time_range = line[line.find("[")+1:line.find("]")]
            if " - " in time_range:
                start_str, end_str = time_range.split(" - ")
                try:
                    start_seconds = self.parse_time_string(start_str)
                    self.media_player.setPosition(int(start_seconds * 1000))
                    self.seek_position(int(start_seconds * 1000))
                except Exception:
                    pass

        super(QTextEdit, self.highlights_textbox).mouseDoubleClickEvent(event)

    def handle_highlight_cut(self):
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()

        if selected_line and "[" in selected_line:
            try:
                time_str = selected_line[selected_line.find("[")+1:selected_line.find("]")]
                timestamp = self.parse_time_string(time_str)

                self.clip_start_time = max(0, timestamp - 2)
                self.clip_end_time = min(self.video_clip.duration, timestamp + 8)

                self.start_entry.setText(self.format_time(self.clip_start_time))
                self.end_entry.setText(self.format_time(self.clip_end_time))
                self.update_clip_controls()

                self.media_player.setPosition(int(self.clip_start_time * 1000))
                self.seek_position(int(self.clip_start_time * 1000))

                self.status_label.setText(f"Clip set: {self.format_time(self.clip_start_time)} to {self.format_time(self.clip_end_time)}")
            except Exception:
                self.status_label.setText("Could not parse highlight timestamp.")

    def handle_highlight_save(self):
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()

        if selected_line and "[" in selected_line:
            try:
                time_str = selected_line[selected_line.find("[")+1:selected_line.find("]")]
                timestamp = self.parse_time_string(time_str)

                self.clip_start_time = max(0, timestamp - 2)
                self.clip_end_time = min(self.video_clip.duration, timestamp + 8)

                if self.validate_clip_times():
                    self.save_clip()
            except Exception:
                self.status_label.setText("Could not parse highlight timestamp.")

    def handle_highlight_reject(self):
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        line_number = cursor.blockNumber()

        if 0 <= line_number < len(self.highlights):
            del self.highlights[line_number]
            self.update_highlights_display()
            self.status_label.setText("Highlight rejected.")

    # ===== UTILITY FUNCTIONS =====
    def handle_error(self, error):
        self.status_label.setText(f"Media Player Error: {error}")
        QMessageBox.critical(self, "Media Error", f"Error code: {error}")

    def enable_controls(self, enabled):
        self.play_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.apply_manual_button.setEnabled(enabled)
        self.transcribe_button.setEnabled(enabled)

    def update_clip_controls(self):
        has_clip_times = (
            self.clip_start_time is not None and
            self.clip_end_time is not None and
            self.clip_end_time > self.clip_start_time
        )
        self.preview_button.setEnabled(has_clip_times)
        self.save_button.setEnabled(has_clip_times)

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def parse_time_string(self, time_str):
        if not time_str:
            return 0
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        else:
            raise ValueError("Invalid time format")
        return hours * 3600 + minutes * 60 + seconds
def main():
    app = QApplication(sys.argv)
    editor = VideoTranscriberEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

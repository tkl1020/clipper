# ==========================================
# clipper-v2.0.py
# Video Player + AI-Based Highlight Detection
# ==========================================

# These are 'import' statements - they bring in code libraries that other developers have written
# so we can use their functionality without having to write everything from scratch
import sys                # Provides access to system-specific parameters and functions
import os                 # Allows interaction with the operating system (files, directories, etc.)
import whisper            # An AI library for transcribing speech from audio to text
import time               # Provides various time-related functions
from transformers import pipeline  # AI-based text processing library

# PyQt5 imports - these are for creating the graphical user interface (GUI)
# QApplication is the core of any PyQt application
# Various widgets like buttons, text boxes, layouts are imported to build the interface
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QProgressBar, QSlider, QStyle, QMessageBox,
    QLineEdit
)
# More PyQt imports for handling core functionality and timing
from PyQt5.QtCore import Qt, QUrl, QTimer, QDir, QThread, pyqtSignal
# PyQt imports for text formatting and display
from PyQt5.QtGui import QTextCursor, QFont
# PyQt libraries for multimedia handling (playing videos)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
# MoviePy library for video editing capabilities
from moviepy.editor import VideoFileClip

# Load an AI text classification model that will detect emotions in text
# This will be used to find potentially interesting/funny moments in videos
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=2)

# This class creates a separate thread for running the transcription process
# Using a separate thread prevents the application from freezing while processing
class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    live_update = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path

    def run(self):
        result = self.model.transcribe(self.audio_path)
        segments = result['segments']
        total_segments = len(segments)

        detected_highlights = []
        batch_texts = []
        batch_starts = []
        batch_ends = []

        last_highlight_end = 0  # to prevent overlaps

        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            start = segment['start']
            end = segment['end']

            # Add to batch
            batch_texts.append(text)
            batch_starts.append(start)
            batch_ends.append(end)

            self.live_update.emit(text)

            # Every 10 or last, evaluate
            if len(batch_texts) >= 10 or i == total_segments - 1:
                predictions = classifier(batch_texts)

                for idx, preds in enumerate(predictions):
                    top_label = preds[0]['label']
                    top_score = preds[0]['score']
                    start_time = batch_starts[idx]
                    end_time = batch_ends[idx]
                    clip_duration = end_time - start_time
                    clip_text = batch_texts[idx]

                    if top_label in ["joy", "surprise"] and top_score > 0.98:
                        # Enforce clip between 20 and 120 seconds
                        if clip_duration < 20:
                            clip_end = start_time + 20
                        elif clip_duration > 120:
                            clip_end = start_time + 120
                        else:
                            clip_end = end_time

                        clip_start = max(0, start_time - 2)  # Small buffer before
                        clip_end = min(clip_start + (clip_end - start_time) + 2, clip_end + 2)  # Extend end by 2s max

                        if clip_start >= last_highlight_end:
                            # Generate title
                            title = self.generate_title(clip_text)
                            detected_highlights.append((
                                clip_start,
                                clip_end,
                                title
                            ))
                            last_highlight_end = clip_end

                batch_texts.clear()
                batch_starts.clear()
                batch_ends.clear()

            percent_complete = int((i + 1) / total_segments * 100)
            self.progress.emit(percent_complete)

        self.finished.emit(detected_highlights)

    def generate_title(self, text):
        words = text.split()
        if len(words) >= 6:
            return ' '.join(words[:6]) + "..."
        else:
            return ' '.join(words)

# This is the main application class that creates the video editor interface
class VideoTranscriberEditor(QWidget):
    # This is the constructor - it sets up everything when the application starts
    def __init__(self):
        super().__init__()  # Initialize the parent class (QWidget)

        # Load the speech recognition model (whisper)
        # "small" is the model size - smaller is faster but less accurate
        self.model = whisper.load_model("tiny")
        import torch
        # If a GPU is available, use it for faster processing
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Initialize variables for the video editor
        self.video_file_path = None      # Path to the video file
        self.video_clip = None           # The actual video content
        self.current_time = 0            # Current playback position in seconds
        self.clip_start_time = None      # Where to start cutting a clip
        self.clip_end_time = None        # Where to end cutting a clip
        self.is_playing = False          # Whether the video is currently playing

        # Initialize variables for transcription
        self.audio_path = None           # Path to the audio file
        self.full_text = ""              # Complete transcription text
        self.pending_segments = []       # Text segments waiting to be displayed
        self.current_typing_text = ""    # Text currently being "typed" onto the screen
        self.current_char_index = 0      # Position in the current text being typed
        # Timer to control the typing animation
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.type_next_character)

        # List to store detected highlight moments
        self.highlights = []

        # Call methods to set up the application
        self.setup_window()              # Configure the main window
        self.setup_media_player()        # Set up the video player
        self.create_ui_components()      # Create buttons, sliders, etc.
        self.create_layouts()            # Arrange the UI elements
        self.setup_connections()         # Connect buttons to functions

        self.apply_dark_theme()          # Apply a nice dark color scheme

        # Timer to update the playback position display
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(100)  # Update every 100 milliseconds
        self.update_timer.timeout.connect(self.update_playback_position)

    # Set up the main application window properties
    def setup_window(self):
        self.setWindowTitle("Clipper v2.0 - AI Highlight Detection")
        self.setGeometry(100, 100, 1200, 800)  # (x, y, width, height)
        self.setMinimumSize(900, 600)          # Minimum allowed window size

    # Set up the video player component
    def setup_media_player(self):
        # Create a widget to display the video
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(360)
        # Create the media player that will handle the video playback
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)
        # Connect media player events to our functions
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

    # Create all UI elements (buttons, text boxes, etc.)
    def create_ui_components(self):
        # Video player components
        self.load_button = QPushButton("Load Video")

        # Play/pause button with icon
        self.play_button = QPushButton()
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
        self.play_button.setIcon(self.play_icon)
        self.play_button.setFixedSize(40, 40)
        self.play_button.setEnabled(False)  # Disabled until a video is loaded

        # Slider for navigating through the video
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setTracking(True)

        # Label to show current playback time and total duration
        self.time_label = QLabel("00:00:00 / 00:00:00")

        # Clip editing buttons
        self.start_button = QPushButton("Mark Start")
        self.start_button.setEnabled(False)

        self.end_button = QPushButton("Mark End")
        self.end_button.setEnabled(False)

        # Save Clip button (with floppy icon)
        self.save_button = QPushButton()
        self.save_button.setText("Save Clip")
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_button.setEnabled(False)

        # Manual time entry for precise clip control
        self.start_label = QLabel("Manual Start (HH:MM:SS):")
        self.start_entry = QLineEdit()

        self.end_label = QLabel("Manual End (HH:MM:SS):")
        self.end_entry = QLineEdit()

        self.apply_manual_button = QPushButton("Apply Manual Times")
        self.apply_manual_button.setEnabled(False)

        # Volume control slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)  # Start at 50% volume
        self.volume_slider.setTracking(True)

        # Transcription components
        self.transcribe_button = QPushButton("Transcribe + Detect Highlights")
        self.transcribe_button.setEnabled(False)

        # Save Transcript button (small floppy icon)
        self.save_transcript_button = QPushButton()
        self.save_transcript_button.setText("Transcript")
        self.save_transcript_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_transcript_button.setEnabled(False)
        self.save_transcript_button.setFixedSize(120, 40)

        # Progress bar for transcription status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hidden until transcription starts

        # Text display for transcription results
        self.result_textbox = QTextEdit()
        self.result_textbox.setReadOnly(True)
        font = QFont()
        font.setPointSize(10)
        self.result_textbox.setFont(font)

        # Highlights section
        self.highlights_label = QLabel("AI-DETECTED HIGHLIGHTS LIST")
        font = QFont()
        font.setPointSize(11)
        self.highlights_textbox = QTextEdit()
        self.highlights_textbox.setReadOnly(True)
        self.highlights_textbox.setFont(font)
        self.highlights_textbox.setMinimumHeight(400)  # <-- Make highlights window bigger

        # Buttons for handling highlights
        self.cut_clip_button = QPushButton("CUT CLIP")
        self.cut_clip_button.setEnabled(False)

        self.save_clip_button = QPushButton("SAVE CLIP")
        self.save_clip_button.setEnabled(False)

        self.reject_button = QPushButton("REJECT")
        self.reject_button.setEnabled(False)

        # Status label to show information messages
        self.status_label = QLabel("Ready to load video")

    # Arrange all UI components into layouts
    def create_layouts(self):
        # Main layout
        main_layout = QVBoxLayout()
        
        # Top section with video player
        video_container = QVBoxLayout()
        video_container.addWidget(self.video_widget)
        
        # Video controls layout
        video_controls = QHBoxLayout()
        video_controls.addWidget(self.load_button)
        video_controls.addWidget(self.play_button)
        video_controls.addWidget(self.timeline_slider)
        video_controls.addWidget(self.time_label)
        
        # Add video controls to video container
        video_container.addLayout(video_controls)
        
        # Clip editing controls
        clip_controls = QHBoxLayout()
        clip_controls.addWidget(self.start_button)
        clip_controls.addWidget(self.end_button)
        clip_controls.addWidget(self.save_button)
        clip_controls.addStretch()
        clip_controls.addWidget(QLabel("Volume:"))
        clip_controls.addWidget(self.volume_slider)
        
        # Manual time entry section
        manual_time_layout = QHBoxLayout()
        manual_time_layout.addWidget(self.start_label)
        manual_time_layout.addWidget(self.start_entry)
        manual_time_layout.addWidget(self.end_label)
        manual_time_layout.addWidget(self.end_entry)
        manual_time_layout.addWidget(self.apply_manual_button)
        
        # Transcription control section
        transcription_controls = QHBoxLayout()
        transcription_controls.addWidget(self.transcribe_button)
        transcription_controls.addWidget(self.progress_bar)
        transcription_controls.addStretch()
        transcription_controls.addWidget(self.save_transcript_button)
        
        # Bottom section with two columns
        bottom_section = QHBoxLayout()
        
        # Left column - Transcription results
        left_column = QVBoxLayout()
        left_column.addWidget(QLabel("TRANSCRIPTION"))
        left_column.addWidget(self.result_textbox)
        
        # Right column - Highlights
        right_column = QVBoxLayout()
        right_column.addWidget(self.highlights_label)
        right_column.addWidget(self.highlights_textbox)
        
        # Highlight action buttons
        highlight_buttons = QHBoxLayout()
        highlight_buttons.addWidget(self.cut_clip_button)
        highlight_buttons.addWidget(self.save_clip_button)
        highlight_buttons.addWidget(self.reject_button)
        right_column.addLayout(highlight_buttons)
        
        # Add columns to bottom section
        bottom_section.addLayout(left_column)
        bottom_section.addLayout(right_column)
        
        # Status bar at the bottom
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        
        # Add everything to main layout
        main_layout.addLayout(video_container)
        main_layout.addLayout(clip_controls)
        main_layout.addLayout(manual_time_layout)
        main_layout.addLayout(transcription_controls)
        main_layout.addLayout(bottom_section)
        main_layout.addLayout(status_layout)
        
        # Set the main layout
        self.setLayout(main_layout)

    # Connect buttons to their functions
    def setup_connections(self):
        # Video loading and control
        self.load_button.clicked.connect(self.load_media)
        self.play_button.clicked.connect(self.toggle_play)
        self.timeline_slider.sliderMoved.connect(self.seek_position)
        self.volume_slider.valueChanged.connect(self.change_volume)

        # Clip editing
        self.start_button.clicked.connect(self.mark_start)
        self.end_button.clicked.connect(self.mark_end)
        self.save_button.clicked.connect(self.save_clip)
        self.apply_manual_button.clicked.connect(self.set_manual_times)

        # Transcription
        self.transcribe_button.clicked.connect(self.transcribe_video)
        self.save_transcript_button.clicked.connect(self.save_transcript)

        # Highlight management
        self.cut_clip_button.clicked.connect(self.handle_highlight_cut)
        self.save_clip_button.clicked.connect(self.handle_highlight_save)
        self.reject_button.clicked.connect(self.handle_highlight_reject)

        # Special event for double-clicking on highlights
        self.highlights_textbox.mouseDoubleClickEvent = self.highlight_double_clicked


    # Apply a dark color theme to the application
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
        self.setStyleSheet(flux_stylesheet)  # Apply the style to the application


    # ===== VIDEO PLAYER FUNCTIONS =====
    # Function to load a video file
    def load_media(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Media Files (*.mp4 *.mov *.avi *.mkv *.mp3 *.wav)")
        if file_dialog.exec_():
            filepath = file_dialog.selectedFiles()[0]
            if not os.path.exists(filepath):
                QMessageBox.critical(self, "Error", "Selected file does not exist.")
                return

            self.video_file_path = filepath
            self.audio_path = filepath

            # Determine if it's audio or video based on file extension
            audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
            ext = os.path.splitext(filepath)[1].lower()

            try:
                if ext in audio_extensions:
                    # Load audio file
                    self.is_audio_only = True
                    self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
                    self.video_clip = None
                    self.video_widget.hide()  # Hide the video widget for audio files
                else:
                    # Load video file
                    self.is_audio_only = False
                    self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filepath)))
                    self.video_clip = VideoFileClip(filepath)
                    self.video_widget.show()  # Show the video widget for video files
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Failed to load media: {str(e)}")
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


    # Toggle between play and pause
    def toggle_play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()  # Pause if playing
        else:
            self.media_player.play()   # Play if paused


    # Handle state changes in the media player
    def media_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.pause_icon)  # Show pause icon
            self.is_playing = True
            self.update_timer.start()  # Start the timer that updates time display
        else:
            self.play_button.setIcon(self.play_icon)  # Show play icon
            self.is_playing = False
            self.update_timer.stop()   # Stop the update timer


    # Handle when the video duration is determined
    def duration_changed(self, duration):
        duration_sec = duration / 1000  # Convert milliseconds to seconds
        self.timeline_slider.setRange(0, duration)  # Set the slider range
        self.time_label.setText(f"00:00:00 / {self.format_time(duration_sec)}")


    # Update the playback position display while video is playing
    def update_playback_position(self):
        if self.is_playing:
            position = self.media_player.position()  # Get current position in ms
            # Update the slider without triggering signals
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(position)
            self.timeline_slider.blockSignals(False)
            current_sec = position / 1000  # Convert to seconds
            duration_sec = self.media_player.duration() / 1000
            # Update the time display
            self.time_label.setText(f"{self.format_time(current_sec)} / {self.format_time(duration_sec)}")
            self.current_time = current_sec


    # Jump to a position in the video when slider is moved
    def seek_position(self, position):
        self.media_player.setPosition(position)
        self.current_time = position / 1000
        duration_sec = self.media_player.duration() / 1000
        self.time_label.setText(f"{self.format_time(self.current_time)} / {self.format_time(duration_sec)}")


    # Adjust volume when slider is moved
    def change_volume(self, value):
        self.media_player.setVolume(value)


    # ===== CLIP EDITOR FUNCTIONS =====
    # Mark the current position as the start of a clip
    def mark_start(self):
        self.clip_start_time = self.current_time
        self.start_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"Start marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()


    # Mark the current position as the end of a clip
    def mark_end(self):
        self.clip_end_time = self.current_time
        self.end_entry.setText(self.format_time(self.current_time))
        self.status_label.setText(f"End marked at {self.format_time(self.current_time)}")
        self.update_clip_controls()


    # Set clip times manually from text input
    def set_manual_times(self):
        try:
            # Convert the text entries to seconds
            start_time = self.parse_time_string(self.start_entry.text())
            end_time = self.parse_time_string(self.end_entry.text())
            # Check if times are valid
            if start_time >= 0 and end_time > start_time and end_time <= self.video_clip.duration:
                self.clip_start_time = start_time
                self.clip_end_time = end_time
                self.status_label.setText(f"Manual times set: {self.format_time(start_time)} to {self.format_time(end_time)}")
                self.update_clip_controls()
            else:
                raise ValueError("Invalid time range")
        except ValueError:
            QMessageBox.warning(self, "Invalid Time", "Please enter valid times in HH:MM:SS format.")


    # Save the selected clip as a new video file
    def save_clip(self):
        if self.validate_clip_times():
            try:
                if self.is_audio_only:
                    QMessageBox.warning(self, "Audio Only", "Saving clips is only supported for videos right now.")
                    return

                original_filename = os.path.basename(self.video_file_path)
                name_without_ext = os.path.splitext(original_filename)[0]
                default_output = f"{name_without_ext}_clip_{self.format_time(self.clip_start_time)}-{self.format_time(self.clip_end_time)}.mp4"

                output_path, _ = QFileDialog.getSaveFileName(self, "Save Clip", os.path.join(QDir.homePath(), default_output), "Video Files (*.mp4)")
                if output_path:
                    self.status_label.setText("Saving clip... Please wait")
                    QApplication.processEvents()

                    subclip = self.video_clip.subclip(self.clip_start_time, self.clip_end_time)
                    subclip.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='medium', threads=4)
                    self.status_label.setText(f"Clip saved: {os.path.basename(output_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save clip: {str(e)}")


    # Check if clip times are valid
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
    # Start the transcription and highlight detection process
    def transcribe_video(self):
        if not self.audio_path:
            QMessageBox.warning(self, "No Video", "Please load a video before transcribing.")
            return

        # Update UI to show processing is starting
        self.status_label.setText("Transcribing and analyzing... please wait.")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.result_textbox.clear()
        self.highlights_textbox.clear()
        self.highlights.clear()
        QApplication.processEvents()  # Update the UI immediately

        # Reset text variables
        self.full_text = ""
        self.pending_segments.clear()

        # Create and start the worker thread for transcription
        self.worker = TranscriptionWorker(self.model, self.audio_path)
        self.worker.progress.connect(self.update_progress)
        self.worker.live_update.connect(self.animate_typing)
        self.worker.finished.connect(self.handle_transcription_finished)
        self.worker.start()


    # Update the progress bar when transcription advances
    def update_progress(self, value):
        self.progress_bar.setValue(value)


    # Create a typing animation for new text segments
    def animate_typing(self, new_text):
        if self.typing_timer.isActive():
            # If already typing, add this to queue
            self.pending_segments.append(new_text)
        else:
            # Start typing this text
            self.current_typing_text = new_text
            self.current_char_index = 0
            self.typing_timer.start(20)  # Type a character every 20ms


    # Type one character at a time for a more natural appearance
    def type_next_character(self):
        if self.current_char_index < len(self.current_typing_text):
            # Add the next character to the full text
            self.full_text += self.current_typing_text[self.current_char_index]
            self.result_textbox.setPlainText(self.full_text)
            self.result_textbox.moveCursor(QTextCursor.End)  # Scroll to end
            self.current_char_index += 1
        else:
            # Finished typing this segment
            self.typing_timer.stop()
            if self.pending_segments:
                # Start typing the next segment if any
                next_text = self.pending_segments.pop(0)
                self.current_typing_text = next_text
                self.current_char_index = 0
                self.typing_timer.start(20)

    # Handle when the transcription and highlight detection is finished
    def handle_transcription_finished(self, detected_highlights):
        self.highlights = detected_highlights
        self.status_label.setText(f"Transcription complete! {len(detected_highlights)} highlights detected.")
        self.progress_bar.setVisible(False)
        self.save_transcript_button.setEnabled(True)
        
        # Display highlights in the highlights text box
        if detected_highlights:
            self.display_highlights()
            self.cut_clip_button.setEnabled(True)
            self.save_clip_button.setEnabled(True)
            self.reject_button.setEnabled(True)
        else:
            self.highlights_textbox.setPlainText("No highlights detected in this video.")
            
        # Complete typing any remaining text
        if self.pending_segments:
            for segment in self.pending_segments:
                self.full_text += segment
            self.result_textbox.setPlainText(self.full_text)
            self.result_textbox.moveCursor(QTextCursor.End)
            self.pending_segments.clear()

    # Display highlights in the highlights text box
    def display_highlights(self):
        self.highlights_textbox.clear()
        for i, (start, end, title) in enumerate(self.highlights):
            highlight_text = f"Highlight #{i+1}: {self.format_time(start)} to {self.format_time(end)}\n"
            highlight_text += f"Title: {title}\n"
            highlight_text += "-" * 50 + "\n"
            self.highlights_textbox.append(highlight_text)

    # Save the full transcript to a text file
    def save_transcript(self):
        if not self.full_text:
            QMessageBox.warning(self, "No Transcript", "There is no transcript to save.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Transcript", 
            os.path.join(QDir.homePath(), f"{os.path.splitext(os.path.basename(self.video_file_path))[0]}_transcript.txt"),
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.full_text)
                self.status_label.setText(f"Transcript saved to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save transcript: {str(e)}")

    # Handle double-click on a highlight entry
    def highlight_double_clicked(self, event):
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()
        
        # Check if the selected line contains a highlight time
        if "Highlight #" in selected_line:
            try:
                # Extract highlight number
                highlight_num = int(selected_line.split("Highlight #")[1].split(":")[0]) - 1
                if 0 <= highlight_num < len(self.highlights):
                    start_time, end_time, _ = self.highlights[highlight_num]
                    # Jump to the start time in the video
                    self.media_player.setPosition(int(start_time * 1000))
                    self.current_time = start_time
                    # Set this as the current clip start/end times
                    self.clip_start_time = start_time
                    self.clip_end_time = end_time
                    self.start_entry.setText(self.format_time(start_time))
                    self.end_entry.setText(self.format_time(end_time))
                    self.update_clip_controls()
                    self.status_label.setText(f"Jumped to highlight #{highlight_num+1}")
            except Exception as e:
                print(f"Error processing highlight click: {e}")

    # Handle cutting the current highlight directly
    def handle_highlight_cut(self):
        if not self.highlights:
            return
            
        # Get currently selected highlight
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()
        
        if "Highlight #" in selected_line:
            highlight_num = int(selected_line.split("Highlight #")[1].split(":")[0]) - 1
            if 0 <= highlight_num < len(self.highlights):
                start_time, end_time, title = self.highlights[highlight_num]
                self.clip_start_time = start_time
                self.clip_end_time = end_time
                self.save_clip()

    # Handle saving the current highlight
    def handle_highlight_save(self):
        if not self.highlights:
            return
            
        # Similar to cut but just sets the times without saving
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()
        
        if "Highlight #" in selected_line:
            highlight_num = int(selected_line.split("Highlight #")[1].split(":")[0]) - 1
            if 0 <= highlight_num < len(self.highlights):
                start_time, end_time, _ = self.highlights[highlight_num]
                self.clip_start_time = start_time
                self.clip_end_time = end_time
                self.start_entry.setText(self.format_time(start_time))
                self.end_entry.setText(self.format_time(end_time))
                self.update_clip_controls()
                self.status_label.setText(f"Set times to highlight #{highlight_num+1}")

    # Handle rejecting/removing a highlight
    def handle_highlight_reject(self):
        if not self.highlights:
            return
            
        cursor = self.highlights_textbox.textCursor()
        cursor.select(QTextCursor.LineUnderCursor)
        selected_line = cursor.selectedText()
        
        if "Highlight #" in selected_line:
            highlight_num = int(selected_line.split("Highlight #")[1].split(":")[0]) - 1
            if 0 <= highlight_num < len(self.highlights):
                # Remove the highlight
                self.highlights.pop(highlight_num)
                # Update display
                self.display_highlights()
                if not self.highlights:
                    self.cut_clip_button.setEnabled(False)
                    self.save_clip_button.setEnabled(False)
                    self.reject_button.setEnabled(False)
                    self.highlights_textbox.setPlainText("All highlights have been reviewed.")

    # Update which clip control buttons are enabled based on current state
    def update_clip_controls(self):
        can_save = (self.clip_start_time is not None and 
                    self.clip_end_time is not None and 
                    self.clip_end_time > self.clip_start_time)
        self.save_button.setEnabled(can_save)

    # Format time in seconds to HH:MM:SS format
    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    # Parse a time string in HH:MM:SS format to seconds
    def parse_time_string(self, time_str):
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            try:
                return int(time_str)
            except ValueError:
                raise ValueError("Invalid time format")

    # Enable or disable controls based on whether media is loaded
    def enable_controls(self, enabled):
        self.play_button.setEnabled(enabled)
        self.start_button.setEnabled(enabled)
        self.end_button.setEnabled(enabled)
        self.apply_manual_button.setEnabled(enabled)
        self.transcribe_button.setEnabled(enabled)

    # Handle errors in the media player
    def handle_error(self):
        error_message = self.media_player.errorString()
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Media Error", f"An error occurred: {error_message}")

# Main entry point for the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoTranscriberEditor()
    window.show()
    sys.exit(app.exec_())
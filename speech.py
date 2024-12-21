import os
import sys
import json
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QScrollArea
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtCore import Qt, QUrl, QTimer, Slot
from rapidfuzz import process
import speech_recognition as sr

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# File paths
json_file = "./WLASL_v0.3.json"
videos_folder = "./videos"

# 1. Load JSON file with words
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words_list = [entry['gloss'] for entry in data]
    id_list = [entry['instances'][1]['video_id'] for entry in data]  # Get the second video_id of each word
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

print(id_list)

# 2. Check if essential files exist
if not os.path.exists(videos_folder):
    print(f"Error: Videos folder {videos_folder} does not exist.")
    exit()

# 3. Initialize Speech Recognizer
recognizer = sr.Recognizer()

# 4. Main Window to handle GUI and video playback
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech-to-Video Playback")
        self.setGeometry(100, 100, 900, 600)

        # Set background color
        self.setStyleSheet("background-color: #2e2e2e;")

        # Setup UI
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Video player widget
        self.video_widget = QVideoWidget(self)
        self.video_widget.setMinimumHeight(400)
        self.layout.addWidget(self.video_widget, stretch=3)

        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video_widget)

        # Label to display recognized text
        self.label = QLabel("Welcome! Speak to play a video.", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            font-size: 24px; 
            font-family: Arial, sans-serif;
            color: #ffffff;
            background-color: #4a4a4a; 
            padding: 15px; 
            border-radius: 10px;
        """)
        self.layout.addWidget(self.label, stretch=1)

        # Start button
        self.button = QPushButton("Start Recognition", self)
        self.button.setStyleSheet("""
            QPushButton {
                font-size: 22px; 
                background-color: #4CAF50; 
                border-radius: 15px; 
                padding: 15px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049; 
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.button.clicked.connect(self.start_recognition)
        self.layout.addWidget(self.button, stretch=1)

        # Create a Scroll Area to make the layout scrollable
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.central_widget)

        # Set scroll area as the central widget
        self.setCentralWidget(self.scroll_area)

        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.countdown = 3

        # To manage video sequence and track total duration
        self.video_queue = []
        self.current_video_index = 0
        self.processed_words = []  # To track the words and their corresponding IDs
        self.total_videos = 0  # Total number of videos

        # Connect media status change to track progress
        self.player.mediaStatusChanged.connect(self.handle_media_status)

    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.play_next_video()

    def update_timer(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.label.setText(f"Time remaining: {self.countdown}")
        else:
            self.timer.stop()
            self.label.setText("Processing speech...")
            self.recognize_and_process()

    def start_recognition(self):
        self.label.setText("Please wait...")
        self.countdown = 3
        self.label.setText(f"Time remaining to speak: {self.countdown}")
        self.timer.start(1000)

    def recognize_and_process(self):
        # Recognize speech using SpeechRecognition
        recognized_text = self.recognize_speech()
        if recognized_text:
            self.label.setText(f"Recognized: {recognized_text}")
            self.process_sentence(recognized_text)
        else:
            self.label.setText("No speech recognized. Please try again.")

    def recognize_speech(self):
        with sr.Microphone() as source:
            self.label.setText("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                return text
            except sr.UnknownValueError:
                print("Speech not recognized.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None

    def process_sentence(self, sentence):
        words = sentence.split()
        self.video_queue = []  # Reset the video queue
        self.processed_words = []  # Reset the list of processed words

        for word in words:
            # Match the recognized word with words_list using fuzzy matching
            best_match = process.extractOne(word.lower(), words_list)
            if best_match:
                best_match_index = words_list.index(best_match[0])
                video_id = id_list[best_match_index]  # Retrieve the corresponding video_id
                video_path = os.path.join(videos_folder, f"{video_id}.mp4")  # Construct the video file path

                # Check if the video file exists in the videos folder
                if os.path.exists(video_path):
                    self.video_queue.append(video_path)
                    self.processed_words.append((best_match[0], video_id))  # Add word and video ID to processed list

        if self.video_queue:
            self.current_video_index = 0
            self.play_next_video()  # Start playing the first video
        else:
            self.label.setText("No matching videos found.")

    @Slot()
    def play_next_video(self):
        if self.current_video_index < len(self.video_queue):
            video_path = self.video_queue[self.current_video_index]
            word, _ = self.processed_words[self.current_video_index]

            # Highlight the current word in the recognized sentence
            self.highlight_sentence(word)

            # Set the video source and play
            self.player.setSource(QUrl.fromLocalFile(video_path))
            self.player.play()

            self.current_video_index += 1  # Move to the next video
        else:
            self.current_video_index = 0  # Reset to the first video
            self.play_next_video()  # Replay the videos

    def highlight_sentence(self, current_word):
        original_sentence = " ".join([word for word, _ in self.processed_words])
        highlighted_text = ""

        for word in original_sentence.split():
            if word.lower() == current_word.lower():
                highlighted_text += f"<span style='color: #ff5733; font-weight: bold;'>{word}</span> "
            else:
                highlighted_text += f"{word} "

        self.label.setText(f"<h3 style='text-align: center;'>{highlighted_text.strip()}</h3>")

# 5. Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
import sys
import os
import json
import re
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtCore import Qt, QUrl, QTimer
import sounddevice as sd
import vosk

# Set encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# File paths
json_file = "./GG5.json"  # Path to store JSON data
videos_folder = "./wlasl_videos"  # Path where video files are stored
model_path = "./vosk/vosk-model-small-en-us-0.15"  # Path to vosk model

# Load JSON file for words and video paths
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        word_data = json.load(f)
        words_list = [entry["word"] for entry in word_data]  # Extract words
        paths_dict = {entry["word"]: entry["video_id"] for entry in word_data}  # Map words to video IDs
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

# Check for necessary files
if not os.path.exists(videos_folder):
    print(f"Error: Videos folder '{videos_folder}' does not exist.")
    exit()
if not os.path.exists(model_path):
    print(f"Error: Model not found at path '{model_path}'.")
    exit()

# Load Vosk model
model = vosk.Model(model_path)


# Speech-to-text recognition function
def recognize_speech():
    recognizer = vosk.KaldiRecognizer(model, 16000)
    with sd.InputStream(channels=1, samplerate=16000, dtype='int16') as stream:
        while True:
            audio_data, overflowed = stream.read(8000)
            if recognizer.AcceptWaveform(audio_data.tobytes()):
                result = recognizer.Result()
                return json.loads(result)['text']
    return None


# Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Recognition and Video Display")

        # Set fullscreen
        self.showFullScreen()

        # Layout setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        # Signer ID input
        self.signer_id_input = QLineEdit(self)
        self.signer_id_input.setPlaceholderText("Enter signer ID here")
        self.layout.addWidget(self.signer_id_input, stretch=1)

        # Video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(400)
        self.layout.addWidget(self.video_widget, stretch=3)

        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video_widget)

        # Sentence label
        self.sentence_label = QLabel("Recognized sentence will appear here.")
        self.sentence_label.setAlignment(Qt.AlignCenter)
        self.sentence_label.setStyleSheet("font-size: 20px; color: black; background-color: #D3D3D3; padding: 10px;")
        self.layout.addWidget(self.sentence_label, stretch=1)

        # Instruction label
        self.label = QLabel("Welcome!")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 24px; color: white; background-color: #4A90E2; padding: 10px;")
        self.layout.addWidget(self.label, stretch=1)

        # Text input field
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter a sentence here")
        self.layout.addWidget(self.text_input, stretch=1)

        # Start button for text input
        self.text_button = QPushButton("Start with Text Input")
        self.text_button.setStyleSheet(""" 
            font-size: 18px;
            background-color: #50E3C2;
            padding: 12px;
            border-radius: 10px;
            color: white;
            transition: background-color 0.3s ease;
        """)
        self.text_button.setFixedHeight(40)
        self.text_button.setCursor(Qt.PointingHandCursor)
        self.text_button.clicked.connect(self.start_from_text)
        self.layout.addWidget(self.text_button, stretch=1)

        # Start button for voice input
        self.voice_button = QPushButton("Start with Voice Input")
        self.voice_button.setStyleSheet(""" 
            font-size: 18px;
            background-color: #50E3C2;
            padding: 12px;
            border-radius: 10px;
            color: white;
            transition: background-color 0.3s ease;
        """)
        self.voice_button.setFixedHeight(40)
        self.voice_button.setCursor(Qt.PointingHandCursor)
        self.voice_button.clicked.connect(self.start_from_voice)
        self.layout.addWidget(self.voice_button, stretch=1)

        # Countdown timer label
        self.timer_label = QLabel("Seconds remaining to speak: 3")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 18px; color: white; background-color: #FF5733; padding: 10px;")
        self.layout.addWidget(self.timer_label, stretch=1)

        # Apply layout
        self.central_widget.setLayout(self.layout)

        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.countdown = 3  # Countdown starts from 3

        self.is_countdown_done = False  # Variable to track countdown status

        self.video_queue = []  # This will store the video queue based on the words in the sentence
        self.current_video_index = 0  # To track the current video being played
        self.words_to_show = []  # To track the words that need to be displayed

        # Connect player media status change to handle video end
        self.player.mediaStatusChanged.connect(self.handle_video_end)

    def update_timer(self):
        if self.countdown > 0:
            self.countdown -= 1
            self.timer_label.setText(f"Seconds remaining: {self.countdown}")
        else:
            self.timer.stop()
            self.is_countdown_done = True
            self.label.setText("Please wait, recognizing speech...")
            self.recognize_and_display()

    def start_from_text(self):
        input_text = self.text_input.text()
        if input_text:
            self.search_in_json(input_text)

    def start_from_voice(self):
        self.label.setText("Please wait...")
        self.countdown = 3
        self.timer_label.setText(f"Seconds remaining: {self.countdown}")
        self.timer.start(1000)
        self.is_countdown_done = False

    def recognize_and_display(self):
        if self.is_countdown_done:
            recognized_text = recognize_speech()
            if recognized_text:
                self.search_in_json(recognized_text)

    def search_in_json(self, text):
        signer_id_input = self.signer_id_input.text()
        if not signer_id_input.isdigit():
            self.label.setText("Please enter a valid signer ID.")
            return
        signer_id = int(signer_id_input)

        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        split_words = text.lower().split()

        print("Words array:", split_words)

        self.sentence = text

        highlighted_sentence = ''
        for word in split_words:
            if word in paths_dict:
                highlighted_sentence += f'<span style="background-color: yellow; font-weight: bold;">{word}</span> '
            else:
                highlighted_sentence += f'{word} '

        self.sentence_label.setText(f"Sentence: {highlighted_sentence}")

        self.video_queue = []
        self.current_video_index = 0

        for word in split_words:
            if word in paths_dict:
                matching_videos = [entry for entry in word_data if
                                   entry["word"] == word and entry["signer_id"] == signer_id]

                if matching_videos:
                    video_id = matching_videos[0]["video_id"]
                    video_path = f"{videos_folder}/{video_id}.mp4"
                    if os.path.exists(video_path):
                        self.video_queue.append((word, video_path))
                        print(f"Video path found for word '{word}' with signer ID {signer_id}: {video_path}")
                else:
                    fallback_videos = [entry for entry in word_data if entry["word"] == word]
                    if fallback_videos:
                        video_id = fallback_videos[0]["video_id"]
                        video_path = f"{videos_folder}/{video_id}.mp4"
                        if os.path.exists(video_path):
                            self.video_queue.append((word, video_path))
                            print(f"Fallback video path found for word '{word}': {video_path}")

        if self.video_queue:
            self.play_next_video()

    def play_next_video(self):
        if self.current_video_index < len(self.video_queue):
            word, video_path = self.video_queue[self.current_video_index]
            print(f"Playing video for word: {word} at {video_path}")

            self.highlight_current_word(word)

            if os.path.exists(video_path):
                self.player.setSource(QUrl.fromLocalFile(video_path))
                self.player.play()
        else:
            self.current_video_index = 0
            self.play_next_video()

    def highlight_current_word(self, word):
        split_words = self.sentence.lower().split()
        highlighted_sentence = ''
        for w in split_words:
            if w == word:
                highlighted_sentence += f'<span style="background-color: yellow; font-weight: bold;">{w}</span> '
            else:
                highlighted_sentence += f'{w} '

        self.sentence_label.setText(f"Sentence: {highlighted_sentence}")

    def handle_video_end(self):
        if self.player.mediaStatus() == QMediaPlayer.EndOfMedia:
            self.current_video_index += 1
            self.play_next_video()


# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.showFullScreen()
sys.exit(app.exec())

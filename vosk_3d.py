import os
import sys
import csv
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QScrollArea
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtCore import Qt, QUrl, QTimer, Slot
from rapidfuzz import process
from vosk import Model, KaldiRecognizer
import pyaudio

# Set UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# File paths
csv_file = "./3d_words.csv"  # CSV file containing words and video IDs
videos_3d_folder = "./3d_videos"  # Folder containing videos for recognized words
vosk_model_path = "./vosk/vosk-model-small-en-us-0.15"

# 1. Load CSV file with words
try:
    words_list = []
    id_list = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            words_list.append(row['word'])  # Assuming the CSV column is named 'word'
            id_list.append(row['video_id'])  # Assuming the CSV column is named 'video_id'
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

print("Loaded words and video IDs:", list(zip(words_list, id_list)))

# 2. Check if essential files and folders exist
if not os.path.exists(videos_3d_folder):
    print(f"Error: Videos folder {videos_3d_folder} does not exist.")
    exit()

# 3. Initialize Vosk Model and Recognizer
try:
    model = Model(vosk_model_path)
    recognizer = KaldiRecognizer(model, 16000)
except Exception as e:
    print(f"Error initializing Vosk model: {e}")
    exit()

# Setup PyAudio for microphone input
p = pyaudio.PyAudio()

# 4. Main Window to handle GUI and video playback
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech-to-Video Playback")
        self.showFullScreen()

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
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

        # Merged label to display both welcome message and recognized words
        self.info_label = QLabel("Welcome!", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            font-size: 20px; color: #ffffff; background-color: #4a4a4a;
            padding: 15px; border-radius: 10px; margin-bottom: 10px;
        """)
        self.info_label.setWordWrap(True)
        self.layout.addWidget(self.info_label, stretch=0)

        # Start button
        self.button = QPushButton("Start Recognition", self)
        self.button.setStyleSheet("""
            font-size: 22px; background-color: #4CAF50; border-radius: 15px;
            padding: 15px; color: white; font-weight: bold;
        """)
        self.button.clicked.connect(self.start_recognition)
        self.layout.addWidget(self.button, stretch=1)

        self.video_queue = []
        self.current_video_index = 0
        self.highlighted_text = []

    def start_recognition(self):
        self.info_label.setText("Processing speech... Please wait.")
        recognized_text = self.recognize_speech_vosk()
        if recognized_text:
            self.info_label.setText(f"Recognized: {recognized_text}")
            self.process_sentence(recognized_text)
        else:
            self.info_label.setText("No speech recognized. Please try again.")

    def recognize_speech_vosk(self):
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()

        print("Speak now...")
        while True:
            data = stream.read(4000)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = eval(result)['text']  # Extracting recognized text
                print(f"Recognized speech: {text}")  # Print recognized text
                stream.close()
                return text
        return None

    def process_sentence(self, sentence):
        words = sentence.split()
        self.video_queue = []
        self.highlighted_text = []

        for word in words:
            best_match = process.extractOne(word.lower(), words_list)
            if best_match and best_match[1] > 70:  # Ensure confidence threshold
                matched_word = best_match[0]
                video_path = os.path.join(videos_3d_folder, f"{matched_word}.mp4")

                if os.path.exists(video_path):
                    self.video_queue.append((matched_word, video_path))
                    self.highlighted_text.append(f"<span style='color:white;'>{matched_word}</span>")
                else:
                    self.highlighted_text.append(f"<span style='color:white;'>{word}</span>")

        self.update_word_display()

        if self.video_queue:
            self.current_video_index = 0
            self.play_next_video()
        else:
            self.info_label.setText("No matching videos found.")

    def update_word_display(self):
        display_text = " ".join(self.highlighted_text)

        # Add the necessary HTML structure to make sure styling is applied
        html_content = f"""
        <html>
            <body style="color: white; font-size: 18px; font-family: Arial, sans-serif; text-align: center; font-weight: bold;">
                <p>{display_text}</p>
            </body>
        </html>
        """
        self.info_label.setText(html_content)

    @Slot()
    def play_next_video(self):
        if self.current_video_index < len(self.video_queue):
            matched_word, video_path = self.video_queue[self.current_video_index]

            # Highlight the current word in yellow
            self.highlighted_text[self.current_video_index] = f"<span style='color:yellow; font-weight: bold;'>{matched_word}</span>"
            self.update_word_display()

            # Play the video
            self.video_widget.showFullScreen()  # Enable full-screen mode
            self.player.setSource(QUrl.fromLocalFile(video_path))
            self.player.play()
        else:
            self.current_video_index = 0  # Reset to the first video for looping
            self.play_next_video()

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            # Reset the color of the word back to white after the video ends
            self.highlighted_text[self.current_video_index] = f"<span style='color:white;'>{self.video_queue[self.current_video_index][0]}</span>"
            self.update_word_display()

            # Move to the next video
            self.current_video_index += 1
            if self.current_video_index < len(self.video_queue):
                self.play_next_video()
            else:
                QTimer.singleShot(100, self.repeat_sentence)  # Delay to ensure it is visible before repeating

    def repeat_sentence(self):
        # Reset the sentence display and start from the first word
        self.current_video_index = 0
        self.play_next_video()

# 5. Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

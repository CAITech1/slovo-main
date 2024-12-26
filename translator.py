import pandas as pd
import os
import cv2
from deep_translator import GoogleTranslator

excel_file_path = 'slovo.xlsx'  # Update with the path to your Excel file
videos_folder_path = 'slovo_videos'  # Update with the path to your slovo_videos folder
input_sentence = "bag"

def load_excel_data(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    return df

def translate_word(word):
    # Translate a single word to Russian
    translation = GoogleTranslator(source='en', target='ru').translate(word)
    return translation

def play_videos_sequentially(translated_words, df, videos_folder):
    for word in translated_words:
        # Find the corresponding video file
        row = df[df['text'] == word]
        if not row.empty:
            video_file = row.iloc[0]['attachment_id'] + '.mp4'  # Assuming video files are in .mp4 format
            video_path = os.path.join(videos_folder, video_file)
            if os.path.exists(video_path):
                print(f"Playing video for word: {word}")
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize the frame to 1920x1080
                    resized_frame = cv2.resize(frame, (450, 650))
                    cv2.imshow(f"Word: {word}", resized_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"Video file not found: {video_path}")
        else:
            print(f"Word '{word}' not found in Excel file.")

# Split sentence into words, translate each, and store in a list
def translate_sentence_to_russian(sentence):
    words = sentence.split()
    translated_words = [translate_word(word) for word in words]
    return translated_words

# Load data, translate, and play slovo_videos
df = load_excel_data(excel_file_path)
translated_words = translate_sentence_to_russian(input_sentence)
print(f"Translated Words: {translated_words}")
play_videos_sequentially(translated_words, df, videos_folder_path)
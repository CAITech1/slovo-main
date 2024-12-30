import os
import cv2
import pandas as pd

excel_file_path = 'slovo-en.xlsx'  # Update with the path to your Excel file
videos_folder_path = 'slovo_videos'  # Update with the path to your slovo_videos folder
input_sentence = "friendly"

def load_excel_data(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)
    return df

def find_word_id(word, df):
    # Find the ID for a given word in the DataFrame
    row = df[df['text'] == word]
    if not row.empty:
        return row.iloc[0]['attachment_id']  # Assuming column 'attachment_id' exists
    return None

def play_videos_sequentially(words, df, videos_folder):
    for word in words:
        word_id = find_word_id(word, df)
        if word_id is not None:
            video_file = f"{word_id}.mp4"  # Assuming video files are in .mp4 format
            video_path = os.path.join(videos_folder, video_file)
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize the frame to 450x650
                    resized_frame = cv2.resize(frame, (450, 650))
                    cv2.imshow(f"Word: {word}", resized_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                print(f"Video file not found: {video_path}")
        else:
            print(f"Word '{word}' not found in the Excel file.")

# Split sentence into words and match them with the Excel data
def match_words_with_ids(sentence, df):
    words = sentence.split()
    for word in words:
        word_id = find_word_id(word, df)
        if word_id is not None:
            print(f"{word_id},{word}")
    return words

# Load data, match words with IDs, and play slovo_videos
df = load_excel_data(excel_file_path)
matched_words = match_words_with_ids(input_sentence, df)
play_videos_sequentially(matched_words, df, videos_folder_path)

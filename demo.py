import argparse
import sys
import time
import subprocess
from collections import deque
from multiprocessing import Manager, Process, Value
from typing import Optional, Tuple

import onnxruntime as ort
from loguru import logger
import cv2
import numpy as np
from omegaconf import OmegaConf

from constants import classes


# Define the path to your subprocess
python_path = "../.venv/Scripts/python.exe"
index = "./index.py"

# Function to call your subprocess (text-to-video)
def subprocess_to_index():
    print("Translating from text to video...")
    try:
        result = subprocess.run(
            [python_path, index],
            check=True,
            text=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")


# BaseRecognition and other classes (unchanged)

class BaseRecognition:
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        self.verbose = verbose
        self.started = None
        self.output_names = None
        self.input_shape = None
        self.input_name = None
        self.session = None
        self.model_path = model_path
        self.window_size = None
        self.tensors_list = tensors_list
        self.prediction_list = prediction_list

    def clear_tensors(self):
        """
        Clear the list of tensors.
        """
        for _ in range(self.window_size):
            self.tensors_list.pop(0)

    def run(self):
        """
        Run the recognition model.
        """
        if self.session is None:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.window_size = self.input_shape[3]
            self.output_names = [output.name for output in self.session.get_outputs()]

        if len(self.tensors_list) >= self.input_shape[3]:
            input_tensor = np.stack(self.tensors_list[: self.window_size], axis=1)[None][None]
            st = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor.astype(np.float32)})[0]
            et = round(time.time() - st, 3)
            gloss = str(classes[outputs.argmax()])
            if gloss != self.prediction_list[-1] and len(self.prediction_list):
                if gloss != "---":
                    self.prediction_list.append(gloss)
            self.clear_tensors()
            if self.verbose:
                logger.info(f"- Prediction time {et}, new gloss: {gloss}")
                logger.info(f" --- {len(self.tensors_list)} frames in queue")

    def kill(self):
        pass


class Recognition(BaseRecognition):
    def __init__(self, model_path: str, tensors_list: list, prediction_list: list, verbose: bool):
        """
        Initialize recognition model.
        """
        super().__init__(model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list, verbose=verbose)
        self.started = True

    def start(self):
        self.run()


class RecognitionMP(Process, BaseRecognition):
    def __init__(self, model_path: str, tensors_list, prediction_list, verbose):
        """
        Initialize recognition model.
        """
        super().__init__()
        BaseRecognition.__init__(self, model_path=model_path, tensors_list=tensors_list, prediction_list=prediction_list, verbose=verbose)
        self.started = Value("i", False)

    def run(self):
        while True:
            BaseRecognition.run(self)
            self.started = True


class Runner:
    STACK_SIZE = 6

    def __init__(self, model_path: str, config: OmegaConf = None, mp: bool = False, verbose: bool = False, length: int = STACK_SIZE) -> None:
        """
        Initialize runner.
        """
        self.multiprocess = mp
        self.cap = cv2.VideoCapture(0)
        self.manager = Manager() if self.multiprocess else None
        self.tensors_list = self.manager.list() if self.multiprocess else []
        self.prediction_list = self.manager.list() if self.multiprocess else []
        self.prediction_list.append("---")
        self.frame_counter = 0
        self.frame_interval = config.frame_interval
        self.length = length
        self.prediction_classes = deque(maxlen=length)
        self.mean = config.mean
        self.std = config.std
        if self.multiprocess:
            self.recognizer = RecognitionMP(model_path, self.tensors_list, self.prediction_list, verbose)
        else:
            self.recognizer = Recognition(model_path, self.tensors_list, self.prediction_list, verbose)

    def add_frame(self, image):
        """
        Add frame to queue.
        """
        self.frame_counter += 1
        if self.frame_counter == self.frame_interval:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize(image, (224, 224))
            image = (image - self.mean) / self.std
            image = np.transpose(image, [2, 0, 1])
            self.tensors_list.append(image)
            self.frame_counter = 0

    @staticmethod
    def resize(im, new_shape=(224, 224)):
        """
        Resize and pad image while preserving aspect ratio.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
        return im

    def run(self):
        """
        Run the runner.
        """
        if self.multiprocess:
            self.recognizer.start()

        screen_width = 1920  # Replace with your screen's width
        screen_height = 1080  # Replace with your screen's height

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

        while self.cap.isOpened():
            if self.recognizer.started:
                _, frame = self.cap.read()
                text_div = np.zeros((50, frame.shape[1], 3), dtype=np.uint8)
                self.add_frame(frame)

                if not self.multiprocess:
                    self.recognizer.start()

                if self.prediction_list:
                    text = "  ".join(self.prediction_list)
                    cv2.putText(text_div, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

                if len(self.prediction_list) > self.length:
                    self.prediction_list.pop(0)

                frame = np.concatenate((frame, text_div), axis=0)

                # Add button to the frame
                frame = add_button_to_frame(frame)

                # Make the window fullscreen
                cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)  # Set window to fullscreen
                cv2.imshow("frame", frame)

                condition = cv2.waitKey(10) & 0xFF
                if condition in {ord("q"), ord("Q"), 27}:  # q or Esc to quit
                    if self.multiprocess:
                        self.recognizer.kill()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break

# Adding the button to the top-left corner of the video feed
# Adding the button to the top-left corner of the video feed
def add_button_to_frame(frame):
    button_text = "Back"
    button_position = (40, 40)
    button_size = (120, 50)
    color = (74, 74, 74)
    text_color = (255, 255, 255)
    radius = 10  # Radius for rounded corners

    # Draw a rounded rectangle for the button
    top_left = (10, 10)
    bottom_right = (button_size[0] + 10, button_size[1] + 10)

    # Draw the four rounded corners
    cv2.ellipse(frame, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, color, -1)
    cv2.ellipse(frame, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 270, 0, 90, color, -1)
    cv2.ellipse(frame, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 90, 0, 90, color, -1)
    cv2.ellipse(frame, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, color, -1)

    # Draw the rectangle part (without the corners)
    cv2.rectangle(frame, (top_left[0] + radius, top_left[1]), (bottom_right[0] - radius, bottom_right[1]), color, -1)
    cv2.rectangle(frame, (top_left[0], top_left[1] + radius), (bottom_right[0], bottom_right[1] - radius), color, -1)

    # Add text inside the button
    cv2.putText(frame, button_text, button_position, cv2.FONT_HERSHEY_COMPLEX, 0.7, text_color, 2)

    return frame

# Function to check if the button is clicked
def check_button_click(x, y):
    # Define button position and size
    button_x_start = 10
    button_y_start = 10
    button_x_end = 210  # 200px width of button + 10px padding
    button_y_end = 60   # 50px height of button + 10px padding

    # Check if the mouse click is inside the button's area
    if button_x_start < x < button_x_end and button_y_start < y < button_y_end:
        return True
    return False

def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if check_button_click(x, y):
            # Simulate changing cursor to a pointer (you can add a print statement for feedback)
            print("Cursor is over the button. Change cursor to pointer.")
        else:
            # Simulate changing cursor back to normal
            print("Cursor is outside the button. Change cursor back to normal.")

# Define the callback function for mouse events
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button clicked
        if check_button_click(x, y):  # Check if click is inside button
            subprocess_to_index()

# Argument parsing function
def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")

    # Static values are set directly here
    static_args = {
        "config": "config.yaml",  # Replace with the actual path
        "mp": False,  # Enable multiprocessing (True/False)
        "verbose": True,  # Enable logging (True/False)
        "length": 4  # Deque length for predictions
    }

    # Add arguments but they will be overridden by static values
    parser.add_argument("-p", "--config", required=False, type=str, help="Path to config")
    parser.add_argument("--mp", required=False, action="store_true", help="Enable multiprocessing")
    parser.add_argument("--verbose", required=False, action="store_true", help="Enable verbose logging")
    parser.add_argument("--length", required=False, type=int, help="Length of prediction deque")

    args = parser.parse_args(params if params else [])
    # Override static args with command-line args
    for key, value in static_args.items():
        setattr(args, key, value)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.config)
    runner = Runner(conf.model_path, conf, args.mp, args.verbose, args.length)
    runner.run()

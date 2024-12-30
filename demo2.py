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

        def mouse_callback(event, x, y, flags, param):
            """
            Callback for mouse clicks.
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                btn_x, btn_y, radius = param
                distance = ((x - (btn_x + radius)) ** 2 + (y - (btn_y + radius)) ** 2) ** 0.5
                if distance <= radius:
                    # Button clicked
                    print("Rounded button clicked, going back to interface...")

                    # Launch the external interface immediately (non-blocking)
                    subprocess.Popen([python_path, index])  # Non-blocking process start

                    # Cleanup resources (after launching the new interface)
                    self.cap.release()
                    cv2.destroyAllWindows()

                    # Exit the current process
                    sys.exit()

        # Set mouse callback and make the window fullscreen
        # Make the window fullscreen
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)  # Create fullscreen window
        cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.cap.isOpened():
            if self.recognizer.started:
                _, frame = self.cap.read()
                text_div = np.zeros((70, frame.shape[1], 3), dtype=np.uint8)
                self.add_frame(frame)

                if not self.multiprocess:
                    self.recognizer.start()

                if self.prediction_list:
                    text = "  ".join(self.prediction_list)
                    cv2.putText(text_div, text, (20,40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)

                if len(self.prediction_list) > self.length:
                    self.prediction_list.pop(0)

                frame = np.concatenate((frame, text_div), axis=0)

                # Add rounded button to the frame and get its parameters
                frame, button_params = add_button_to_frame(frame)

                # Attach the mouse callback with button parameters
                cv2.setMouseCallback("frame", mouse_callback, param=button_params)

                # Show the frame
                cv2.imshow("frame", frame)

                # Exit conditions
                condition = cv2.waitKey(10) & 0xFF
                if condition in {ord("q"), ord("Q"), 27}:  # q or Esc to quit
                    if self.multiprocess:
                        self.recognizer.kill()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break


def add_button_to_frame(frame):
    """
    Add a rounded button to the frame.

    Parameters:
    frame (np.ndarray): The input video frame.

    Returns:
    tuple: Updated frame and button rectangle (x, y, radius).
    """
    # Button properties
    center_x, center_y = 100, 100  # Position of the button (top-left corner)
    radius = 50  # Radius of the rounded button
    button_color = (0, 122, 255)  # Button color (BGR)
    text_color = (255, 255, 255)  # Text color
    thickness = -1  # Fill the circle

    # Draw the button as a circle
    cv2.circle(frame, (center_x, center_y), radius, button_color, thickness)

    # Add text to the button
    text = "Back"
    font_scale = 1 # Increased font scale
    font_thickness = 4  # Increased thickness
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # Return the updated frame and button parameters
    return frame, (center_x - radius, center_y - radius, radius)


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

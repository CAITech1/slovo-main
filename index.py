import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess

python_path = "../.venv/Scripts/python.exe"
vosk = "./vosk_code.py"  # Path to the vosk_code.py script
slovo = "./demo2.py"  # Path to the demo.py script
background = "./assets/Index_background.jpg"
button = "./assets/Index_transparent_button.png"  # Path to the button image

# Translation functions:
def translate_text_to_video():
    print("Translating from text to video...")
    try:
        result = subprocess.run(
            [python_path, vosk],
            check=True,
            text=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the external file: {e}\nOutput: {e.output}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

def translate_video_to_text():
    print("Translating from video to text...")
    try:
        text_result = subprocess.run(
            [python_path, slovo],
            check=True,
            text=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the external file: {e}\nOutput: {e.output}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

# Tkinter window setup
root = tk.Tk()
root.title("Sign Language Translation Interface")
root.attributes('-fullscreen', True)  # Make the window fullscreen

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Load the background image
background_image = Image.open(background)
resized_background = background_image.resize((screen_width, screen_height))
background_photo = ImageTk.PhotoImage(resized_background)

# Create a canvas to add image and buttons
canvas = tk.Canvas(root, width=screen_width, height=screen_height)
canvas.pack()

# Add the background image to the canvas
canvas.create_image(0, 0, image=background_photo, anchor=tk.NW)

# Create a transparent button (using a transparent image)
button_image = Image.open(button)
button_width, button_height = 400, 80  # Define button size
button_image_resized = button_image.resize((button_width, button_height))
button_photo = ImageTk.PhotoImage(button_image_resized)

# Calculate button positions
button1_x, button1_y = screen_width // 4 - 45, int(screen_height * 0.9)  # Move 20 pixels left
button2_x, button2_y = (screen_width // 4) * 3, int(screen_height * 0.9)

# Add buttons on the canvas
button1 = canvas.create_image(button1_x, button1_y, image=button_photo, tags="button1")
button2 = canvas.create_image(button2_x, button2_y, image=button_photo, tags="button2")

# Bind button actions
canvas.tag_bind(button1, "<Button-1>", lambda event: translate_text_to_video())
canvas.tag_bind(button2, "<Button-1>", lambda event: translate_video_to_text())

# Change cursor to pointer on hover
def on_hover(event):
    canvas.config(cursor="hand2")  # Use "hand2" for pointer cursor in Tkinter

def on_leave(event):
    canvas.config(cursor="")  # Reset to default cursor

# Bind hover effects to the buttons
canvas.tag_bind(button1, "<Enter>", on_hover)
canvas.tag_bind(button1, "<Leave>", on_leave)
canvas.tag_bind(button2, "<Enter>", on_hover)
canvas.tag_bind(button2, "<Leave>", on_leave)

# Run the application
root.mainloop()
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import os
import sys

# -------------------
# Paths
# -------------------
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RECORDED_PATH = os.path.join(BASE_FOLDER, 'PROJECT', 'GUI.py')
LIVE_PATH = os.path.join(BASE_FOLDER, 'vision_aiii', 'main.py')
BG_IMAGE_PATH = os.path.join(BASE_FOLDER, "vs.jpg")  # your background image

# -------------------
# Functions
# -------------------
def run_script(script_path, script_name):
    if os.path.exists(script_path):
        folder = os.path.dirname(script_path)
        subprocess.Popen(
            [sys.executable, script_path],
            cwd=folder,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        status_label.config(text=f"{script_name} launched successfully!", fg="green")
    else:
        status_label.config(text=f"{script_name} not found!", fg="red")
        messagebox.showerror("Error", f"{script_name} not found at {script_path}")

def run_recorded(): run_script(RECORDED_PATH, "Recorded Video")
def run_live(): run_script(LIVE_PATH, "Live Video")

# -------------------
# Draw Rounded Button on Canvas
# -------------------
def create_round_button(canvas, x, y, w, h, r, text, command,
                        bg="#4CAF50", hover="#45a049", fg="white", font=("Arial", 20, "bold")):
    """Draws a rounded rectangle button directly on the canvas"""

    # Rounded rectangle points
    points = [
        x+r, y,
        x+w-r, y,
        x+w, y,
        x+w, y+r,
        x+w, y+h-r,
        x+w, y+h,
        x+w-r, y+h,
        x+r, y+h,
        x, y+h,
        x, y+h-r,
        x, y+r,
        x, y
    ]
    btn = canvas.create_polygon(points, smooth=True, splinesteps=36, fill=bg, outline="")

    # Text
    txt = canvas.create_text(x+w//2, y+h//2, text=text, fill=fg, font=font)

    # Bind click + hover
    def on_click(event): command()
    def on_enter(event): canvas.itemconfig(btn, fill=hover)
    def on_leave(event): canvas.itemconfig(btn, fill=bg)

    for tag in (btn, txt):
        canvas.tag_bind(tag, "<Button-1>", on_click)
        canvas.tag_bind(tag, "<Enter>", on_enter)
        canvas.tag_bind(tag, "<Leave>", on_leave)

    return btn, txt

# -------------------
# GUI
# -------------------
root = tk.Tk()
root.title("Vision AI")
root.state('zoomed')
root.update()
WINDOW_WIDTH, WINDOW_HEIGHT = root.winfo_width(), root.winfo_height()

# Background
try:
    bg_image = Image.open(BG_IMAGE_PATH).resize((WINDOW_WIDTH, WINDOW_HEIGHT), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
except Exception as e:
    print(f"Could not load background: {e}")
    bg_photo = None

canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, highlightthickness=0, bd=0)
canvas.pack(fill="both", expand=True)

if bg_photo:
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Heading (Bigger + Bold)
heading_text = "Vision AI"
font_style = ("Helvetica", 64, "bold")  # Increased size + bold
canvas.create_text(WINDOW_WIDTH//2+3, 103, text=heading_text, font=font_style, fill="gray20")  # shadow
canvas.create_text(WINDOW_WIDTH//2, 100, text=heading_text, font=font_style, fill="white")

# -------------------
# Buttons (Edge-Free)
# -------------------
LEFT_MARGIN = 80
btn_spacing = 120

create_round_button(canvas, LEFT_MARGIN, 200, 350, 70, 35,
                    "📂 Recorded Video", run_recorded,
                    bg="#4CAF50", hover="#45a049")

create_round_button(canvas, LEFT_MARGIN, 200+btn_spacing, 350, 70, 35,
                    "📡 Live Video", run_live,
                    bg="#2196F3", hover="#1e88e5")

status_label = tk.Label(root, text="Select an option to launch...",
                        font=("Arial", 22), fg="white", bg="black")
canvas.create_window(LEFT_MARGIN, 200 + 2*btn_spacing, window=status_label, anchor="nw")

# Smaller Exit Button
create_round_button(canvas, LEFT_MARGIN, 200+3*btn_spacing, 140, 50, 25,
                    "❌ Exit", root.destroy,
                    bg="red", hover="#cc0000", font=("Arial", 16, "bold"))

root.mainloop()

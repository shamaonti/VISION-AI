import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
import imutils
from yolo_my import check_image

# --- Dummy login credentials ---
VALID_EMAIL = "user@example.com"
VALID_PASSWORD = "password123"

# Global video variables
vid, result, label_data = None, None, None

def hand_open_file(tag_entry, video_label):
    global vid, result, label_data
    file = askopenfile(mode='r', filetypes=[('MP4 Files', '*.mp4')])
    if file is None:
        return
    path = file.name

    vid = cv2.VideoCapture(path)
    if not vid.isOpened():
        print("Error reading video file")
        return

    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('summarized_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    label_data = tag_entry.get()
    hand_frame(video_label)

def hand_frame(video_label):
    global vid, result, label_data

    if not vid or not vid.isOpened():
        return

    ret, frame = vid.read()
    if not ret or frame is None:
        vid.release()
        result.release()
        print("Summarized video saved as 'summarized_output.avi'")
        return

    frame = cv2.flip(frame, 1)
    data0, img = check_image(frame, label_data)

    if data0 == 1:
        result.write(frame)
        frame = imutils.resize(img, width=700)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    video_label.after(10, lambda: hand_frame(video_label))

def open_main_app(root):
    # Use the main root window directly instead of opening a new one
    app = root
    app.geometry("1600x900")
    app.title("VISION AI")
    app.configure(bg="#1f1f1f")


    # Background image
    bg_image = Image.open("uu.jpg").resize((1600, 900))
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(app, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Title
    title = tk.Label(app, text="VISION AI", font=("Verdana", 35, "bold"), fg="white", bg="#1f1f1f")
    title.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 5))

    # Tag input
    tag_label = tk.Label(app, text="Enter Tag:", font=("Arial", 20, "bold"), bg="#2c2c2c", fg="white")
    tag_label.grid(row=4, column=0, padx=(165, 0), pady=(10, 5), sticky="w")

    tag_entry = tk.Entry(app, font=("Arial", 16), width=25, bd=2, relief="solid", fg="#333", bg="#f4f4f4")
    tag_entry.grid(row=5, column=0, padx=(120, 0), pady=(0, 10), sticky="w")


    # Video label for display
    video_label = tk.Label(app)
    video_label.grid(row=1, column=3, rowspan=10, padx=20, pady=20, sticky="n")

    # Buttons
    button_font = ("Arial", 14)

    start_btn = tk.Button(app, text="Start Video", font=button_font, width=20,
                          command=lambda: hand_open_file(tag_entry, video_label),
                          fg="white", bg="#7CFC00", activebackground="#66cc66", bd=0)
    start_btn.grid(row=6, column=0, padx=(120, 0), pady=5, sticky="w")

    exit_btn = tk.Button(app, text="Exit", font=button_font, width=20,
                         command=lambda: exit_fun(app),
                         fg="white", bg="#ff4d4d", activebackground="#e60000", bd=0)
    exit_btn.grid(row=7, column=0, padx=(120, 0), pady=(1, 20), sticky="w")

    # Configure rows/columns
    for i in range(4):
        app.columnconfigure(i, weight=1)
    for r in range(1, 12):
        app.rowconfigure(r, weight=1)

def exit_fun(window):
    global vid, result
    if vid and vid.isOpened():
        vid.release()
    if result:
        result.release()
    window.destroy()

def show_login(root):
    # Background Image
    bg_img = Image.open("mm.jpg").resize((root.winfo_screenwidth(), root.winfo_screenheight()))
    bg_photo = ImageTk.PhotoImage(bg_img)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.image = bg_photo
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Outer frame for border effect
    outer_frame = tk.Frame(root, bg="#ffffff", bd=6, relief="solid")
    outer_frame.place(relx=0.5, rely=0.5, anchor="center", width=420, height=420)

    # Inner login frame
    login = tk.Frame(outer_frame, bg="#f5f5f5")  # Neutral soft background
    login.place(relx=0.5, rely=0.5, anchor="center", width=400, height=400)

    # Title
    title_label = tk.Label(login, text="Login Form", font=("Helvetica", 24, "bold"), fg="#333", bg="#f5f5f5")
    title_label.pack(pady=30)

    # Email Entry
    tk.Label(login, text="Email", font=("Arial", 12, "bold"), bg="#f5f5f5", fg="#333").pack(pady=(10, 0))
    email_entry = tk.Entry(login, font=("Arial", 12), width=30, bd=4, relief="solid")
    email_entry.pack(pady=5)

    # Password Entry
    tk.Label(login, text="Password", font=("Arial", 12, "bold"), bg="#f5f5f5", fg="#333").pack(pady=(10, 0))
    password_entry = tk.Entry(login, font=("Arial", 12), show="*", width=30, bd=4, relief="solid")
    password_entry.pack(pady=5)

    # Forgot Password
    def forgot_password():
        messagebox.showinfo("Forgot Password", "Please contact support@example.com")

    # Validate Login
    def validate_login():
        email = email_entry.get().strip()
        password = password_entry.get().strip()
        if email == VALID_EMAIL and password == VALID_PASSWORD:
            messagebox.showinfo("Login", "Login Successful!")
            outer_frame.destroy()
            open_main_app(root)
        else:
            messagebox.showerror("Error", "Invalid email or password!")

    # Buttons
    tk.Button(login, text="Forgot password?", command=forgot_password,
              fg="blue", bg="#f5f5f5", bd=0, cursor="hand2").pack(pady=5)

    tk.Button(login, text="Login", command=validate_login,
              font=("Arial", 12), bg="#4CAF50", fg="white", width=20).pack(pady=20)

# ---------- Start Application ----------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("VISION AI")
    root.state('zoomed')             # Start in fullscreen
    root.resizable(True, True)
    root.configure(bg="#f0f4f8")

    open_main_app(root)   # 👈 Directly open main app (no login)
    root.mainloop()
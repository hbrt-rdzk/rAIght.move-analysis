from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import csv
import os

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("RepLabelApp")

        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0

        # Create UI elements
        self.canvas = tk.Canvas(root, width=1080, height=720)
        self.canvas.pack()

        self.label_video_path = tk.Label(root, text="Video Path: ", anchor='w')
        self.label_video_path.pack(side=tk.BOTTOM, fill=tk.X)

        self.toolbar = tk.Frame(root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.button_load = tk.Button(self.toolbar, text="Load Video", command=self.load_video)
        self.button_load.pack(side=tk.LEFT, padx=5, pady=5)

        self.button_previous = tk.Button(self.toolbar, text="Previous Video", command=self.load_previous_video)
        self.button_previous.pack(side=tk.LEFT, padx=5, pady=5)

        self.button_next = tk.Button(self.toolbar, text="Next Video", command=self.load_next_video)
        self.button_next.pack(side=tk.LEFT, padx=5, pady=5)

        self.total_frames_label = tk.Label(root, text="Total Frames: 0")
        self.total_frames_label.pack(side=tk.TOP, fill=tk.X)

        self.frame_number_label = tk.Label(root, text="Frame: 0")
        self.frame_number_label.pack(side=tk.TOP, fill=tk.X)

        self.button_save_start_frame = tk.Button(self.toolbar, text="Save Start Frame", command=self.save_start_frame)
        self.button_save_start_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.button_save_start_frame.config(state=tk.NORMAL)

        self.button_save_end_frame = tk.Button(self.toolbar, text="Save End Frame", command=self.save_end_frame)
        self.button_save_end_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.button_save_end_frame.config(state=tk.DISABLED)

        self.slider = ttk.Scale(root, from_=0, to=0, orient='horizontal', command=self.on_slider_change)
        self.slider.pack(fill='x')

        self.video_paths = []
        self.current_video_index = 0

    def save_start_frame(self):
        self.start_frame = self.current_frame
        messagebox.showinfo("Info", f"Start Frame saved: {self.start_frame}")
        self.button_save_start_frame.config(state=tk.DISABLED)
        self.button_save_end_frame.config(state=tk.NORMAL)
    
    def save_end_frame(self):
        self.end_frame = self.current_frame
        self.button_save_end_frame.config(state=tk.DISABLED)

        if not hasattr(self, 'start_frame'):
            messagebox.showerror("Error", "Please save start first")
            return

        video_name = os.path.basename(self.video_paths[self.current_video_index])
        csv_file = "result.csv"
        is_new_file = not os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if is_new_file:
                writer.writerow(['video', 'start_frame', 'end_frame'])
            writer.writerow([video_name, self.start_frame, self.end_frame])
        
        messagebox.showinfo("Info", f"End Frame saved: {self.end_frame}\nRepetition has been saved")
        self.button_save_start_frame.config(state=tk.NORMAL)
        self.button_save_end_frame.config(state=tk.DISABLED)

    def load_video(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.video_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.mp4', '.avi'))]
            if self.video_paths:
                self.current_video_index = 0
                self.load_current_video()
                self.button_previous.config(state=tk.DISABLED if self.current_video_index == 0 else tk.NORMAL)
                self.button_next.config(state=tk.NORMAL if len(self.video_paths) > 1 else tk.DISABLED)

    def load_current_video(self):
        self.cap = cv2.VideoCapture(self.video_paths[self.current_video_index])
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.total_frames - 1)
        self.total_frames_label.config(text=f"Total Frames: {self.total_frames}")
        self.show_frame(0)
        self.label_video_path.config(text=f"Video Path: {self.video_paths[self.current_video_index]}")

    def load_previous_video(self):
        if self.video_paths and self.current_video_index > 0:
            self.current_video_index -= 1
            self.load_current_video()
            self.button_previous.config(state=tk.DISABLED if self.current_video_index == 0 else tk.NORMAL)
            self.button_next.config(state=tk.NORMAL)
            self.slider.set(0)

    def load_next_video(self):
        if self.video_paths and self.current_video_index < len(self.video_paths) - 1:
            self.current_video_index += 1
            self.load_current_video()
            self.button_previous.config(state=tk.NORMAL)
            self.button_next.config(state=tk.DISABLED if self.current_video_index == len(self.video_paths) - 1 else tk.NORMAL)
            self.slider.set(0) 

    def show_frame(self, frame_number):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame_number + 1
                self.update_image(frame)
                self.frame_number_label.config(text=f"Frame: {self.current_frame}")

    def update_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        
        new_width = width // 3
        new_height = height // 3
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        x_center = (canvas_width - new_width) // 2
        y_center = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_center, y_center, anchor=tk.NW, image=self.photo)
        self.root.update_idletasks()

    def on_slider_change(self, value):
        frame_number = int(float(value))
        self.show_frame(frame_number)

def main():
    root = tk.Tk()
    app = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    main()

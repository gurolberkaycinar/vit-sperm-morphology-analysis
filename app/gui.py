import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Canvas, Scrollbar
from PIL import Image, ImageTk
import torch
import re
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from app.models import get_model
from pathlib import Path


class PyTorchGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Review")
        self.root.geometry("1200x600")
        
        self.main_paned = tk.PanedWindow(root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True)
        
        self.left_frame = tk.Frame(self.main_paned, width=200, bg="lightgray")
        self.main_paned.add(self.left_frame, stretch="never", minsize=200)

        self.center_right_paned = tk.PanedWindow(self.main_paned, orient="horizontal")
        self.main_paned.add(self.center_right_paned, stretch="always")

        # Center frame (scrollable thumbnails)
        self.center_frame = tk.Frame(self.center_right_paned, bg="white")
        self.center_right_paned.add(self.center_frame, stretch="always", minsize=300)

        self.right_frame = tk.Frame(self.center_right_paned, bg="white")
        self.center_right_paned.add(self.right_frame, stretch="always", minsize=400)

        self.model_button = tk.Button(self.left_frame, text="Load Model", command=self.load_model)
        self.model_button.pack(pady=10)

        self.folder_button = tk.Button(self.left_frame, text="Select Folder", command=self.select_folder)
        self.folder_button.pack(pady=10)

        self.scroll_canvas = tk.Canvas(self.center_frame)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scroll_bar = tk.Scrollbar(self.center_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_bar.pack(side="right", fill="y")
        self.scroll_canvas.configure(yscrollcommand=self.scroll_bar.set)
        self.image_frame = tk.Frame(self.scroll_canvas)
        self.scroll_canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        self.image_frame.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        self.right_frame.grid_rowconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.image_display_label = tk.Label(self.right_frame, text="Image Preview", bg="white")
        self.image_display_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.image_paths = []


    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Models", "*.pth")])
        if model_path:
            print(model_path)
            print(re.search(r'/([^/]+?)_', model_path).group(1))
            self.model = get_model(re.search(r'/([^/]+?)_', model_path).group(1))
            self.model.load_state_dict(torch.load(model_path))
            self.model = self.model.to(self.device)
            self.model.eval()
        
    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            folder = Path(folder_path)
            self.image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + \
                               list(folder.glob("*.png"))
            print(f"Found {len(self.image_paths)} images in {folder_path}")
            if self.image_paths:
                self.display_thumbnails()
            else:
                print("No images found in the selected folder.")

    def display_thumbnails(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        if not self.image_paths:
            print("No images to display.")
            return

        for idx, image_path in enumerate(self.image_paths):
            try:
                img = Image.open(image_path).resize((100, 100))
                img_tk = ImageTk.PhotoImage(img)
                lbl = Label(self.image_frame, image=img_tk)
                lbl.image = img_tk
                lbl.grid(row=idx // 5, column=idx % 5, padx=5, pady=5)
                lbl.bind("<Button-1>", lambda e, path=image_path: self.show_prediction(path))
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    def show_prediction(self, image_path):
        if not self.model:
            print("No model loaded.")
            return

        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            if hasattr(output, 'logits'):
                output = output.logits
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu()

        self.display_image_and_predictions(img, probabilities)

    def display_image_and_predictions(self, img, probabilities):
        # Display the selected image
        img_resized = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_display_label.config(image=img_tk)
        self.image_display_label.image = img_tk


        probabilities = probabilities.squeeze().detach().cpu().numpy()

        classes = ["AmorfHead", "AsymmetricNeck", "CurlyTail", "DoubleHead", "DoubleTail",
                "LongTail", "NArrowAcrosome", "Normal", "PinHead", "PyriformHead",
                "RoundHead", "ShortTail", "TaperedHead", "ThickNeck", "ThinNeck",
                "TwistedNeck", "TwistedTail", "VacuolatedHead"]
        classes = classes[:len(probabilities)]


        self.ax.clear()
        self.ax.bar(classes, probabilities, color="skyblue")
        self.ax.set_title("Class Predictions")
        self.ax.set_xlabel("Classes")
        self.ax.set_ylabel("Probability")
        self.ax.set_xticklabels(classes, rotation=45, ha='right')

        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = PyTorchGUIApp(root)
    root.mainloop()

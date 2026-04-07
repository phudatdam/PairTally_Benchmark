import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
try:
    from datasets import load_dataset
    import os
except ImportError:
    print("Error: 'datasets' library not found. Please install it using: pip install datasets")
    exit(1)

# Path Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
COCOUNT_RAW_DIR = PROJECT_ROOT / "dataset" / "CoCount-train" / "CoCount-train-raw"
DATA_DIR = COCOUNT_RAW_DIR / "data"

class CoCountRawVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CoCount-train-raw Visualizer")
        self.root.geometry("1200x900")

        # State
        self.dataset = None
        self.current_idx = 0
        self.tk_image = None
        self.pil_rendered = None

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # Top Panel
        top_frame = tk.Frame(self.root, pady=10, padx=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = tk.Label(top_frame, text="Initializing...", font=("Arial", 11), justify=tk.LEFT)
        self.lbl_info.pack(side=tk.LEFT)

        # Navigation Frame (Jump)
        jump_frame = tk.Frame(top_frame)
        jump_frame.pack(side=tk.RIGHT)
        
        tk.Label(jump_frame, text="Jump to index:").pack(side=tk.LEFT, padx=5)
        self.entry_jump = tk.Entry(jump_frame, width=10)
        self.entry_jump.pack(side=tk.LEFT, padx=5)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_index())
        tk.Button(jump_frame, text="Go", command=self.jump_to_index).pack(side=tk.LEFT)

        # Main Canvas
        self.canvas = tk.Canvas(self.root, bg="#2c3e50", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Panel
        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(bottom_frame, text="<< Previous (Left)", command=self.prev_img, width=20)
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.lbl_progress = tk.Label(bottom_frame, text="0 / 0", font=("Arial", 10, "bold"))
        self.lbl_progress.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(bottom_frame, text="Next (Right) >>", command=self.next_img, width=20)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

        # Bindings
        self.root.bind("<Left>", lambda e: self.prev_img())
        self.root.bind("<Right>", lambda e: self.next_img())
        self.canvas.bind("<Configure>", lambda e: self.render_canvas())

    def load_data(self):
        if not DATA_DIR.exists():
            messagebox.showerror("Error", f"Directory not found: {DATA_DIR}\nEnsure CoCount-train-raw is downloaded.")
            self.root.destroy()
            return

        parquet_files = [str(f.resolve()) for f in DATA_DIR.glob("*.parquet")]
        if not parquet_files:
            messagebox.showerror("Error", f"No parquet files found in {DATA_DIR}")
            self.root.destroy()
            return

        print(f"Found {len(parquet_files)} parquet files. Loading...")
        try:
            self.dataset = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
            self.display_current()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            self.root.destroy()

    def display_current(self):
        if self.dataset is None: return
        
        row = self.dataset[self.current_idx]
        img_name = row.get('image_name', 'Unknown')
        pos_cap = row.get('pos_caption', 'N/A')
        neg_cap = row.get('neg_caption', 'N/A')
        
        # Base Image
        base_img = row['image'].convert("RGB")
        draw = ImageDraw.Draw(base_img)

        # Draw Positive Exemplars (Green)
        pos_exams = row.get('positive_exemplars', [])
        for bbox in pos_exams:
            draw.rectangle(bbox, outline="#00ff00", width=3)
            
        # Draw Negative Exemplars (Red)
        neg_exams = row.get('negative_exemplars', [])
        for bbox in neg_exams:
            draw.rectangle(bbox, outline="#ff0000", width=3)

        # Draw Positive Points (Green Dots)
        pos_points = row.get('pos_points', [])
        r = 3
        for pt in pos_points:
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill="#00ff00", outline="white")

        # Draw Negative Points (Red Dots)
        neg_points = row.get('neg_points', [])
        for pt in neg_points:
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill="#ff0000", outline="white")

        self.pil_rendered = base_img
        info_text = f"Image: {img_name}\n"
        info_text += f"Pos: {pos_cap} ({len(pos_points)} pts, Green)\n"
        info_text += f"Neg: {neg_cap} ({len(neg_points)} pts, Red)"
        self.lbl_info.config(text=info_text)
        self.lbl_progress.config(text=f"IMAGE {self.current_idx + 1} OF {len(self.dataset)}")
        self.render_canvas()

    def render_canvas(self):
        if not self.pil_rendered: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 50 or ch < 50: return

        # Scale to fit while maintaining aspect ratio
        scale = min(cw / self.pil_rendered.width, ch / self.pil_rendered.height)
        nw, nh = int(self.pil_rendered.width * scale), int(self.pil_rendered.height * scale)
        
        img_resized = self.pil_rendered.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_image, anchor=tk.CENTER)

    def jump_to_index(self):
        val = self.entry_jump.get().strip()
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(self.dataset):
                self.current_idx = idx
                self.display_current()
            else:
                messagebox.showwarning("Out of range", f"Please enter a number between 1 and {len(self.dataset)}")

    def next_img(self):
        if self.current_idx < len(self.dataset) - 1:
            self.current_idx += 1
            self.display_current()

    def prev_img(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.display_current()

if __name__ == "__main__":
    root = tk.Tk()
    app = CoCountRawVisualizer(root)
    root.mainloop()
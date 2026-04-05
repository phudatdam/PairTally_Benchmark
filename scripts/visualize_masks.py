import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import tkinter as tk
from tkinter import messagebox
import json
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

# Path Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
IMAGE_DIR = PROJECT_ROOT / 'dataset' / 'processed_dataset' / 'Image'
ANNO_DIR = PROJECT_ROOT / 'dataset' / 'processed_dataset' / 'Anno'
MASK_ROOT = PROJECT_ROOT / 'dataset' / 'processed_dataset' / 'mask'
WEIRD_TXT = PROJECT_ROOT / 'dataset' / 'weird_bbox.txt'

# Colors for the 3 masks (RGBA: Red, Green, Blue with 50% transparency)
MASK_COLORS = [
    (255, 0, 0, 128),   # Mask 0: Red
    (0, 255, 0, 128),   # Mask 1: Green
    (0, 0, 255, 128),   # Mask 2: Blue
]

class MaskVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PairTally - SAM Mask Visualizer")
        self.root.geometry("1100x850")

        # State
        self.image_list = []
        self.current_idx = 0
        self.pil_rendered = None
        self.tk_image = None
        self.current_img_stem = None

        self.show_normal = tk.BooleanVar(value=True)
        self.show_weird = tk.BooleanVar(value=True)

        self.setup_ui()
        self.refresh_data()

    def setup_ui(self):
        # Top Info Panel
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        row1 = tk.Frame(control_frame)
        row1.pack(side=tk.TOP, fill=tk.X)

        self.lbl_info = tk.Label(row1, text="Scanning for masks...", font=("Arial", 12, "bold"))
        self.lbl_info.pack(side=tk.LEFT)

        self.lbl_class = tk.Label(row1, text="", font=("Arial", 12), fg="#d35400")
        self.lbl_class.pack(side=tk.LEFT, padx=30)

        self.btn_refresh = tk.Button(row1, text="Refresh List", command=self.refresh_data)
        self.btn_refresh.pack(side=tk.RIGHT)

        row2 = tk.Frame(control_frame)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        tk.Label(row2, text="Jump to:").pack(side=tk.LEFT)
        self.entry_jump = tk.Entry(row2, width=20)
        self.entry_jump.pack(side=tk.LEFT, padx=5)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_image())
        tk.Button(row2, text="Go", command=self.jump_to_image).pack(side=tk.LEFT)

        tk.Label(row2, text="Sets:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(20, 5))
        tk.Checkbutton(row2, text="Normal", variable=self.show_normal, command=self.refresh_data).pack(side=tk.LEFT)
        tk.Checkbutton(row2, text="Weird", variable=self.show_weird, command=self.refresh_data).pack(side=tk.LEFT)

        # Main Canvas
        self.canvas = tk.Canvas(self.root, bg="#2c3e50", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom Navigation
        nav_frame = tk.Frame(self.root, pady=10)
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_prev = tk.Button(nav_frame, text="<< Previous", command=self.prev_img, width=15)
        self.btn_prev.pack(side=tk.LEFT, padx=20)

        self.lbl_progress = tk.Label(nav_frame, text="0 / 0", font=("Arial", 10))
        self.lbl_progress.pack(side=tk.LEFT, expand=True)

        self.btn_next = tk.Button(nav_frame, text="Next >>", command=self.next_img, width=15)
        self.btn_next.pack(side=tk.RIGHT, padx=20)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self.handle_shortcut(self.prev_img))
        self.root.bind("<Right>", lambda e: self.handle_shortcut(self.next_img))
        self.canvas.bind("<Configure>", lambda e: self.render_canvas())

    def handle_shortcut(self, func):
        if self.root.focus_get() == self.entry_jump:
            return
        func()

    def refresh_data(self):
        """Finds only the folders in the mask directory that contain generated masks."""
        old_stem = self.current_img_stem
        weird_names = set()
        if WEIRD_TXT.exists():
            with open(WEIRD_TXT, 'r', encoding='utf-8') as f:
                weird_names = {line.strip() for line in f if line.strip()}

        if not MASK_ROOT.exists():
            self.image_list = []
        else:
            self.image_list = []
            # Only include folders that have masks and match filtering criteria
            for folder in MASK_ROOT.iterdir():
                if not folder.is_dir(): continue
                
                # Determine image and json paths
                img_path = next((IMAGE_DIR / f"{folder.name}{ext}" for ext in ['.jpg', '.jpeg', '.png'] if (IMAGE_DIR / f"{folder.name}{ext}").exists()), None)
                json_path = ANNO_DIR / f"{folder.name}.json"
                
                if img_path and json_path.exists() and any(folder.glob("mask*.png")):
                    is_weird = img_path.name in weird_names
                    if (is_weird and self.show_weird.get()) or (not is_weird and self.show_normal.get()):
                        self.image_list.append({
                            "stem": folder.name, 
                            "img": img_path, 
                            "json": json_path, 
                            "mask_dir": folder,
                            "is_weird": is_weird
                        })

        self.image_list.sort(key=lambda x: x['stem'])
        
        # Try to maintain position
        if old_stem and any(i['stem'] == old_stem for i in self.image_list):
            self.current_idx = next(i for i, x in enumerate(self.image_list) if x['stem'] == old_stem)
        else:
            self.current_idx = 0
            
        self.load_current()

    def load_current(self):
        if not self.image_list:
            self.lbl_info.config(text="No masks found yet.")
            self.lbl_class.config(text="")
            self.lbl_progress.config(text="0 / 0")
            self.current_img_stem = None
            self.canvas.delete("all")
            return

        item = self.image_list[self.current_idx]
        self.current_img_stem = item['stem']
        with open(item['json'], 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Composite Original + Masks
        base_img = Image.open(item['img']).convert("RGBA")
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        exam_bboxes = data.get('exam_bbox', [])
        for i in range(3):
            mask_path = item['mask_dir'] / f"mask{i}.png"
            if mask_path.exists():
                mask_data = Image.open(mask_path).convert("L")
                color_fill = MASK_COLORS[i]
                colored_mask = Image.new("RGBA", base_img.size, color_fill)
                overlay.paste(colored_mask, (0, 0), mask_data)
            
            if i < len(exam_bboxes):
                draw.rectangle(exam_bboxes[i], outline=MASK_COLORS[i][:3] + (255,), width=3)

        self.pil_rendered = Image.alpha_composite(base_img, overlay).convert("RGB")
        weird_status = " (WEIRD)" if item['is_weird'] else ""
        self.lbl_info.config(text=f"Viewing: {item['stem']}{weird_status}")
        self.lbl_class.config(text=f"Class: {data.get('class_name', 'Unknown')}", fg="red" if item['is_weird'] else "#d35400")
        self.lbl_progress.config(text=f"{self.current_idx + 1} / {len(self.image_list)}")
        self.render_canvas()

    def render_canvas(self):
        if not self.pil_rendered: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        scale = min(cw / self.pil_rendered.width, ch / self.pil_rendered.height)
        nw, nh = int(self.pil_rendered.width * scale), int(self.pil_rendered.height * scale)
        img_resized = self.pil_rendered.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_image, anchor=tk.CENTER)

    def jump_to_image(self):
        query = self.entry_jump.get().strip()
        if not query: return
        
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(self.image_list):
                self.current_idx = idx
                self.load_current()
                return

        for i, item in enumerate(self.image_list):
            if query.lower() in item['stem'].lower():
                self.current_idx = i
                self.load_current()
                return
        messagebox.showinfo("Not Found", f"No image found containing '{query}'")

    def next_img(self): self.current_idx = min(len(self.image_list) - 1, self.current_idx + 1); self.load_current()
    def prev_img(self): self.current_idx = max(0, self.current_idx - 1); self.load_current()

if __name__ == "__main__":
    root = tk.Tk(); app = MaskVisualizerApp(root); root.mainloop()
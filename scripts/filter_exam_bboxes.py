import tkinter as tk
from tkinter import messagebox
import json
import os
import sys
from pathlib import Path
try:
    from PIL import Image, ImageTk, ImageResampling
    LANCZOS = ImageResampling.LANCZOS
except ImportError:
    from PIL import Image, ImageTk
    LANCZOS = Image.LANCZOS
import shutil

# Configuration: Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / 'dataset' / 'CoCount-train'
IMAGE_DIR = DATASET_DIR / 'processed_dataset' / 'Image'
ANNO_DIR = DATASET_DIR / 'processed_dataset' / 'Anno'
SOURCE_ANNO_DIR = DATASET_DIR / 'processed_dataset' / 'Anno_with_exam_bbox'
LOG_FILE = DATASET_DIR / 'insufficient_exam_bboxes.txt'

class ExamBBoxFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Exam BBox Filter & Refiner")
        self.root.geometry("1200x900")

        # State
        self.current_idx = 0
        self.image_list = []
        self.pil_image = None
        self.tk_image = None
        self.scale_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0
        
        self.current_anno = []  # List of [xmin, ymin, xmax, ymax]
        self.original_anno = [] # For reset
        self.remaining_boxes = [] # List of {"bbox": [], "score": float}
        
        self.only_incomplete = tk.BooleanVar(value=False)
        self.show_overlap = tk.BooleanVar(value=True)
        self.show_remaining = tk.BooleanVar(value=False)
        self.hovered_box_id = None
        self.hovered_type = None # 'active' or 'remaining'
        
        self.active_rect_ids = {} # canvas_id -> bbox
        self.remaining_rect_ids = {} # canvas_id -> (bbox, score)

        self.setup_ui()
        self.load_file_list()
        self.load_current()

    def setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, padx=10, pady=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Status & Progress
        status_frame = tk.Frame(control_frame)
        status_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.lbl_info = tk.Label(status_frame, text="Loading...", font=("Arial", 11, "bold"))
        self.lbl_info.pack(side=tk.LEFT)
        self.lbl_stats = tk.Label(status_frame, text="Active: 0 | Remaining: 0", font=("Arial", 11))
        self.lbl_stats.pack(side=tk.RIGHT)

        # Controls Row
        row2 = tk.Frame(control_frame)
        row2.pack(side=tk.TOP, fill=tk.X)

        # Workflow Group
        wf_group = tk.LabelFrame(row2, text="Workflow", padx=5, pady=5)
        wf_group.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        
        tk.Button(wf_group, text="⟲ Reset to Original", command=self.reset_current).pack(side=tk.LEFT, padx=5)
        self.btn_save = tk.Button(wf_group, text="💾 Save & Next", bg="#27ae60", fg="white", font=("Arial", 9, "bold"), command=self.save_and_next)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Display Group
        disp_group = tk.LabelFrame(row2, text="Display Settings", padx=5, pady=5)
        disp_group.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Checkbutton(disp_group, text="Show Remaining (Dashed Magenta)", variable=self.show_remaining, command=self.redraw_all).pack(side=tk.LEFT)
        tk.Checkbutton(disp_group, text="Highlight Overlaps (>75% IoU)", variable=self.show_overlap, command=self.redraw_all).pack(side=tk.LEFT)
        tk.Checkbutton(disp_group, text="Incomplete Only (<10)", variable=self.only_incomplete, command=self.refresh_list_and_load).pack(side=tk.LEFT)

        # Navigation Group
        nav_group = tk.LabelFrame(row2, text="Navigation", padx=5, pady=5)
        nav_group.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        tk.Button(nav_group, text="⏪ Prev", command=self.prev_image).pack(side=tk.LEFT, padx=2)
        tk.Button(nav_group, text="Next ⏩", command=self.next_image).pack(side=tk.LEFT, padx=2)
        
        tk.Label(nav_group, text="Go to:").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_jump = tk.Entry(nav_group, width=10)
        self.entry_jump.pack(side=tk.LEFT, padx=2)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_image())

        tk.Button(row2, text="❓ Help", command=self.show_help).pack(side=tk.RIGHT, padx=5, pady=10)

        # Canvas
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Events
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas_frame.bind("<Configure>", lambda e: self.redraw_all())
        
        # Shortcuts
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.save_and_next())

    def load_file_list(self):
        if not ANNO_DIR.exists():
            messagebox.showerror("Error", f"Directory not found: {ANNO_DIR}")
            sys.exit(1)
        
        all_images = sorted([f.name for f in IMAGE_DIR.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if self.only_incomplete.get():
            filtered = []
            for img_name in all_images:
                stem = Path(img_name).stem
                anno_path = ANNO_DIR / f"{stem}.json"
                if anno_path.exists():
                    try:
                        with open(anno_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if len(data.get('exam_bbox', [])) < 10:
                                filtered.append(img_name)
                    except:
                        pass
                else:
                    filtered.append(img_name)
            self.image_list = filtered
        else:
            self.image_list = all_images
            
        if not self.image_list:
            self.image_list = []

    def refresh_list_and_load(self):
        old_name = self.image_list[self.current_idx] if self.image_list else None
        self.load_file_list()
        if old_name in self.image_list:
            self.current_idx = self.image_list.index(old_name)
        else:
            self.current_idx = 0
        self.load_current()

    def load_current(self):
        if not self.image_list:
            self.canvas.delete("all")
            self.lbl_info.config(text="No images match the current filter.")
            self.lbl_stats.config(text="Active: 0 | Remaining: 0")
            self.pil_image = None
            return
            
        img_name = self.image_list[self.current_idx]
        stem = Path(img_name).stem
        img_path = IMAGE_DIR / img_name
        anno_path = ANNO_DIR / f"{stem}.json"
        source_path = SOURCE_ANNO_DIR / f"{stem}.json"

        if not anno_path.exists():
            self.next_image()
            return

        # Load Image
        with Image.open(img_path) as img:
            self.pil_image = img.convert("RGB")

        # Load Active Annotations
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.current_anno = data.get('exam_bbox', [])
            self.original_anno = list(self.current_anno)
            class_name = data.get('class_name', 'Unknown')

        # Load All Potential Annotations with Scores
        self.remaining_boxes = []
        if source_path.exists():
            with open(source_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
                all_candidates = source_data.get('exam_bbox', [])
                
                # Filter out boxes already in the active set
                active_tuples = [tuple(b) for b in self.current_anno]
                for item in all_candidates:
                    if tuple(item['bbox']) not in active_tuples:
                        self.remaining_boxes.append(item)
            
            # Sort remaining by score descending
            self.remaining_boxes.sort(key=lambda x: x['score'], reverse=True)

        self.lbl_info.config(text=f"Class: {class_name} | Image: {img_name} ({self.current_idx+1}/{len(self.image_list)})")
        self.update_stats()
        self.redraw_all()

    def _calculate_iou(self, boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute IoU
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    def update_stats(self):
        count = len(self.current_anno)
        rem = len(self.remaining_boxes)
        self.lbl_stats.config(text=f"Active: {count} | Remaining: {rem}", fg="#27ae60" if count >= 10 else "#c0392b")

    def redraw_all(self):
        self.canvas.delete("all")
        if not self.pil_image: return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10: return

        # Scale and Center
        iw, ih = self.pil_image.size
        self.scale_factor = min(cw/iw, ch/ih)
        nw, nh = int(iw * self.scale_factor), int(ih * self.scale_factor)
        
        display_img = self.pil_image.resize((nw, nh), LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_img)
        
        self.x_offset = (cw - nw) / 2
        self.y_offset = (ch - nh) / 2
        self.canvas.create_image(self.x_offset, self.y_offset, image=self.tk_image, anchor=tk.NW)

        # Draw Boxes
        self.active_rect_ids = {}
        self.remaining_rect_ids = {}

        # Identify overlapping boxes
        overlapping_indices = set()
        if self.show_overlap.get():
            for i in range(len(self.current_anno)):
                for j in range(i + 1, len(self.current_anno)):
                    if self._calculate_iou(self.current_anno[i], self.current_anno[j]) > 0.75:
                        overlapping_indices.add(i)
                        overlapping_indices.add(j)

        # 1. Active BBoxes (Cyan)
        for i, bbox in enumerate(self.current_anno):
            coords = self._img_to_canvas_bbox(bbox)
            is_overlap = i in overlapping_indices
            color = "orange" if is_overlap else "cyan"
            rid = self.canvas.create_rectangle(coords, outline=color, width=2)
            self.active_rect_ids[rid] = bbox
            if is_overlap:
                self.canvas.addtag_withtag("overlap", rid)

        # 2. Remaining BBoxes (Dashed Magenta)
        if self.show_remaining.get():
            for item in self.remaining_boxes:
                bbox = item['bbox']
                coords = self._img_to_canvas_bbox(bbox)
                rid = self.canvas.create_rectangle(coords, outline="#ff00ff", width=1, dash=(4, 4))
                self.remaining_rect_ids[rid] = item

    def _img_to_canvas_bbox(self, bbox):
        x1 = bbox[0] * self.scale_factor + self.x_offset
        y1 = bbox[1] * self.scale_factor + self.y_offset
        x2 = bbox[2] * self.scale_factor + self.x_offset
        y2 = bbox[3] * self.scale_factor + self.y_offset
        return [x1, y1, x2, y2]

    def on_mouse_move(self, event):
        # Reset previous hover
        if self.hovered_box_id:
            if self.hovered_type == 'active' and self.hovered_box_id in self.active_rect_ids:
                if "overlap" in self.canvas.gettags(self.hovered_box_id):
                    self.canvas.itemconfig(self.hovered_box_id, outline="orange", width=2)
                else:
                    self.canvas.itemconfig(self.hovered_box_id, outline="cyan", width=2)
            elif self.hovered_type == 'remaining' and self.hovered_box_id in self.remaining_rect_ids:
                self.canvas.itemconfig(self.hovered_box_id, outline="#ff00ff", width=1)
        
        self.hovered_box_id = None
        
        # Check for hover on active
        for rid, bbox in self.active_rect_ids.items():
            c = self.canvas.coords(rid)
            if c[0] <= event.x <= c[2] and c[1] <= event.y <= c[3]:
                self.canvas.itemconfig(rid, outline="yellow", width=3)
                self.hovered_box_id = rid
                self.hovered_type = 'active'
                return

        # Check for hover on remaining
        if self.show_remaining.get():
            for rid, item in self.remaining_rect_ids.items():
                c = self.canvas.coords(rid)
                if c[0] <= event.x <= c[2] and c[1] <= event.y <= c[3]:
                    self.canvas.itemconfig(rid, outline="white", width=2)
                    self.hovered_box_id = rid
                    self.hovered_type = 'remaining'
                    return

    def on_left_click(self, event):
        if not self.hovered_box_id: return

        if self.hovered_type == 'active':
            bbox_to_remove = self.active_rect_ids[self.hovered_box_id]
            self.current_anno.remove(bbox_to_remove)
            
            # Log if we are now below 10 and can't refill
            if len(self.current_anno) < 10 and not self.remaining_boxes:
                self.log_insufficient()
            
            # Auto-refill from top score if below 10
            self.try_auto_refill()
            
        elif self.hovered_type == 'remaining':
            item = self.remaining_rect_ids[self.hovered_box_id]
            self.current_anno.append(item['bbox'])
            self.remaining_boxes.remove(item)

        self.hovered_box_id = None
        self.update_stats()
        self.redraw_all()

    def try_auto_refill(self):
        """If active boxes < 10, pull the best scoring remaining box."""
        while len(self.current_anno) < 10 and self.remaining_boxes:
            best_item = self.remaining_boxes.pop(0)
            self.current_anno.append(best_item['bbox'])

    def log_insufficient(self):
        img_name = self.image_list[self.current_idx]
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{img_name}\n")
        print(f"Logged {img_name} to insufficient list.")

    def reset_current(self):
        if messagebox.askyesno("Reset", "Restore original 10 boxes for this image?"):
            self.load_current()

    def save_and_next(self):
        stem = Path(self.image_list[self.current_idx]).stem
        anno_path = ANNO_DIR / f"{stem}.json"
        
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['exam_bbox'] = self.current_anno
        
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        self.next_image()

    def next_image(self):
        if self.current_idx < len(self.image_list) - 1:
            self.current_idx += 1
            self.load_current()
        else:
            messagebox.showinfo("Done", "End of dataset reached.")

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_current()

    def jump_to_image(self):
        val = self.entry_jump.get().strip()
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(self.image_list):
                self.current_idx = idx
                self.load_current()
                return
        
        for i, name in enumerate(self.image_list):
            if val.lower() in name.lower():
                self.current_idx = i
                self.load_current()
                return

    def show_help(self):
        help_text = (
            "--- Exam BBox Refiner Help ---\n\n"
            "🖱 MOUSE CONTROLS\n"
            "• Left Click (on Cyan): Remove a box. If active < 10, the next best "
            "scoring box from the source folder is automatically added.\n"
            "• Left Click (on Dashed Magenta): Choose this specific box to add to active.\n\n"
            "🛠 AUTO-REFILL LOGIC\n"
            "• The tool automatically tries to maintain 10 active boxes by pulling "
            "from the 'Anno_with_exam_bbox' pool based on SAM scores.\n"
            "• If you remove a box and no more candidates exist, the image is marked "
            "as having insufficient exemplars.\n\n"
            "⌨ SHORTCUTS\n"
            "• Space: Save current set and go to next image.\n"
            "• Arrow Left/Right: Navigate without saving."
        )
        messagebox.showinfo("Usage Guide", help_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExamBBoxFilterApp(root)
    root.mainloop()
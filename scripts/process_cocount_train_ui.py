import os
import json
import shutil
import math
from pathlib import Path
import random
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datasets import load_dataset
from collections import defaultdict

# Path Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
COCOUNT_DIR = PROJECT_ROOT / "dataset" / "CoCount-train" / "CoCount-train-raw"
OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "CoCount-train" / "processed_dataset"
OUTPUT_IMAGE = OUTPUT_ROOT / "Image"
OUTPUT_ANNO = OUTPUT_ROOT / "Anno"

class CoCountProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CoCount-train Sampler UI")
        self.root.geometry("1400x900")

        self.ds = None
        # super_code -> { pair_key -> [idx] }
        self.grouped_data = defaultdict(lambda: defaultdict(list))
        # Global selection - stores indices in the original dataset
        self.selected_indices = set()
        
        # UI State
        self.current_super = None
        self.current_pair = None
        self.thumbnails = [] # To prevent garbage collection of images
        
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # Top Panel
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_prev_pair = tk.Button(top_frame, text="< Prev Pair", command=self.prev_pair)
        self.btn_prev_pair.pack(side=tk.LEFT, padx=5)

        tk.Label(top_frame, text="Supercategory:").pack(side=tk.LEFT, padx=5)
        self.combo_super = ttk.Combobox(top_frame, state="readonly", width=15)
        self.combo_super.pack(side=tk.LEFT, padx=5)
        self.combo_super.bind("<<ComboboxSelected>>", lambda e: self.on_super_select())
        
        tk.Label(top_frame, text="Category Pair:").pack(side=tk.LEFT, padx=5)
        self.combo_pair = ttk.Combobox(top_frame, state="readonly", width=60)
        self.combo_pair.pack(side=tk.LEFT, padx=5)
        self.combo_pair.bind("<<ComboboxSelected>>", lambda e: self.show_pair_images())
        
        self.btn_next_pair = tk.Button(top_frame, text="Next Pair >", command=self.next_pair)
        self.btn_next_pair.pack(side=tk.LEFT, padx=5)

        self.btn_confirm = tk.Button(top_frame, text="Confirm & Sync Export", bg="#27ae60", fg="white", font=("Arial", 10, "bold"), command=self.export_selection)
        self.btn_confirm.pack(side=tk.RIGHT, padx=20)

        # Progress Bars Frame
        progress_frame = tk.Frame(self.root, padx=10, pady=5)
        progress_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_pair_prog = tk.Label(progress_frame, text="Pair Progress (0/0 - 0%)", font=("Arial", 9))
        self.lbl_pair_prog.pack(anchor=tk.W)
        self.bar_pair = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.bar_pair.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_super_prog = tk.Label(progress_frame, text="Supercategory Progress (0/0 - 0%)", font=("Arial", 9))
        self.lbl_super_prog.pack(anchor=tk.W)
        self.bar_super = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.bar_super.pack(fill=tk.X, pady=(0, 5))

        # Image Grid with Scrollbar
        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.container, bg="#2c3e50")
        self.scrollbar = ttk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#2c3e50")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Keyboard Shortcuts
        self.root.bind("<Left>", lambda e: self.handle_shortcut(self.prev_pair))
        self.root.bind("<Right>", lambda e: self.handle_shortcut(self.next_pair))

    def handle_shortcut(self, func):
        focused = self.root.focus_get()
        if isinstance(focused, (ttk.Combobox, tk.Entry)):
            return
        func()

    def load_data(self):
        data_dir = COCOUNT_DIR / "data"
        if not data_dir.exists():
            messagebox.showerror("Error", f"CoCount-train not found at {data_dir}.")
            return

        parquet_files = [str(f.resolve()) for f in data_dir.glob("*.parquet")]
        if not parquet_files:
            messagebox.showerror("Error", "No .parquet files found.")
            return

        print("Loading dataset and indexing unique images...")
        self.ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train")
        
        # To avoid performance hit, we only iterate over metadata
        meta_ds = self.ds.remove_columns(["image"])
        seen_images = set()
        
        for i, row in enumerate(meta_ds):
            img_name = row['image_name']
            if img_name not in seen_images:
                seen_images.add(img_name)
                
                # Determine supercategory code
                cat_field = row.get('category', 'OTR_Unknown')
                parts = cat_field.split('_')
                sc_code = parts[1] if len(parts) > 1 else "OTR"
                
                # Pair key for grouping (order independent)
                p_cap = row.get('pos_caption', 'Unknown').strip()
                n_cap = row.get('neg_caption', 'Unknown').strip()
                pair_key = tuple(sorted([p_cap, n_cap]))
                
                self.grouped_data[sc_code][pair_key].append(i)

        # Automatically choose a random 10% for each category pair
        print("Automatically selecting 10% sample...")
        for sc_code in self.grouped_data:
            for pair_key, indices in self.grouped_data[sc_code].items():
                target_count = math.ceil(len(indices) * 0.1)
                if indices:
                    selected = random.sample(indices, target_count)
                    self.selected_indices.update(selected)

        supercats = sorted(list(self.grouped_data.keys()))
        self.combo_super['values'] = supercats
        if supercats:
            self.combo_super.current(0)
            self.on_super_select()

    def next_pair(self):
        current_pair_idx = self.combo_pair.current()
        if current_pair_idx < len(self.combo_pair['values']) - 1:
            self.combo_pair.current(current_pair_idx + 1)
            self.show_pair_images()
        else:
            current_super_idx = self.combo_super.current()
            if current_super_idx < len(self.combo_super['values']) - 1:
                self.combo_super.current(current_super_idx + 1)
                self.on_super_select()

    def prev_pair(self):
        current_pair_idx = self.combo_pair.current()
        if current_pair_idx > 0:
            self.combo_pair.current(current_pair_idx - 1)
            self.show_pair_images()
        else:
            current_super_idx = self.combo_super.current()
            if current_super_idx > 0:
                self.combo_super.current(current_super_idx - 1)
                self.on_super_select()
                # Go to the last pair of the new supercategory
                self.combo_pair.current(len(self.combo_pair['values']) - 1)
                self.show_pair_images()

    def on_super_select(self):
        self.current_super = self.combo_super.get()
        pairs = sorted(list(self.grouped_data[self.current_super].keys()))
        self.combo_pair['values'] = [f"{p[0]} vs {p[1]}" for p in pairs]
        if self.combo_pair['values']:
            self.combo_pair.current(0)
            self.show_pair_images()
        self.update_progress()

    def show_pair_images(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnails = []
        
        pair_idx = self.combo_pair.current()
        pairs = sorted(list(self.grouped_data[self.current_super].keys()))
        self.current_pair = pairs[pair_idx]
        
        indices = self.grouped_data[self.current_super][self.current_pair]
        cols = 4
        for i, idx in enumerate(indices):
            row = self.ds[idx]
            cell = tk.Frame(self.scrollable_frame, bd=2, relief=tk.FLAT, bg="#34495e", width=320, height=360)
            cell.grid(row=i // cols, column=i % cols, padx=8, pady=8)
            cell.grid_propagate(False)
            
            pil_img = row['image']
            thumb = pil_img.copy()
            thumb.thumbnail((300, 250))
            tk_img = ImageTk.PhotoImage(thumb)
            self.thumbnails.append(tk_img)
            
            lbl_img = tk.Label(cell, image=tk_img, bg="#34495e")
            lbl_img.pack(pady=5)
            lbl_name = tk.Label(cell, text=row['image_name'], font=("Arial", 7), fg="#ecf0f1", bg="#34495e", wraplength=200)
            lbl_name.pack()
            
            is_sel = idx in self.selected_indices
            btn_sel = tk.Button(cell, text="Deselect" if is_sel else "Select", 
                                bg="#3498db" if is_sel else "#ecf0f1", fg="white" if is_sel else "black",
                                command=lambda _idx=idx, _cell=cell: self.toggle_selection(_idx, _cell))
            btn_sel.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.update_progress()

    def toggle_selection(self, idx, cell):
        if idx in self.selected_indices: self.selected_indices.remove(idx)
        else: self.selected_indices.add(idx)
        btn = cell.winfo_children()[-1]
        is_sel = idx in self.selected_indices
        btn.config(text="Deselect" if is_sel else "Select", bg="#3498db" if is_sel else "#ecf0f1", fg="white" if is_sel else "black")
        self.update_progress()

    def update_progress(self):
        if not self.current_super or not self.current_pair: return
        pair_indices = self.grouped_data[self.current_super][self.current_pair]
        pair_sel = sum(1 for i in pair_indices if i in self.selected_indices)
        pair_target = math.ceil(len(pair_indices) * 0.1)
        self.lbl_pair_prog.config(text=f"Pair [{self.current_pair[0]} - {self.current_pair[1]}] Progress: {pair_sel} / {pair_target} (10% of {len(pair_indices)})")
        self.bar_pair['value'] = min(100, (pair_sel / pair_target * 100) if pair_target > 0 else 0)
        super_indices = [idx for p_indices in self.grouped_data[self.current_super].values() for idx in p_indices]
        super_sel = sum(1 for i in super_indices if i in self.selected_indices)
        super_target = math.ceil(len(super_indices) * 0.1)
        self.lbl_super_prog.config(text=f"Supercategory [{self.current_super}] Progress: {super_sel} / {super_target} (10% of {len(super_indices)})")
        self.bar_super['value'] = min(100, (super_sel / super_target * 100) if super_target > 0 else 0)

    def export_selection(self):
        if not self.selected_indices: return
        if not messagebox.askyesno("Confirm Sync", f"Export {len(self.selected_indices)} scenes (making {len(self.selected_indices)*2} samples)? Unselected files in output will be removed."): return
        OUTPUT_IMAGE.mkdir(parents=True, exist_ok=True); OUTPUT_ANNO.mkdir(parents=True, exist_ok=True)
        processed = set()
        for idx in self.selected_indices:
            row = self.ds[idx]; stem = Path(row['image_name']).stem; ext = Path(row['image_name']).suffix or ".jpg"
            p_img, p_json = f"{stem}_positive{ext}", f"{stem}_positive.json"
            n_img, n_json = f"{stem}_negative{ext}", f"{stem}_negative.json"
            processed.update([p_img, p_json, n_img, n_json])
            row['image'].save(OUTPUT_IMAGE / p_img); row['image'].save(OUTPUT_IMAGE / n_img)
            with open(OUTPUT_ANNO / p_json, 'w', encoding='utf-8') as f:
                json.dump({"class_name": row['pos_caption'], "loc_bbox": [], "exam_bbox": row['positive_exemplars'], "source_img_name": row['image_name']}, f, indent=4)
            with open(OUTPUT_ANNO / n_json, 'w', encoding='utf-8') as f:
                json.dump({"class_name": row['neg_caption'], "loc_bbox": [], "exam_bbox": row['negative_exemplars'], "source_img_name": row['image_name']}, f, indent=4)
        for f in OUTPUT_IMAGE.iterdir():
            if f.name not in processed: f.unlink()
        for f in OUTPUT_ANNO.iterdir():
            if f.name not in processed: f.unlink()
        messagebox.showinfo("Complete", f"Sync complete. {len(self.selected_indices)} unique images processed.")

if __name__ == "__main__":
    root = tk.Tk(); app = CoCountProcessorApp(root); root.mainloop()
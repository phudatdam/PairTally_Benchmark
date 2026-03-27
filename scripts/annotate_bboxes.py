import tkinter as tk
from tkinter import messagebox
import json
import os
import sys
try:
    # Pillow 9.1.0+
    from PIL import Image, ImageTk, ImageResampling
    LANCZOS = ImageResampling.LANCZOS
except ImportError:
    # Old Pillow
    from PIL import Image, ImageTk
    LANCZOS = Image.LANCZOS
import random
import shutil

# Configuration: Paths relative to this script
# Assuming script is in /scripts/ and folders are in project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed_dataset', 'Image')
ANNO_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'processed_dataset', 'Anno')
PROCESSED_ROOT = os.path.join(PROJECT_ROOT, 'dataset', 'processed_dataset')
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
REMOVED_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'removed')
REMOVED_TXT = os.path.join(DATASET_ROOT, 'removed.txt')
WEIRD_TXT = os.path.join(DATASET_ROOT, 'weird_bbox.txt')

class BBoxAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PairTally Bounding Box Annotator")
        self.root.geometry("1024x768")

        # State variables
        self.current_image_index = 0
        self.image_list = []
        self.bboxes = []  # List of [xmin, ymin, xmax, ymax]
        self.current_start_point = None # (x, y)
        self.temp_point_id = None # Canvas ID for the first click dot
        self.box_ids = [] # List of Canvas IDs for drawn boxes
        self.current_anno_data = None
        self.current_img_name = None
        
        # Dragging state
        self.scale_factor = 1.0
        self.image_id = None
        self.display_width = 0
        self.display_height = 0
        self.x_offset = 0
        self.y_offset = 0
        self.pil_image = None
        self.temp_rect_id = None
        self.is_viewing_removed = False
        
        # Set visibility state
        self.show_normal = tk.BooleanVar(value=True)
        self.show_weird = tk.BooleanVar(value=True)
        self.show_removed = tk.BooleanVar(value=False)
        self.annotation_enabled = tk.BooleanVar(value=False)
        self.is_dragging = False
        self.drag_start = None
        self.hovered_box_id = None

        # Exam box tracking
        self.exam_box_data = {} # canvas_id -> bbox
        self.hovered_exam_box_id = None

        # Rectangle mode state
        self.rect_mode_var = tk.BooleanVar(value=False)
        self.rect_size = 50
        self.preview_rect_id = None
        self.last_mouse_x = None
        self.tk_image = None
        self.default_btn_bg = None
        self.last_mouse_y = None

        # Initialize GUI
        self.setup_ui()
        
        # Load file list
        self.load_file_list()
        
        # Start
        self.load_current_image()

    def setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Row 1: Info
        self.lbl_info = tk.Label(control_frame, text="Loading...", font=("Arial", 12, "bold"))
        self.lbl_info.pack(side=tk.TOP, anchor=tk.W)

        # Row 2: Drawing Controls
        row2 = tk.Frame(control_frame)
        row2.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.chk_annotate = tk.Checkbutton(row2, text="Annotate Mode", 
                                            variable=self.annotation_enabled, 
                                            command=self.update_ui_state)
        self.chk_annotate.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_progress = tk.Label(row2, text="Boxes: 0/10", font=("Arial", 12))
        self.lbl_progress.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_undo = tk.Button(row2, text="Undo", command=self.undo_click)
        self.btn_undo.pack(side=tk.LEFT, padx=5)

        self.btn_restart = tk.Button(row2, text="Restart", command=self.restart_image)
        self.btn_restart.pack(side=tk.LEFT, padx=5)

        self.chk_rect_mode = tk.Checkbutton(row2, text="Rect Mode (scroll=size)", 
                                              variable=self.rect_mode_var, 
                                              command=self.toggle_rect_mode)
        self.chk_rect_mode.pack(side=tk.LEFT, padx=5)

        self.scale_ratio = tk.Scale(row2, from_=0.2, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Ratio W/H", length=100)
        self.scale_ratio.set(1.0)
        self.scale_ratio.pack(side=tk.LEFT, padx=5)

        self.btn_confirm = tk.Button(row2, text="Confirm & Save", command=self.confirm_and_next, state=tk.DISABLED, bg="green", fg="white")
        self.btn_confirm.pack(side=tk.LEFT, padx=5)

        # Row 3: Navigation
        row3 = tk.Frame(control_frame)
        row3.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.btn_prev = tk.Button(row3, text="<< Previous", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.btn_skip = tk.Button(row3, text="Skip/Next >>", command=self.skip_image)
        self.btn_skip.pack(side=tk.LEFT, padx=5)

        tk.Label(row3, text="Jump to:").pack(side=tk.LEFT, padx=(20, 5))
        self.entry_jump = tk.Entry(row3)
        self.entry_jump.pack(side=tk.LEFT, padx=5)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_image())
        tk.Button(row3, text="Go", command=self.jump_to_image).pack(side=tk.LEFT, padx=2)

        self.btn_weird = tk.Button(row2, text="Weird BBox: NO", command=self.toggle_weird_bbox)
        self.btn_weird.pack(side=tk.LEFT, padx=(5, 5))
        self.default_btn_bg = self.btn_weird.cget('bg')

        tk.Label(row3, text="Sets:").pack(side=tk.LEFT, padx=(20, 5))
        tk.Checkbutton(row3, text="Normal", variable=self.show_normal, command=self.refresh_list_and_load).pack(side=tk.LEFT)
        tk.Checkbutton(row3, text="Weird", variable=self.show_weird, command=self.refresh_list_and_load).pack(side=tk.LEFT)
        tk.Checkbutton(row3, text="Removed", variable=self.show_removed, command=self.refresh_list_and_load).pack(side=tk.LEFT)

        self.btn_help = tk.Button(row3, text="Help", command=self.show_help)
        self.btn_help.pack(side=tk.RIGHT, padx=5)

        self.btn_remove = tk.Button(row2, text="Remove Image", command=self.remove_current_image, bg="#ff4444", fg="white")
        self.btn_remove.pack(side=tk.LEFT, padx=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#333333", cursor="tcross", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas_frame.bind("<Configure>", self.on_resize)
        self.root.bind("<Control-z>", lambda e: self.undo_click())
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Leave>", self.on_canvas_leave)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.handle_shortcut(self.prev_image))
        self.root.bind("<Right>", lambda e: self.handle_shortcut(self.skip_image))
        self.root.bind("<Up>", lambda e: self.handle_shortcut(lambda: self.change_ratio(-0.1)))
        self.root.bind("<Down>", lambda e: self.handle_shortcut(lambda: self.change_ratio(0.1)))
        self.root.bind("<space>", lambda e: self.handle_shortcut(self.confirm_and_next))
        self.root.bind("s", lambda e: self.handle_shortcut(self.toggle_rect_mode_key))
        self.root.bind("r", lambda e: self.handle_shortcut(self.restart_image))

    def on_resize(self, event):
        # Redraw canvas content on resize, but only if an image is loaded
        if self.pil_image:
            self.redraw_all()

    def _canvas_to_img_coords(self, canvas_x, canvas_y):
        if self.scale_factor == 0: return canvas_x, canvas_y
        # Adjust for image offset on canvas
        img_x = (canvas_x - self.x_offset) / self.scale_factor
        img_y = (canvas_y - self.y_offset) / self.scale_factor
        return img_x, img_y

    def _img_to_canvas_coords(self, img_x, img_y):
        # Adjust for image offset on canvas
        canvas_x = (img_x * self.scale_factor) + self.x_offset
        canvas_y = (img_y * self.scale_factor) + self.y_offset
        return canvas_x, canvas_y

    def _img_bbox_to_canvas_bbox(self, bbox):
        x1, y1 = self._img_to_canvas_coords(bbox[0], bbox[1])
        x2, y2 = self._img_to_canvas_coords(bbox[2], bbox[3])
        return [x1, y1, x2, y2]

    def redraw_all(self):
        self.canvas.delete("all")
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1 or not self.pil_image:
            return

        img_w, img_h = self.pil_image.size
        
        w_ratio = canvas_w / img_w
        h_ratio = canvas_h / img_h
        self.scale_factor = min(w_ratio, h_ratio)
        
        self.display_width = int(img_w * self.scale_factor)
        self.display_height = int(img_h * self.scale_factor)

        display_image = self.pil_image.resize((self.display_width, self.display_height), LANCZOS)
        self.tk_image = ImageTk.PhotoImage(display_image)
        
        self.x_offset = (canvas_w - self.display_width) / 2
        self.y_offset = (canvas_h - self.display_height) / 2
        
        self.image_id = self.canvas.create_image(self.x_offset, self.y_offset, image=self.tk_image, anchor=tk.NW)

        self.draw_boxes()

    def load_file_list(self):
        if not os.path.exists(IMAGE_DIR):
            messagebox.showerror("Error", f"Image directory not found: {IMAGE_DIR}")
            sys.exit(1)

        # Load current state of weird and removed images
        weird_names = set()
        if os.path.exists(WEIRD_TXT):
            with open(WEIRD_TXT, 'r', encoding='utf-8') as f:
                weird_names = {line.strip() for line in f if line.strip()}

        active_images = set(os.listdir(IMAGE_DIR))
        removed_images = set()
        removed_img_dir = os.path.join(REMOVED_DIR, 'Image')
        if os.path.exists(removed_img_dir):
            removed_images = set(os.listdir(removed_img_dir))

        final_set = set()
        if self.show_normal.get():
            # Active images that aren't marked weird
            final_set.update(active_images - weird_names)
        if self.show_weird.get():
            # Active images that ARE marked weird
            final_set.update(active_images & weird_names)
        if self.show_removed.get():
            final_set.update(removed_images)

        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.image_list = sorted([f for f in final_set if os.path.splitext(f)[1].lower() in valid_exts])

        if not self.image_list:
            self.image_list = []

    def refresh_list_and_load(self):
        old_name = self.current_img_name
        self.load_file_list()
        if old_name in self.image_list:
            self.current_image_index = self.image_list.index(old_name)
        else:
            if self.current_image_index >= len(self.image_list):
                self.current_image_index = max(0, len(self.image_list) - 1)
        self.load_current_image()

    def load_current_image(self):
        if not self.image_list:
            self.canvas.delete("all")
            self.lbl_info.config(text="No images to display in current set.")
            self.lbl_progress.config(text="Boxes: 0/10")
            self.current_img_name = None
            return

        if self.current_image_index >= len(self.image_list) or self.current_image_index < 0:
            messagebox.showinfo("Finished", "All images have been processed.")
            self.root.destroy()
            return

        self.current_img_name = self.image_list[self.current_image_index]
        
        # Determine path and set type
        image_path = os.path.join(IMAGE_DIR, self.current_img_name)
        self.is_viewing_removed = False
        if not os.path.exists(image_path):
            image_path = os.path.join(REMOVED_DIR, 'Image', self.current_img_name)
            self.is_viewing_removed = True

        json_name = os.path.splitext(self.current_img_name)[0] + ".json"
        if not self.is_viewing_removed:
            json_path = os.path.join(ANNO_DIR, json_name)
        else:
            json_path = os.path.join(REMOVED_DIR, 'Anno', json_name)

        if not os.path.exists(json_path):
            print(f"Warning: JSON not found for {self.current_img_name}, skipping.")
            self.skip_image()
            return

        # Load JSON
        with open(json_path, 'r') as f:
            self.current_anno_data = json.load(f)
        
        # Load Image
        with Image.open(image_path) as img:
            self.pil_image = img.copy()

        # Reset State
        self.bboxes = []
        self.box_ids = []
        self.current_start_point = None
        self.temp_point_id = None
        self.temp_rect_id = None
        self.is_dragging = False
        self.hovered_box_id = None
        self.exam_box_data = {}
        self.hovered_exam_box_id = None

        # Load existing annotations or annotations from pair (for reusing boxes)
        existing_bboxes = self.current_anno_data.get('loc_bbox', [])
        
        if not existing_bboxes:
            # Try to find pair image to copy boxes from
            base_name = os.path.splitext(self.current_img_name)[0]
            pair_name = None
            if base_name.endswith('_positive'):
                pair_name = base_name.replace('_positive', '_negative')
            elif base_name.endswith('_negative'):
                pair_name = base_name.replace('_negative', '_positive')
            
            if pair_name:
                pair_json_path = os.path.join(ANNO_DIR, pair_name + ".json")
                if os.path.exists(pair_json_path):
                    try:
                        with open(pair_json_path, 'r') as f:
                            pair_data = json.load(f)
                            if pair_data.get('loc_bbox'):
                                existing_bboxes = pair_data['loc_bbox']
                    except Exception as e:
                        print(f"Error loading pair JSON: {e}")

        # Store original bboxes
        self.bboxes = existing_bboxes

        self.redraw_all()
        self.update_ui_state()
        self.update_weird_button_state()
        self.canvas.focus_set()

    def draw_boxes(self):
        self.box_ids = []
        self.exam_box_data = {}
        # Draw exam_bboxes
        exam_bboxes = self.current_anno_data.get('exam_bbox', [])
        for bbox in exam_bboxes:
            scaled_bbox = self._img_bbox_to_canvas_bbox(bbox)
            rect_id = self.canvas.create_rectangle(scaled_bbox, outline="cyan", width=2)
            self.exam_box_data[rect_id] = bbox
        # Draw loc_bboxes
        for bbox in self.bboxes:
            scaled_bbox = self._img_bbox_to_canvas_bbox(bbox)
            rect_id = self.canvas.create_rectangle(scaled_bbox, outline="red", width=2)
            self.box_ids.append(rect_id)

    def handle_shortcut(self, func):
        if self.root.focus_get() == self.entry_jump:
            return
        func()

    def toggle_rect_mode_key(self):
        self.rect_mode_var.set(not self.rect_mode_var.get())
        self.toggle_rect_mode()

    def change_ratio(self, delta):
        if not self.annotation_enabled.get(): return
        val = self.scale_ratio.get()
        self.scale_ratio.set(val + delta)
        if self.rect_mode_var.get() and self.last_mouse_x is not None:
            self.update_preview_rect(self.last_mouse_x, self.last_mouse_y)
        self.update_ui_state()

    def update_ui_state(self):
        if not self.current_img_name:
            return

        is_enabled = self.annotation_enabled.get()
        class_name = self.current_anno_data.get('class_name', 'Unknown')
        self.lbl_info.config(text=f"Class: {class_name} | Image: {self.current_img_name} ({self.current_image_index + 1}/{len(self.image_list)})")
        self.lbl_progress.config(text=f"Boxes: {len(self.bboxes)}/10")
        
        if is_enabled and len(self.bboxes) == 10 and not self.is_viewing_removed:
            self.btn_confirm.config(state=tk.NORMAL)
        else:
            self.btn_confirm.config(state=tk.DISABLED)
        
        if self.is_viewing_removed:
            self.btn_remove.config(text="Restore Image", bg="blue", state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.btn_undo.config(state=tk.DISABLED)
            self.btn_restart.config(state=tk.DISABLED)
            self.btn_weird.config(state=tk.DISABLED)
            self.canvas.config(cursor="arrow")
        else:
            self.btn_remove.config(text="Remove Image", bg="#ff4444", state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.btn_undo.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.btn_restart.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.chk_rect_mode.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.scale_ratio.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.btn_weird.config(state=tk.NORMAL if is_enabled else tk.DISABLED)
            self.canvas.config(cursor="tcross" if is_enabled else "arrow")

    def on_mouse_down(self, event):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)

        # Check if click is within the image area
        if not (self.x_offset <= x_canvas < self.x_offset + self.display_width and
                self.y_offset <= y_canvas < self.y_offset + self.display_height):
            return

        if self.rect_mode_var.get():
            if len(self.bboxes) >= 10:
                return
            x_img, y_img = self._canvas_to_img_coords(x_canvas, y_canvas)

            # Apply ratio from slider
            target_ratio = self.scale_ratio.get()

            # Add randomness to aspect ratio (+/- 10% of the target ratio)
            ratio_randomness = 1.0 + (random.random() - 0.5) * 0.2
            final_ratio = target_ratio * ratio_randomness

            # Calculate width and height while preserving approximate area of rect_size^2
            w = self.rect_size * (final_ratio ** 0.5)
            h = self.rect_size / (final_ratio ** 0.5)

            half_w = w / 2
            half_h = h / 2

            self.add_new_box([x_img - half_w, y_img - half_h, x_img + half_w, y_img + half_h])
            return

        if len(self.bboxes) >= 10:
            return

        x_img, y_img = self._canvas_to_img_coords(x_canvas, y_canvas)

        if self.current_start_point is None:
            # Start of interaction (Click 1 or Start Drag)
            self.current_start_point = (x_img, y_img)
            self.drag_start = (x_canvas, y_canvas)
            self.is_dragging = False
            
            r = 3
            self.temp_point_id = self.canvas.create_oval(x_canvas-r, y_canvas-r, x_canvas+r, y_canvas+r, fill="red", outline="yellow")
        else:
            # Click 2: Complete the box immediately
            if self.temp_point_id:
                self.canvas.delete(self.temp_point_id)
                self.temp_point_id = None
            
            self.finalize_box(x_img, y_img)
            self.current_start_point = None # Interaction done

    def on_mouse_drag(self, event):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        if self.current_start_point is None:
            return

        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)

        if not self.is_dragging:
            dx = abs(x_canvas - self.drag_start[0])
            dy = abs(y_canvas - self.drag_start[1])
            if dx > 5 or dy > 5:
                self.is_dragging = True
                if self.temp_point_id:
                    self.canvas.delete(self.temp_point_id)
                    self.temp_point_id = None

        if self.is_dragging:
            if self.temp_rect_id:
                self.canvas.delete(self.temp_rect_id)
            
            ix1, iy1 = self.current_start_point
            cx1, cy1 = self._img_to_canvas_coords(ix1, iy1)
            
            self.temp_rect_id = self.canvas.create_rectangle(cx1, cy1, x_canvas, y_canvas, outline="red", width=2, dash=(2,2))

    def on_mouse_up(self, event):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        if self.current_start_point is None or not self.is_dragging:
            return

        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)

        if self.temp_rect_id:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None

        ix_end, iy_end = self._canvas_to_img_coords(x_canvas, y_canvas)
        self.finalize_box(ix_end, iy_end)
        self.current_start_point = None
        self.is_dragging = False

    def finalize_box(self, ix_end, iy_end):
        ix1, iy1 = self.current_start_point
        bbox = [min(ix1, ix_end), min(iy1, iy_end), max(ix1, ix_end), max(iy1, iy_end)]
        self.add_new_box(bbox)

    def add_new_box(self, bbox):
        scaled_bbox = self._img_bbox_to_canvas_bbox(bbox)
        rect_id = self.canvas.create_rectangle(scaled_bbox, outline="red", width=2)
        self.box_ids.append(rect_id)
        self.bboxes.append(bbox)
        self.update_ui_state()
        self.save_annotations()

    def save_annotations(self):
        if self.current_anno_data is None:
            return
        self.current_anno_data['loc_bbox'] = self.bboxes
        json_path = os.path.join(ANNO_DIR, os.path.splitext(self.current_img_name)[0] + ".json")
        try:
            with open(json_path, 'w') as f:
                json.dump(self.current_anno_data, f, indent=4)
        except IOError as e:
            print(f"Error saving annotations: {e}")

    def undo_click(self):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        if self.current_start_point is not None:
            # Undo first point
            self.canvas.delete(self.temp_point_id)
            self.temp_point_id = None
            self.current_start_point = None
        elif self.bboxes:
            # Undo last completed box, revert to first point of that box
            last_box = self.bboxes.pop()
            last_rect_id = self.box_ids.pop()
            self.canvas.delete(last_rect_id)
            self.update_ui_state()

            self.save_annotations()
        
        self.update_ui_state()
    
    def on_right_click(self, event):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)

        if self.rect_mode_var.get():
            # Check if clicked on an exam box to copy size
            for rect_id, bbox_orig in self.exam_box_data.items():
                bbox_scaled = self._img_bbox_to_canvas_bbox(bbox_orig)
                if bbox_scaled[0] <= x_canvas <= bbox_scaled[2] and bbox_scaled[1] <= y_canvas <= bbox_scaled[3]:
                    w = bbox_orig[2] - bbox_orig[0]
                    h = bbox_orig[3] - bbox_orig[1]
                    if h > 0:
                        self.rect_size = (w * h) ** 0.5
                        self.scale_ratio.set(w / h)
                        self.update_preview_rect(event.x, event.y)
                        return

        # Iterate backwards to catch top-most box first
        for i in range(len(self.bboxes) - 1, -1, -1):
            box_orig = self.bboxes[i]
            box_scaled = self._img_bbox_to_canvas_bbox(box_orig)
            if box_scaled[0] <= x_canvas <= box_scaled[2] and box_scaled[1] <= y_canvas <= box_scaled[3]:
                # Delete this box
                self.canvas.delete(self.box_ids[i])
                self.bboxes.pop(i)
                self.box_ids.pop(i)
                self.hovered_box_id = None
                self.update_ui_state()
                self.save_annotations()
                return

    def on_mouse_move(self, event):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        # Handle rect mode preview
        if self.rect_mode_var.get():
            self.update_preview_rect(event.x, event.y)

        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)

        # Reset previously hovered box
        if self.hovered_box_id:
            if self.hovered_box_id in self.box_ids: # check if still exists
                self.canvas.itemconfig(self.hovered_box_id, fill='', outline='red', width=2)
            self.hovered_box_id = None
        
        # Reset previously hovered exam box
        if self.hovered_exam_box_id:
            if self.hovered_exam_box_id in self.exam_box_data:
                self.canvas.itemconfig(self.hovered_exam_box_id, outline='cyan', width=2)
            self.hovered_exam_box_id = None

        if self.is_dragging or self.current_start_point:
            return

        # Find box under mouse
        for i in range(len(self.bboxes) - 1, -1, -1):
            box_orig = self.bboxes[i]
            box_scaled = self._img_bbox_to_canvas_bbox(box_orig)
            if box_scaled[0] <= x_canvas <= box_scaled[2] and box_scaled[1] <= y_canvas <= box_scaled[3]:
                rect_id = self.box_ids[i]
                self.canvas.itemconfig(rect_id, fill='red', stipple='gray25', outline='yellow', width=2)
                self.hovered_box_id = rect_id
                return

        # Highlight exam box if in rect mode
        if self.rect_mode_var.get():
            for rect_id, bbox_orig in self.exam_box_data.items():
                bbox_scaled = self._img_bbox_to_canvas_bbox(bbox_orig)
                if bbox_scaled[0] <= x_canvas <= bbox_scaled[2] and bbox_scaled[1] <= y_canvas <= bbox_scaled[3]:
                    self.canvas.itemconfig(rect_id, outline='magenta', width=3)
                    self.hovered_exam_box_id = rect_id
                    break

    def toggle_rect_mode(self):
        # When turning off rect mode, remove any lingering preview
        if not self.rect_mode_var.get() and self.preview_rect_id:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None
        # When turning on rect mode, cancel any pending click-click box
        if self.rect_mode_var.get() and self.current_start_point:
            self.canvas.delete(self.temp_point_id)
            self.temp_point_id = None
            self.current_start_point = None

    def on_canvas_leave(self, event):
        if self.preview_rect_id:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None

    def on_mouse_wheel(self, event):
        if not self.rect_mode_var.get() or not self.annotation_enabled.get():
            return
        
        # On Windows, event.delta is +/- 120.
        self.rect_size += (event.delta / 120) * 5
        self.rect_size = max(10, self.rect_size) # Minimum size
        self.update_preview_rect(event.x, event.y)

    def update_preview_rect(self, event_x, event_y):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        if self.preview_rect_id: self.canvas.delete(self.preview_rect_id)
        x_canvas, y_canvas = self.canvas.canvasx(event_x), self.canvas.canvasy(event_y)
        
        ratio = self.scale_ratio.get()
        w = self.rect_size * (ratio ** 0.5)
        h = self.rect_size / (ratio ** 0.5)

        w_scaled = w * self.scale_factor
        h_scaled = h * self.scale_factor
        
        half_w_scaled = w_scaled / 2
        half_h_scaled = h_scaled / 2
        self.preview_rect_id = self.canvas.create_rectangle(x_canvas - half_w_scaled, y_canvas - half_h_scaled, x_canvas + half_w_scaled, y_canvas + half_h_scaled, outline="red", width=2, dash=(3, 5))

    def confirm_and_next(self):
        if len(self.bboxes) != 10 or self.is_viewing_removed or not self.annotation_enabled.get():
            return

        # Ensure annotations are saved, especially if copied from pair and no direct edits were made
        self.save_annotations()
        self.current_image_index += 1
        self.load_current_image()

    def skip_image(self):
        self.current_image_index += 1
        self.load_current_image()
        
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
        else:
            messagebox.showinfo("Info", "This is the first image.")

    def restart_image(self):
        if self.is_viewing_removed or not self.annotation_enabled.get(): return
        self.bboxes = []
        self.box_ids = []
        self.current_start_point = None
        self.temp_point_id = None
        self.temp_rect_id = None
        self.is_dragging = False
        self.hovered_box_id = None
        
        self.redraw_all()
        
        self.update_ui_state()
        self.save_annotations()

    def remove_current_image(self):
        if not self.current_img_name or not self.annotation_enabled.get(): return
        
        is_restore = self.is_viewing_removed
        msg = f"Move '{self.current_img_name}' back to original folder?" if is_restore else f"Move '{self.current_img_name}' and its annotation to removed folder?"
        if not messagebox.askyesno("Confirm", msg): return

        # Clear current image state to ensure no locks
        self.canvas.delete("all")
        self.pil_image = None
        self.tk_image = None

        img_path = os.path.join(IMAGE_DIR, self.current_img_name)
        if is_restore:
            img_path = os.path.join(REMOVED_DIR, 'Image', self.current_img_name)
            
        json_name = os.path.splitext(self.current_img_name)[0] + ".json"
        json_path = os.path.join(ANNO_DIR, json_name)
        if is_restore:
            json_path = os.path.join(REMOVED_DIR, 'Anno', json_name)

        dest_img_dir = os.path.join(REMOVED_DIR, 'Image')
        dest_anno_dir = os.path.join(REMOVED_DIR, 'Anno')
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_anno_dir, exist_ok=True)

        try:
            if is_restore:
                # Move back from removed to original
                shutil.move(img_path, os.path.join(IMAGE_DIR, self.current_img_name))
                shutil.move(json_path, os.path.join(ANNO_DIR, json_name))
                # Clean up removed.txt
                if os.path.exists(REMOVED_TXT):
                    with open(REMOVED_TXT, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    with open(REMOVED_TXT, 'w', encoding='utf-8') as f:
                        for line in lines:
                            if line.strip() != self.current_img_name:
                                f.write(line)
            else:
                # Move to removed
                if os.path.exists(img_path):
                    shutil.move(img_path, os.path.join(dest_img_dir, self.current_img_name))
                if os.path.exists(json_path):
                    shutil.move(json_path, os.path.join(dest_anno_dir, json_name))
                with open(REMOVED_TXT, 'a', encoding='utf-8') as f:
                    f.write(self.current_img_name + "\n")
            
            self.refresh_list_and_load()
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed: {e}")
            self.refresh_list_and_load()

    def toggle_weird_bbox(self):
        if not self.current_img_name or not self.annotation_enabled.get(): return
        
        weird_files = set()
        if os.path.exists(WEIRD_TXT):
            with open(WEIRD_TXT, 'r', encoding='utf-8') as f:
                weird_files = {line.strip() for line in f if line.strip()}
        
        if self.current_img_name in weird_files:
            weird_files.remove(self.current_img_name)
        else:
            weird_files.add(self.current_img_name)
        
        with open(WEIRD_TXT, 'w', encoding='utf-8') as f:
            for item in sorted(list(weird_files)):
                f.write(f"{item}\n")
        
        self.update_weird_button_state()

    def update_weird_button_state(self):
        if not self.current_img_name: return
        weird_files = set()
        if os.path.exists(WEIRD_TXT):
            with open(WEIRD_TXT, 'r', encoding='utf-8') as f:
                weird_files = {line.strip() for line in f if line.strip()}
        
        if self.current_img_name in weird_files:
            self.btn_weird.config(text="Weird BBox: YES", bg="orange")
        else:
            self.btn_weird.config(text="Weird BBox: NO", bg=self.default_btn_bg)

    def show_help(self):
        help_text = (
            "Mouse Controls:\n"
            "- Left Click (Drag or Click-Click): Draw a bounding box.\n"
            "- Right Click: Delete a box / (Rect Mode) Copy size from existing box (Cyan).\n"
            "- Mouse Wheel: (Rect Mode) Change rectangle size.\n\n"
            "Keyboard Shortcuts:\n"
            "- Left / Right Arrow: Previous / Skip Image.\n"
            "- Space: Confirm & Save (requires 10 boxes).\n"
            "- 'S': Toggle Rectangle Mode.\n"
            "- 'R': Restart current image (Clear all boxes).\n"
            "- Ctrl+Z: Undo last point/box.\n"
            "- Up / Down Arrow: Increase / Decrease aspect ratio.\n\n"
            "Note: Enable 'Annotate Mode' to perform any actions that modify annotations."
        )
        messagebox.showinfo("Help - Usage Guide", help_text)

    def jump_to_image(self):
        query = self.entry_jump.get().strip()
        if not query: return
        
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(self.image_list):
                self.current_image_index = idx
                self.load_current_image()
                return

        for i, name in enumerate(self.image_list):
            if query.lower() in name.lower():
                self.current_image_index = i
                self.load_current_image()
                return
        messagebox.showinfo("Not Found", f"No image found containing '{query}'")

if __name__ == "__main__":
    root = tk.Tk()
    app = BBoxAnnotationApp(root)
    root.mainloop()
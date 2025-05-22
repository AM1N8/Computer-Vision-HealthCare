import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class XRayAnomalyDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Chest X-Ray Anomaly Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.model_path = None
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """Configure custom styles for modern appearance"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', 
                           font=('Arial', 20, 'bold'), 
                           background='#f0f0f0',
                           foreground='#2c3e50')
        
        self.style.configure('Subtitle.TLabel', 
                           font=('Arial', 12), 
                           background='#f0f0f0',
                           foreground='#34495e')
        
        self.style.configure('Modern.TButton',
                           font=('Arial', 10, 'bold'),
                           padding=(20, 10))
        
        self.style.configure('Status.TLabel',
                           font=('Arial', 10),
                           background='#f0f0f0',
                           foreground='#27ae60')
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        ttk.Label(title_frame, text="Chest X-Ray Anomaly Detection", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="Upload chest X-ray images to detect anomalies using YOLOv8", 
                 style='Subtitle.TLabel').pack(pady=(5, 0))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="15")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Model loading section
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(model_frame, text="📁 Load Model", 
                  command=self.load_model, 
                  style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_status = ttk.Label(model_frame, text="No model loaded", 
                                     style='Subtitle.TLabel')
        self.model_status.pack(side=tk.LEFT)
        
        # Image processing section
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(fill=tk.X)
        
        ttk.Button(image_frame, text="📷 Upload Image", 
                  command=self.upload_image, 
                  style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(image_frame, text="🔍 Detect Anomalies", 
                  command=self.detect_anomalies, 
                  style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(image_frame, text="💾 Save Result", 
                  command=self.save_result, 
                  style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(image_frame, text="🔄 Clear", 
                  command=self.clear_images, 
                  style='Modern.TButton').pack(side=tk.LEFT)
        
        # Image display area
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Original image panel
        original_frame = ttk.LabelFrame(display_frame, text="Original Image", padding="10")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        original_frame.rowconfigure(0, weight=1)
        original_frame.columnconfigure(0, weight=1)
        
        self.original_canvas = tk.Canvas(original_frame, bg='white', width=400, height=400)
        self.original_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for original image
        original_v_scroll = ttk.Scrollbar(original_frame, orient=tk.VERTICAL, command=self.original_canvas.yview)
        original_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        original_h_scroll = ttk.Scrollbar(original_frame, orient=tk.HORIZONTAL, command=self.original_canvas.xview)
        original_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.original_canvas.configure(yscrollcommand=original_v_scroll.set, xscrollcommand=original_h_scroll.set)
        
        # Detection result panel
        result_frame = ttk.LabelFrame(display_frame, text="Detection Results", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(result_frame, bg='white', width=400, height=400)
        self.result_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for result image
        result_v_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_canvas.yview)
        result_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        result_h_scroll = ttk.Scrollbar(result_frame, orient=tk.HORIZONTAL, command=self.result_canvas.xview)
        result_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.result_canvas.configure(yscrollcommand=result_v_scroll.set, xscrollcommand=result_h_scroll.set)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Please load a model and upload an image")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              style='Status.TLabel', relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
    
    def load_model(self):
        """Load the trained YOLOv8 model"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select YOLOv8 Model File",
                filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
            )
            
            if file_path:
                self.status_var.set("Loading model...")
                self.root.update()
                
                self.model = YOLO(file_path)
                self.model_path = file_path
                
                model_name = os.path.basename(file_path)
                self.model_status.config(text=f"✅ Model loaded: {model_name}")
                self.status_var.set(f"Model loaded successfully: {model_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.status_var.set("Error loading model")
    
    def upload_image(self):
        """Upload and display chest X-ray image"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Chest X-Ray Image",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.current_image_path = file_path
                
                # Load and display original image
                image = Image.open(file_path)
                self.current_image = image.copy()
                
                # Resize image for display while maintaining aspect ratio
                display_image = self.resize_image_for_display(image, 400, 400)
                photo = ImageTk.PhotoImage(display_image)
                
                # Clear and update canvas
                self.original_canvas.delete("all")
                self.original_canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
                self.original_canvas.image = photo  # Keep a reference
                
                # Update scroll region
                self.original_canvas.configure(scrollregion=self.original_canvas.bbox("all"))
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
            self.status_var.set("Error loading image")
    
    def detect_anomalies(self):
        """Run anomaly detection on the uploaded image"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        if not self.current_image:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            self.status_var.set("Detecting anomalies...")
            self.root.update()
            
            # Run inference
            results = self.model(self.current_image_path)
            
            # Process results
            if len(results) > 0:
                result = results[0]
                
                # Get the original image as numpy array
                img_array = cv2.imread(self.current_image_path)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
                # Draw bounding boxes and labels
                annotated_img = img_rgb.copy()
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []
                    
                    # Get class names
                    class_names = self.model.names if hasattr(self.model, 'names') else {}
                    
                    detection_count = 0
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        if conf > 0.5:  # Confidence threshold
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Prepare label
                            class_id = int(classes[i]) if i < len(classes) else 0
                            class_name = class_names.get(class_id, 'Anomaly')
                            label = f"{class_name}: {conf:.2f}"
                            
                            # Draw label background
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (255, 0, 0), -1)
                            
                            # Draw label text
                            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            detection_count += 1
                    
                    status_msg = f"Detection complete - {detection_count} anomalies found"
                else:
                    status_msg = "Detection complete - No anomalies detected"
                
                # Convert back to PIL Image and display
                self.processed_image = Image.fromarray(annotated_img)
                display_image = self.resize_image_for_display(self.processed_image, 400, 400)
                photo = ImageTk.PhotoImage(display_image)
                
                # Clear and update result canvas
                self.result_canvas.delete("all")
                self.result_canvas.create_image(200, 200, image=photo, anchor=tk.CENTER)
                self.result_canvas.image = photo  # Keep a reference
                
                # Update scroll region
                self.result_canvas.configure(scrollregion=self.result_canvas.bbox("all"))
                
                self.status_var.set(status_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed:\n{str(e)}")
            self.status_var.set("Detection failed")
    
    def save_result(self):
        """Save the detection result image"""
        if not self.processed_image:
            messagebox.showwarning("Warning", "No detection result to save!")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Detection Result",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.processed_image.save(file_path)
                self.status_var.set(f"Result saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Detection result saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save result:\n{str(e)}")
    
    def clear_images(self):
        """Clear all images from the interface"""
        self.original_canvas.delete("all")
        self.result_canvas.delete("all")
        self.current_image = None
        self.current_image_path = None
        self.processed_image = None
        self.status_var.set("Images cleared - Ready for new input")
    
    def resize_image_for_display(self, image, max_width, max_height):
        """Resize image for display while maintaining aspect ratio"""
        img_width, img_height = image.size
        
        # Calculate scaling factor
        scale_w = max_width / img_width
        scale_h = max_height / img_height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = XRayAnomalyDetector(root)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
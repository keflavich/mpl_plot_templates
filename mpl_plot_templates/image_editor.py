"""
Image Editor GUI Application

A graphical user interface for editing images with HSV, contrast, brightness, and alpha controls.
This module provides a tkinter-based GUI that allows users to load, edit, and save images with
various adjustments.

Example usage:

    # Basic usage (opens empty editor)
    import tkinter as tk
    from mpl_plot_templates.image_editor import ImageEditor

    root = tk.Tk()
    app = ImageEditor(root)
    root.mainloop()

    # Initialize with an image file
    root = tk.Tk()
    app = ImageEditor(root, initial_image="path/to/image.jpg")
    root.mainloop()

    # Initialize with a numpy array
    import numpy as np
    image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)  # Example RGB image
    root = tk.Tk()
    app = ImageEditor(root, initial_image=image_array)
    root.mainloop()

Note: When using a numpy array as input, it must be in RGB format (3 channels) with shape (height, width, 3).
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageEditor:
    """
    A GUI application for editing images with HSV, contrast, brightness, and alpha controls.

    This class provides a graphical interface for loading images and applying various
    adjustments including hue rotation, saturation, value, contrast, brightness, and
    transparency modifications.

    Attributes:
        root (tk.Tk): The main tkinter window
        original_image (numpy.ndarray): The unmodified input image
        current_image (numpy.ndarray): The currently modified image
        photo (ImageTk.PhotoImage): The image displayed in the canvas
    """

    def __init__(self, root, initial_image=None):
        """
        Initialize the ImageEditor application.

        Args:
            root (tk.Tk): The main tkinter window
            initial_image (str or numpy.ndarray, optional): Initial image to load. Can be either:
                - A string path to an image file
                - A numpy array containing the image data (in RGB format)
        """
        self.root = root
        self.root.title("Image Editor")

        # Initialize variables
        self.original_image = None
        self.current_image = None
        self.photo = None

        # Initialize slider variables for global adjustments
        self.hue_var = tk.DoubleVar(value=0)
        self.saturation_var = tk.DoubleVar(value=0)
        self.value_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=0)
        self.brightness_var = tk.DoubleVar(value=0)
        self.alpha_var = tk.DoubleVar(value=100)

        # RGB channel HSV rotation variables
        self.r_hsv_var = tk.DoubleVar(value=0)
        self.g_hsv_var = tk.DoubleVar(value=0)
        self.b_hsv_var = tk.DoubleVar(value=0)

        # RGB channel contrast variables
        self.r_contrast_var = tk.DoubleVar(value=0)
        self.g_contrast_var = tk.DoubleVar(value=0)
        self.b_contrast_var = tk.DoubleVar(value=0)

        # RGB channel brightness variables
        self.r_brightness_var = tk.DoubleVar(value=0)
        self.g_brightness_var = tk.DoubleVar(value=0)
        self.b_brightness_var = tk.DoubleVar(value=0)

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create a left frame for controls
        self.left_frame = ttk.Frame(self.main_frame, padding="5")
        self.left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=5, pady=5)

        # Create controls frame
        self.controls_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding="5")
        self.controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), padx=5, pady=5)

        # Create buttons frame
        self.buttons_frame = ttk.Frame(self.left_frame)
        self.buttons_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create buttons
        ttk.Button(self.buttons_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Save Image", command=self.save_image).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.buttons_frame, text="Export Code", command=self.export_code).grid(row=0, column=2, padx=5, pady=5)

        # Create image display area (now on the right)
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600, bg="#f0f0f0")
        self.canvas.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create section for global HSV controls
        self.global_frame = ttk.LabelFrame(self.controls_frame, text="Global Adjustments", padding="5")
        self.global_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create section for Red channel adjustments
        self.red_frame = ttk.LabelFrame(self.controls_frame, text="Red Channel Adjustments", padding="5")
        self.red_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create section for Green channel adjustments
        self.green_frame = ttk.LabelFrame(self.controls_frame, text="Green Channel Adjustments", padding="5")
        self.green_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create section for Blue channel adjustments
        self.blue_frame = ttk.LabelFrame(self.controls_frame, text="Blue Channel Adjustments", padding="5")
        self.blue_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create section for opacity/alpha
        self.opacity_frame = ttk.LabelFrame(self.controls_frame, text="Opacity", padding="5")
        self.opacity_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        # Create global adjustment sliders
        self.create_slider("Hue", -180, 180, 0, 0, self.global_frame)
        self.create_slider("Saturation", -100, 100, 1, 0, self.global_frame)
        self.create_slider("Value", -100, 100, 2, 0, self.global_frame)
        self.create_slider("Contrast", -100, 100, 3, 0, self.global_frame)
        self.create_slider("Brightness", -100, 100, 4, 0, self.global_frame)

        # Add reset button for global adjustments
        reset_global_frame = ttk.Frame(self.global_frame)
        reset_global_frame.grid(row=5, column=0, sticky=(tk.E, tk.W), padx=5, pady=5)
        ttk.Button(reset_global_frame, text="Reset Global", command=self.reset_global).grid(row=0, column=0, padx=5, pady=5)

        # Create Red channel adjustment sliders
        self.create_slider("Hue", -90, 90, 0, 0, self.red_frame, var_name="r_hsv_var")
        self.create_slider("Contrast", -100, 100, 1, 0, self.red_frame, var_name="r_contrast_var")
        self.create_slider("Brightness", -100, 100, 2, 0, self.red_frame, var_name="r_brightness_var")

        # Add reset button for red channel
        reset_red_frame = ttk.Frame(self.red_frame)
        reset_red_frame.grid(row=3, column=0, sticky=(tk.E, tk.W), padx=5, pady=5)
        ttk.Button(reset_red_frame, text="Reset Red Channel", command=self.reset_red).grid(row=0, column=0, padx=5, pady=5)

        # Create Green channel adjustment sliders
        self.create_slider("Hue", -90, 90, 0, 0, self.green_frame, var_name="g_hsv_var")
        self.create_slider("Contrast", -100, 100, 1, 0, self.green_frame, var_name="g_contrast_var")
        self.create_slider("Brightness", -100, 100, 2, 0, self.green_frame, var_name="g_brightness_var")

        # Add reset button for green channel
        reset_green_frame = ttk.Frame(self.green_frame)
        reset_green_frame.grid(row=3, column=0, sticky=(tk.E, tk.W), padx=5, pady=5)
        ttk.Button(reset_green_frame, text="Reset Green Channel", command=self.reset_green).grid(row=0, column=0, padx=5, pady=5)

        # Create Blue channel adjustment sliders
        self.create_slider("Hue", -90, 90, 0, 0, self.blue_frame, var_name="b_hsv_var")
        self.create_slider("Contrast", -100, 100, 1, 0, self.blue_frame, var_name="b_contrast_var")
        self.create_slider("Brightness", -100, 100, 2, 0, self.blue_frame, var_name="b_brightness_var")

        # Add reset button for blue channel
        reset_blue_frame = ttk.Frame(self.blue_frame)
        reset_blue_frame.grid(row=3, column=0, sticky=(tk.E, tk.W), padx=5, pady=5)
        ttk.Button(reset_blue_frame, text="Reset Blue Channel", command=self.reset_blue).grid(row=0, column=0, padx=5, pady=5)

        # Create alpha/opacity slider
        self.create_slider("Alpha", 0, 100, 0, 0, self.opacity_frame)

        # Add reset all button to opacity frame
        reset_all_frame = ttk.Frame(self.opacity_frame)
        reset_all_frame.grid(row=1, column=0, sticky=(tk.E, tk.W), padx=5, pady=5)
        ttk.Button(reset_all_frame, text="Reset All", command=self.reset_all).grid(row=0, column=0, padx=5, pady=5)

        # Bind slider events
        self.bind_slider_events()

        # Make window resizable
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=4)  # Give more weight to canvas
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Load initial image if provided
        if initial_image is not None:
            try:
                if isinstance(initial_image, str):
                    self.original_image = cv2.imread(initial_image)
                    if self.original_image is None:
                        messagebox.showerror("Error", f"Failed to load image from {initial_image}")
                        return
                    self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                elif isinstance(initial_image, np.ndarray):
                    if len(initial_image.shape) == 3:  # Check if it's a color image
                        if initial_image.shape[2] == 3:  # Check if it has 3 channels
                            self.original_image = initial_image.copy()
                        else:
                            raise ValueError("Image array must have 3 color channels (RGB)")
                    else:
                        raise ValueError("Image array must be a 3D array (height, width, channels)")
                else:
                    raise TypeError("Initial image must be either a file path or a numpy array")

                if self.original_image is not None:
                    self.current_image = self.original_image.copy()
                    self.update_image()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading initial image: {str(e)}")

    def create_slider(self, name, from_, to, row, column, parent_frame, var_name=None):
        """
        Create a labeled slider control.

        Args:
            name (str): The name/label of the slider
            from_ (float): The minimum value of the slider
            to (float): The maximum value of the slider
            row (int): The row position in the grid
            column (int): The column position in the grid (unused in vertical layout)
            parent_frame (ttk.Frame): The parent frame for the slider
            var_name (str, optional): Custom variable name to use. If None, uses lowercase name + '_var'

        Returns:
            ttk.Scale: The created slider widget
        """
        # Create a frame for each slider group
        slider_frame = ttk.Frame(parent_frame)
        slider_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)

        # Add label
        ttk.Label(slider_frame, text=f"{name}:", width=10).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

        # Get the appropriate variable based on the slider name
        if var_name is None:
            var_name = f"{name.lower()}_var"
        var = getattr(self, var_name)

        # Add slider
        slider = ttk.Scale(
            slider_frame,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            length=300,
            variable=var
        )
        slider.grid(row=0, column=1, padx=5, pady=2, sticky=(tk.W, tk.E))

        # Add value label
        value_label = ttk.Label(slider_frame, text="0")
        value_label.grid(row=0, column=2, padx=5, pady=2)

        # Setup dynamic updating of value label
        def update_label(*args):
            value_label.configure(text=f"{var.get():.1f}")

        var.trace_add("write", update_label)
        update_label()  # Initialize with current value

        # Make slider expandable
        slider_frame.columnconfigure(1, weight=1)

        return slider

    def bind_slider_events(self):
        """
        Bind the update_image method to all slider events.
        This ensures the image updates whenever any slider is moved.
        """
        # Find all frames that might contain sliders
        slider_container_frames = [
            self.global_frame,
            self.red_frame,
            self.green_frame,
            self.blue_frame,
            self.opacity_frame
        ]

        # For each container, find all slider frames then the actual sliders
        for container in slider_container_frames:
            for frame in container.winfo_children():
                if isinstance(frame, ttk.Frame):
                    # Find the slider in this frame
                    for child in frame.winfo_children():
                        if isinstance(child, ttk.Scale):
                            # Bind events to this slider
                            child.bind("<B1-Motion>", self.update_image)
                            child.bind("<ButtonRelease-1>", self.update_image)

        # Also bind the slider variables to update_image to catch programmatic changes
        def on_var_change(*args):
            self.update_image()

        var_names = [
            "hue_var", "saturation_var", "value_var",
            "contrast_var", "brightness_var", "alpha_var",
            "r_hsv_var", "g_hsv_var", "b_hsv_var",
            "r_contrast_var", "r_brightness_var",
            "g_contrast_var", "g_brightness_var",
            "b_contrast_var", "b_brightness_var"
        ]

        for var_name in var_names:
            var = getattr(self, var_name)
            var.trace_add("write", on_var_change)

    def load_image(self):
        """
        Open a file dialog to select and load an image file.
        The loaded image is converted from BGR to RGB color space.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("Error", f"Failed to load image from {file_path}")
                    return

                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.current_image = self.original_image.copy()

                # Reset global adjustment sliders to default values
                self.hue_var.set(0)
                self.saturation_var.set(0)
                self.value_var.set(0)
                self.contrast_var.set(0)
                self.brightness_var.set(0)
                self.alpha_var.set(100)

                # Reset RGB channel HSV sliders
                self.r_hsv_var.set(0)
                self.g_hsv_var.set(0)
                self.b_hsv_var.set(0)

                # Reset RGB channel contrast and brightness sliders
                self.r_contrast_var.set(0)
                self.g_contrast_var.set(0)
                self.b_contrast_var.set(0)
                self.r_brightness_var.set(0)
                self.g_brightness_var.set(0)
                self.b_brightness_var.set(0)

                self.update_image()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def save_image(self):
        """
        Open a file dialog to save the current modified image.
        The image is converted from RGB to BGR before saving.
        """
        if self.current_image is None:
            messagebox.showinfo("Info", "No image to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving image: {str(e)}")

    def update_image(self, *args):
        """
        Update the image based on current slider values.
        Applies HSV adjustments, contrast, brightness, and alpha modifications.
        Also applies per-channel HSV rotation, contrast, and brightness.

        Args:
            *args: Variable length argument list for event handling
                (can be an event from slider interaction or variable, name, mode from trace)
        """
        if self.original_image is None:
            return

        # Get current values from global adjustment sliders
        hue = self.hue_var.get()
        saturation = self.saturation_var.get()
        value = self.value_var.get()
        contrast = self.contrast_var.get()
        brightness = self.brightness_var.get()
        alpha = self.alpha_var.get() / 100.0

        # Get RGB channel adjustment values
        # HSV rotation
        r_hsv = self.r_hsv_var.get()
        g_hsv = self.g_hsv_var.get()
        b_hsv = self.b_hsv_var.get()

        # Contrast
        r_contrast = self.r_contrast_var.get()
        g_contrast = self.g_contrast_var.get()
        b_contrast = self.b_contrast_var.get()

        # Brightness
        r_brightness = self.r_brightness_var.get()
        g_brightness = self.g_brightness_var.get()
        b_brightness = self.b_brightness_var.get()

        # Check if per-channel processing is needed
        needs_channel_processing = (
            r_hsv != 0 or g_hsv != 0 or b_hsv != 0 or
            r_contrast != 0 or g_contrast != 0 or b_contrast != 0 or
            r_brightness != 0 or g_brightness != 0 or b_brightness != 0
        )

        # If no per-channel processing is needed, use the original pipeline
        if not needs_channel_processing:
            # Convert to HSV
            hsv = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Apply HSV adjustments
            hsv[:,:,0] = (hsv[:,:,0] + hue) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + saturation/100), 0, 255)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + value/100), 0, 255)

            # Convert back to RGB
            modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Apply global contrast and brightness
            modified = cv2.convertScaleAbs(modified, alpha=1 + contrast/100, beta=brightness)
        else:
            # Start with the original image
            result = self.original_image.copy()

            # Process HSV rotation for each channel
            if r_hsv != 0 or g_hsv != 0 or b_hsv != 0:
                # Split the original image
                r, g, b = cv2.split(self.original_image)

                # Create masks for isolating each channel
                r_mask = np.zeros_like(self.original_image)
                g_mask = np.zeros_like(self.original_image)
                b_mask = np.zeros_like(self.original_image)

                # Copy each channel to its appropriate position
                r_mask[:,:,0] = r
                g_mask[:,:,1] = g
                b_mask[:,:,2] = b

                # Process red channel if needed
                if r_hsv != 0:
                    # Convert red mask to HSV and rotate hue
                    r_hsv_img = cv2.cvtColor(r_mask, cv2.COLOR_RGB2HSV)
                    r_hsv_img[:,:,0] = (r_hsv_img[:,:,0] + r_hsv) % 180

                    # Convert back to RGB
                    r_rotated = cv2.cvtColor(r_hsv_img, cv2.COLOR_HSV2RGB)

                    # Add the rotated red channel to the result
                    result = cv2.add(result - r_mask, r_rotated)

                # Process green channel if needed
                if g_hsv != 0:
                    # Convert green mask to HSV and rotate hue
                    g_hsv_img = cv2.cvtColor(g_mask, cv2.COLOR_RGB2HSV)
                    g_hsv_img[:,:,0] = (g_hsv_img[:,:,0] + g_hsv) % 180

                    # Convert back to RGB
                    g_rotated = cv2.cvtColor(g_hsv_img, cv2.COLOR_HSV2RGB)

                    # Add the rotated green channel to the result
                    result = cv2.add(result - g_mask, g_rotated)

                # Process blue channel if needed
                if b_hsv != 0:
                    # Convert blue mask to HSV and rotate hue
                    b_hsv_img = cv2.cvtColor(b_mask, cv2.COLOR_RGB2HSV)
                    b_hsv_img[:,:,0] = (b_hsv_img[:,:,0] + b_hsv) % 180

                    # Convert back to RGB
                    b_rotated = cv2.cvtColor(b_hsv_img, cv2.COLOR_HSV2RGB)

                    # Add the rotated blue channel to the result
                    result = cv2.add(result - b_mask, b_rotated)

            # Apply per-channel contrast and brightness
            if (r_contrast != 0 or r_brightness != 0 or
                g_contrast != 0 or g_brightness != 0 or
                b_contrast != 0 or b_brightness != 0):

                # Split the image into channels
                r, g, b = cv2.split(result)

                # Apply contrast and brightness to each channel
                if r_contrast != 0 or r_brightness != 0:
                    contrast_factor = 1 + r_contrast/100
                    r = cv2.convertScaleAbs(r, alpha=contrast_factor, beta=r_brightness)

                if g_contrast != 0 or g_brightness != 0:
                    contrast_factor = 1 + g_contrast/100
                    g = cv2.convertScaleAbs(g, alpha=contrast_factor, beta=g_brightness)

                if b_contrast != 0 or b_brightness != 0:
                    contrast_factor = 1 + b_contrast/100
                    b = cv2.convertScaleAbs(b, alpha=contrast_factor, beta=b_brightness)

                # Merge the channels back
                result = cv2.merge([r, g, b])

            # Now apply global HSV processing
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Apply global HSV adjustments
            hsv[:,:,0] = (hsv[:,:,0] + hue) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + saturation/100), 0, 255)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + value/100), 0, 255)

            # Convert back to RGB
            modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Apply global contrast and brightness
            modified = cv2.convertScaleAbs(modified, alpha=1 + contrast/100, beta=brightness)

        # Apply alpha
        if alpha < 1.0:
            modified = cv2.addWeighted(self.original_image, 1-alpha, modified, alpha, 0)

        self.current_image = modified

        # Update display
        self.display_image()

    def display_image(self):
        """
        Display the current image on the canvas.
        The image is automatically resized to fit the canvas while maintaining aspect ratio.
        """
        if self.current_image is not None:
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # If canvas hasn't been drawn yet, use the configured size
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800  # Default width
                canvas_height = 600  # Default height

            img_height, img_width = self.current_image.shape[:2]
            scale = min(canvas_width/img_width, canvas_height/img_height)

            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            resized = cv2.resize(self.current_image, (new_width, new_height))

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))

            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width//2, canvas_height//2,
                image=self.photo, anchor=tk.CENTER
            )

    def _on_canvas_resize(self, event):
        """Handle canvas resize event by redisplaying the image"""
        if self.current_image is not None:
            # Use a delay to prevent excessive redraws during resize
            if hasattr(self, '_resize_timer'):
                self.root.after_cancel(self._resize_timer)
            self._resize_timer = self.root.after(100, self.display_image)

    def reset_global(self):
        """Reset all global adjustment sliders to default values"""
        self.hue_var.set(0)
        self.saturation_var.set(0)
        self.value_var.set(0)
        self.contrast_var.set(0)
        self.brightness_var.set(0)
        self.update_image()

    def reset_red(self):
        """Reset all Red channel adjustment sliders to default values"""
        self.r_hsv_var.set(0)
        self.r_contrast_var.set(0)
        self.r_brightness_var.set(0)
        self.update_image()

    def reset_green(self):
        """Reset all Green channel adjustment sliders to default values"""
        self.g_hsv_var.set(0)
        self.g_contrast_var.set(0)
        self.g_brightness_var.set(0)
        self.update_image()

    def reset_blue(self):
        """Reset all Blue channel adjustment sliders to default values"""
        self.b_hsv_var.set(0)
        self.b_contrast_var.set(0)
        self.b_brightness_var.set(0)
        self.update_image()

    def reset_all(self):
        """Reset all sliders to default values"""
        # Reset global adjustments
        self.reset_global()

        # Reset channel adjustments
        self.reset_red()
        self.reset_green()
        self.reset_blue()

        # Reset alpha
        self.alpha_var.set(100)

        # Update the image
        self.update_image()

    def export_code(self):
        """
        Export the current image adjustments as Python code.
        This generates a Python script that can reproduce the current edits on any image.
        """
        if self.original_image is None:
            messagebox.showinfo("Info", "No image loaded")
            return

        # Ask for a file to save the Python code
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # Get current slider values
            hue = self.hue_var.get()
            saturation = self.saturation_var.get()
            value = self.value_var.get()
            contrast = self.contrast_var.get()
            brightness = self.brightness_var.get()
            alpha = self.alpha_var.get() / 100.0  # Convert to 0-1 range
            r_hsv = self.r_hsv_var.get()
            g_hsv = self.g_hsv_var.get()
            b_hsv = self.b_hsv_var.get()
            r_contrast = self.r_contrast_var.get()
            g_contrast = self.g_contrast_var.get()
            b_contrast = self.b_contrast_var.get()
            r_brightness = self.r_brightness_var.get()
            g_brightness = self.g_brightness_var.get()
            b_brightness = self.b_brightness_var.get()

            # Create the code string with proper variable values
            code = self._generate_python_script(
                hue, saturation, value, contrast, brightness, alpha,
                r_hsv, g_hsv, b_hsv, r_contrast, g_contrast, b_contrast,
                r_brightness, g_brightness, b_brightness
            )

            # Write the code to the file
            with open(file_path, 'w') as f:
                f.write(code)

            messagebox.showinfo("Success", f"Code exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Error exporting code: {str(e)}")

    def _generate_python_script(self, hue, saturation, value, contrast, brightness, alpha,
                               r_hsv, g_hsv, b_hsv, r_contrast, g_contrast, b_contrast,
                               r_brightness, g_brightness, b_brightness):
        """
        Generate Python code for applying the current image adjustments.

        Returns:
            str: Python code as a string
        """
        # Build the script manually to avoid f-string issues
        code = []
        code.append('"""')
        code.append('Image Adjustment Script')
        code.append('')
        code.append('This script was automatically generated by the Image Editor tool.')
        code.append('It applies the same adjustments that were made in the GUI to any image.')
        code.append('')
        code.append('Usage:')
        code.append('    python this_script.py input_image.jpg output_image.jpg')
        code.append('"""')
        code.append('')
        code.append('import cv2')
        code.append('import numpy as np')
        code.append('import sys')
        code.append('')
        code.append('def process_image(image_path, output_path=None):')
        code.append('    """')
        code.append('    Apply image adjustments to the input image and save the result.')
        code.append('    ')
        code.append('    Args:')
        code.append('        image_path (str): Path to the input image')
        code.append('        output_path (str, optional): Path to save the output image.')
        code.append('            If None, displays the image instead.')
        code.append('    """')
        code.append('    # Read the image')
        code.append('    original_image = cv2.imread(image_path)')
        code.append('    if original_image is None:')
        code.append('        print(f"Error: Could not read image {image_path}")')
        code.append('        return')
        code.append('    ')
        code.append('    # Convert BGR to RGB (OpenCV uses BGR by default)')
        code.append('    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)')
        code.append('    ')
        code.append('    # Get adjustment values')
        code.append(f'    hue = {hue}')
        code.append(f'    saturation = {saturation}')
        code.append(f'    value = {value}')
        code.append(f'    contrast = {contrast}')
        code.append(f'    brightness = {brightness}')
        code.append(f'    alpha = {alpha}')
        code.append(f'    r_hsv = {r_hsv}')
        code.append(f'    g_hsv = {g_hsv}')
        code.append(f'    b_hsv = {b_hsv}')
        code.append(f'    r_contrast = {r_contrast}')
        code.append(f'    g_contrast = {g_contrast}')
        code.append(f'    b_contrast = {b_contrast}')
        code.append(f'    r_brightness = {r_brightness}')
        code.append(f'    g_brightness = {g_brightness}')
        code.append(f'    b_brightness = {b_brightness}')
        code.append('    ')
        code.append('    # Check if per-channel processing is needed')
        code.append('    needs_channel_processing = (')
        code.append('        r_hsv != 0 or g_hsv != 0 or b_hsv != 0 or')
        code.append('        r_contrast != 0 or g_contrast != 0 or b_contrast != 0 or')
        code.append('        r_brightness != 0 or g_brightness != 0 or b_brightness != 0')
        code.append('    )')
        code.append('    ')
        code.append('    # Apply adjustments')
        code.append('    if not needs_channel_processing:')
        code.append('        # Convert to HSV')
        code.append('        hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV).astype(np.float32)')
        code.append('')
        code.append('        # Apply HSV adjustments')
        code.append('        hsv[:,:,0] = (hsv[:,:,0] + hue) % 180')
        code.append('        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + saturation/100), 0, 255)')
        code.append('        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + value/100), 0, 255)')
        code.append('')
        code.append('        # Convert back to RGB')
        code.append('        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)')
        code.append('        ')
        code.append('        # Apply global contrast and brightness')
        code.append('        modified = cv2.convertScaleAbs(modified, alpha=1 + contrast/100, beta=brightness)')
        code.append('    else:')
        code.append('        # Start with the original image')
        code.append('        result = original_image.copy()')
        code.append('        ')
        code.append('        # Process HSV rotation for each channel')
        code.append('        if r_hsv != 0 or g_hsv != 0 or b_hsv != 0:')
        code.append('            # Split the original image')
        code.append('            r, g, b = cv2.split(original_image)')
        code.append('            ')
        code.append('            # Create masks for isolating each channel')
        code.append('            r_mask = np.zeros_like(original_image)')
        code.append('            g_mask = np.zeros_like(original_image)')
        code.append('            b_mask = np.zeros_like(original_image)')
        code.append('            ')
        code.append('            # Copy each channel to its appropriate position')
        code.append('            r_mask[:,:,0] = r')
        code.append('            g_mask[:,:,1] = g')
        code.append('            b_mask[:,:,2] = b')
        code.append('            ')
        code.append('            # Process red channel if needed')
        code.append('            if r_hsv != 0:')
        code.append('                # Convert red mask to HSV and rotate hue')
        code.append('                r_hsv_img = cv2.cvtColor(r_mask, cv2.COLOR_RGB2HSV)')
        code.append('                r_hsv_img[:,:,0] = (r_hsv_img[:,:,0] + r_hsv) % 180')
        code.append('                ')
        code.append('                # Convert back to RGB')
        code.append('                r_rotated = cv2.cvtColor(r_hsv_img, cv2.COLOR_HSV2RGB)')
        code.append('                ')
        code.append('                # Add the rotated red channel to the result')
        code.append('                result = cv2.add(result - r_mask, r_rotated)')
        code.append('            ')
        code.append('            # Process green channel if needed')
        code.append('            if g_hsv != 0:')
        code.append('                # Convert green mask to HSV and rotate hue')
        code.append('                g_hsv_img = cv2.cvtColor(g_mask, cv2.COLOR_RGB2HSV)')
        code.append('                g_hsv_img[:,:,0] = (g_hsv_img[:,:,0] + g_hsv) % 180')
        code.append('                ')
        code.append('                # Convert back to RGB')
        code.append('                g_rotated = cv2.cvtColor(g_hsv_img, cv2.COLOR_HSV2RGB)')
        code.append('                ')
        code.append('                # Add the rotated green channel to the result')
        code.append('                result = cv2.add(result - g_mask, g_rotated)')
        code.append('            ')
        code.append('            # Process blue channel if needed')
        code.append('            if b_hsv != 0:')
        code.append('                # Convert blue mask to HSV and rotate hue')
        code.append('                b_hsv_img = cv2.cvtColor(b_mask, cv2.COLOR_RGB2HSV)')
        code.append('                b_hsv_img[:,:,0] = (b_hsv_img[:,:,0] + b_hsv) % 180')
        code.append('                ')
        code.append('                # Convert back to RGB')
        code.append('                b_rotated = cv2.cvtColor(b_hsv_img, cv2.COLOR_HSV2RGB)')
        code.append('                ')
        code.append('                # Add the rotated blue channel to the result')
        code.append('                result = cv2.add(result - b_mask, b_rotated)')
        code.append('        ')
        code.append('        # Apply per-channel contrast and brightness')
        code.append('        if (r_contrast != 0 or r_brightness != 0 or ')
        code.append('            g_contrast != 0 or g_brightness != 0 or ')
        code.append('            b_contrast != 0 or b_brightness != 0):')
        code.append('            ')
        code.append('            # Split the image into channels')
        code.append('            r, g, b = cv2.split(result)')
        code.append('            ')
        code.append('            # Apply contrast and brightness to each channel')
        code.append('            if r_contrast != 0 or r_brightness != 0:')
        code.append('                contrast_factor = 1 + r_contrast/100')
        code.append('                r = cv2.convertScaleAbs(r, alpha=contrast_factor, beta=r_brightness)')
        code.append('            ')
        code.append('            if g_contrast != 0 or g_brightness != 0:')
        code.append('                contrast_factor = 1 + g_contrast/100')
        code.append('                g = cv2.convertScaleAbs(g, alpha=contrast_factor, beta=g_brightness)')
        code.append('            ')
        code.append('            if b_contrast != 0 or b_brightness != 0:')
        code.append('                contrast_factor = 1 + b_contrast/100')
        code.append('                b = cv2.convertScaleAbs(b, alpha=contrast_factor, beta=b_brightness)')
        code.append('            ')
        code.append('            # Merge the channels back')
        code.append('            result = cv2.merge([r, g, b])')
        code.append('        ')
        code.append('        # Now apply global HSV processing')
        code.append('        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)')
        code.append('        ')
        code.append('        # Apply global HSV adjustments')
        code.append('        hsv[:,:,0] = (hsv[:,:,0] + hue) % 180')
        code.append('        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + saturation/100), 0, 255)')
        code.append('        hsv[:,:,2] = np.clip(hsv[:,:,2] * (1 + value/100), 0, 255)')
        code.append('        ')
        code.append('        # Convert back to RGB')
        code.append('        modified = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)')
        code.append('        ')
        code.append('        # Apply global contrast and brightness')
        code.append('        modified = cv2.convertScaleAbs(modified, alpha=1 + contrast/100, beta=brightness)')
        code.append('    ')
        code.append('    # Apply alpha')
        code.append('    if alpha < 1.0:')
        code.append('        modified = cv2.addWeighted(original_image, 1-alpha, modified, alpha, 0)')
        code.append('    ')
        code.append('    # Convert back to BGR for saving or display')
        code.append('    output_image = cv2.cvtColor(modified, cv2.COLOR_RGB2BGR)')
        code.append('    ')
        code.append('    # Save or display the result')
        code.append('    if output_path:')
        code.append('        cv2.imwrite(output_path, output_image)')
        code.append('        print(f"Processed image saved to {output_path}")')
        code.append('    else:')
        code.append('        cv2.imshow("Processed Image", output_image)')
        code.append('        cv2.waitKey(0)')
        code.append('        cv2.destroyAllWindows()')
        code.append('')
        code.append('if __name__ == "__main__":')
        code.append('    # Check command line arguments')
        code.append('    if len(sys.argv) < 2:')
        code.append('        print("Usage: python this_script.py input_image.jpg [output_image.jpg]")')
        code.append('        sys.exit(1)')
        code.append('    ')
        code.append('    input_path = sys.argv[1]')
        code.append('    output_path = sys.argv[2] if len(sys.argv) > 2 else None')
        code.append('    ')
        code.append('    process_image(input_path, output_path)')

        return '\n'.join(code)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python this_script.py input_image.jpg [output_image.jpg]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    process_image(input_path, output_path)
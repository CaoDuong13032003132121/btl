import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage, signal
import os
import colorsys

# ================= IMAGE PROCESSOR CLASS ==================
class ImageProcessor:
    @staticmethod
    def convert_color_space(img, color_space):
        try:
            if color_space == "RGB":
                return img
            elif color_space == "HSV":
                # Chuyển RGB sang HSV sử dụng colorsys
                if len(img.shape) == 3:
                    hsv_img = np.zeros_like(img, dtype=np.uint8)
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            r, g, b = img[i, j] / 255.0
                            h, s, v = colorsys.rgb_to_hsv(r, g, b)
                            hsv_img[i, j] = [int(h * 179), int(s * 255), int(v * 255)]
                    return hsv_img
                return img
            elif color_space == "CMYK":
                # Improved CMYK conversion
                rgb = img.astype(np.float32) / 255.0
                k = 1 - np.max(rgb, axis=2)
                c = (1 - rgb[:,:,0] - k) / (1 - k + 1e-7)
                m = (1 - rgb[:,:,1] - k) / (1 - k + 1e-7)
                y = (1 - rgb[:,:,2] - k) / (1 - k + 1e-7)
                cmyk = np.stack([c, m, y, k], axis=2)
                return (cmyk * 255).astype(np.uint8)
            elif color_space == "YCbCr":
                # Chuyển RGB sang YCbCr
                if len(img.shape) == 3:
                    img_float = img.astype(np.float32)
                    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
                    
                    y = 0.299 * r + 0.587 * g + 0.114 * b
                    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
                    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
                    
                    ycrcb = np.stack([y, cr, cb], axis=2)
                    return np.clip(ycrcb, 0, 255).astype(np.uint8)
                return img
            elif color_space == "GRAY":
                # Chuyển RGB sang Grayscale
                if len(img.shape) == 3:
                    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
                    return gray.astype(np.uint8)
                else:
                    return img
            else:
                return img
        except Exception as e:
            print("Color conversion error:", e)
            return img

    @staticmethod
    def adjust_image(img, brightness, contrast, gamma):
        # Brightness and contrast adjustment
        img_adj = img.astype(np.float32)
        img_adj = img_adj * contrast + brightness
        img_adj = np.clip(img_adj, 0, 255).astype(np.uint8)
        
        # Gamma correction
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
            if len(img_adj.shape) == 3:
                img_adj = table[img_adj]
            else:
                img_adj = table[img_adj]
        return img_adj

    @staticmethod
    def apply_filter(img, filter_type, kernel_size):
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        if filter_type == "Gaussian":
            # Gaussian blur using scipy
            if len(img.shape) == 3:
                result = np.zeros_like(img)
                for i in range(3):
                    result[:,:,i] = ndimage.gaussian_filter(img[:,:,i], sigma=k/3)
                return result
            else:
                return ndimage.gaussian_filter(img, sigma=k/3)
        elif filter_type == "Median":
            # Median filter using scipy
            if len(img.shape) == 3:
                result = np.zeros_like(img)
                for i in range(3):
                    result[:,:,i] = ndimage.median_filter(img[:,:,i], size=k)
                return result
            else:
                return ndimage.median_filter(img, size=k)
        elif filter_type == "Bilateral":
            # Bilateral filter approximation using Gaussian
            if len(img.shape) == 3:
                result = np.zeros_like(img)
                for i in range(3):
                    result[:,:,i] = ndimage.gaussian_filter(img[:,:,i], sigma=k/2)
                return result
            else:
                return ndimage.gaussian_filter(img, sigma=k/2)
        elif filter_type == "Laplacian":
            # Laplacian filter
            if len(img.shape) == 3:
                gray = ImageProcessor.convert_color_space(img, "GRAY")
            else:
                gray = img
            laplacian = ndimage.laplace(gray.astype(np.float32))
            return np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)
        elif filter_type == "Sobel":
            # Sobel filter
            if len(img.shape) == 3:
                gray = ImageProcessor.convert_color_space(img, "GRAY")
            else:
                gray = img
            sobelx = ndimage.sobel(gray, axis=1)
            sobely = ndimage.sobel(gray, axis=0)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def denoise_image(img):
        # Non-local means denoising approximation using median filter
        if len(img.shape) == 3:
            result = np.zeros_like(img)
            for i in range(3):
                result[:,:,i] = ndimage.median_filter(img[:,:,i], size=3)
            return result
        else:
            return ndimage.median_filter(img, size=3)

    @staticmethod
    def sharpen_image(img):
        """Làm nét ảnh - ĐÃ SỬA LỖI"""
        try:
            # Sử dụng PIL filter thay vì convolution để tránh lỗi
            if len(img.shape) == 3:
                # Chuyển numpy array sang PIL Image
                pil_img = Image.fromarray(img.astype(np.uint8))
                # Áp dụng bộ lọc làm nét
                sharpened = pil_img.filter(ImageFilter.SHARPEN)
                # Chuyển lại thành numpy array
                return np.array(sharpened)
            else:
                # Ảnh grayscale
                pil_img = Image.fromarray(img.astype(np.uint8))
                sharpened = pil_img.filter(ImageFilter.SHARPEN)
                return np.array(sharpened)
        except Exception as e:
            print(f"Lỗi làm nét ảnh: {e}")
            return img

    @staticmethod
    def edge_detection(img):
        # Canny edge detection approximation
        if len(img.shape) == 3:
            gray = ImageProcessor.convert_color_space(img, "GRAY")
        else:
            gray = img
        
        # Gaussian blur
        blurred = ndimage.gaussian_filter(gray.astype(np.float32), sigma=1.0)
        
        # Gradient calculation
        grad_x = ndimage.sobel(blurred, axis=1)
        grad_y = ndimage.sobel(blurred, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Simple thresholding
        edges = np.zeros_like(magnitude)
        edges[magnitude > 50] = 128
        edges[magnitude > 150] = 255
        
        # Convert to 3 channels if needed
        if len(img.shape) == 3:
            edges = np.stack([edges, edges, edges], axis=2)
        
        return edges.astype(np.uint8)

    @staticmethod
    def display_histogram(img, parent):
        for widget in parent.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        if len(img.shape) == 3:
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histr, _ = np.histogram(img[:,:,i], bins=256, range=[0, 256])
                ax.plot(histr, color=col)
        else:
            ax.hist(img.ravel(), 256, [0, 256])
        ax.set_title("Histogram Ảnh")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    @staticmethod
    def display_color_channels(img, parent_frame, color_space="RGB"):
        """Hiển thị các kênh màu riêng biệt - ĐÃ SỬA LỖI"""
        try:
            # Xóa nội dung cũ
            for widget in parent_frame.winfo_children():
                widget.destroy()

            # Tạo frame chính với scrollbar
            canvas = tk.Canvas(parent_frame, bg="#f5f5f5", highlightthickness=0)
            scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas, bg="#f5f5f5")
            
            scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Tiêu đề
            title_label = tk.Label(scroll_frame, 
                                 text=f"PHÂN TÍCH KÊNH MÀU - HỆ {color_space}", 
                                 font=("Segoe UI", 14, "bold"),
                                 fg="#2c3e50", bg="#f5f5f5")
            title_label.pack(pady=10)

            # Chuyển đổi ảnh sang hệ màu mong muốn
            converted_img = ImageProcessor.convert_color_space(img, color_space)

            # Xác định kênh màu dựa trên hệ màu
            if color_space == "RGB":
                if len(converted_img.shape) == 3:
                    channels = [converted_img[:,:,0], converted_img[:,:,1], converted_img[:,:,2]]  # R, G, B
                    channel_names = ['Kênh ĐỎ (Red)', 'Kênh XANH LÁ (Green)', 'Kênh XANH DƯƠNG (Blue)']
                    colors = ['red', 'green', 'blue']
                else:
                    channels = [converted_img]
                    channel_names = ['Ảnh Grayscale']
                    colors = ['gray']
                    
            elif color_space == "HSV":
                if len(converted_img.shape) == 3:
                    channels = [converted_img[:,:,0], converted_img[:,:,1], converted_img[:,:,2]]  # H, S, V
                    channel_names = ['Kênh Hue (Màu sắc)', 'Kênh Saturation (Độ bão hòa)', 'Kênh Value (Độ sáng)']
                    colors = ['hsv', 'hot', 'gray']
                else:
                    channels = [converted_img]
                    channel_names = ['Ảnh Grayscale']
                    colors = ['gray']
                
            elif color_space == "YCbCr":
                if len(converted_img.shape) == 3:
                    channels = [converted_img[:,:,0], converted_img[:,:,1], converted_img[:,:,2]]  # Y, Cr, Cb
                    channel_names = ['Kênh Y (Độ chói)', 'Kênh Cr (Red Difference)', 'Kênh Cb (Blue Difference)']
                    colors = ['gray', 'cool', 'spring']
                else:
                    channels = [converted_img]
                    channel_names = ['Ảnh Grayscale']
                    colors = ['gray']
                    
            elif color_space == "CMYK":
                if len(converted_img.shape) == 3:
                    channels = [converted_img[:,:,0], converted_img[:,:,1], converted_img[:,:,2], converted_img[:,:,3]]
                    channel_names = ['Kênh Cyan', 'Kênh Magenta', 'Kênh Yellow', 'Kênh Black']
                    colors = ['cyan', 'magenta', 'yellow', 'gray']
                else:
                    channels = [converted_img]
                    channel_names = ['Ảnh CMYK']
                    colors = ['gray']
                    
            elif color_space == "GRAY":
                channels = [converted_img]
                channel_names = ['Ảnh Grayscale']
                colors = ['gray']
            else:
                channels = [converted_img]
                channel_names = ['Ảnh Gốc']
                colors = ['gray']

            # Hiển thị từng kênh
            for i, (channel, name, color) in enumerate(zip(channels, channel_names, colors)):
                # Tạo frame cho mỗi kênh
                channel_container = tk.Frame(scroll_frame, bg="white", relief="raised", bd=1)
                channel_container.pack(fill="x", padx=10, pady=5)

                # Tên kênh
                name_frame = tk.Frame(channel_container, bg="#4a90e2")
                name_frame.pack(fill="x")
                
                name_label = tk.Label(name_frame, text=name, font=("Segoe UI", 11, "bold"),
                                    fg="white", bg="#4a90e2", pady=5)
                name_label.pack()

                # Nội dung kênh
                content_frame = tk.Frame(channel_container, bg="white")
                content_frame.pack(fill="x", padx=10, pady=10)

                # Chuẩn bị ảnh để hiển thị
                if len(channel.shape) == 2:  # Grayscale
                    display_img = channel
                else:  # Color channel
                    display_img = channel

                # Chuyển đổi sang PIL Image
                if len(display_img.shape) == 2:
                    pil_img = Image.fromarray(display_img.astype(np.uint8))
                else:
                    pil_img = Image.fromarray(display_img.astype(np.uint8))

                # Resize ảnh để hiển thị
                width, height = pil_img.size
                max_size = 300
                if width > max_size or height > max_size:
                    ratio = min(max_size/width, max_size/height)
                    new_size = (int(width * ratio), int(height * ratio))
                    pil_img = pil_img.resize(new_size, Image.LANCZOS)

                # Chuyển sang PhotoImage
                img_tk = ImageTk.PhotoImage(pil_img)

                # Hiển thị ảnh
                img_label = tk.Label(content_frame, image=img_tk, bg="white")
                img_label.image = img_tk  # Giữ reference
                img_label.pack(side="left", padx=10)

                # Thông tin thống kê
                stats_frame = tk.Frame(content_frame, bg="white")
                stats_frame.pack(side="right", fill="y", padx=10)

                stats_text = f"""THỐNG KÊ:
Min: {np.min(channel):.1f}
Max: {np.max(channel):.1f}
Mean: {np.mean(channel):.1f}
Std: {np.std(channel):.1f}"""

                stats_label = tk.Label(stats_frame, text=stats_text, 
                                     font=("Consolas", 9), justify="left",
                                     fg="#2c3e50", bg="white")
                stats_label.pack(anchor="w")

        except Exception as e:
            error_label = tk.Label(parent_frame, 
                                 text=f"Lỗi hiển thị kênh màu: {str(e)}", 
                                 fg="red", bg="#f5f5f5")
            error_label.pack(pady=20)
            print(f"Error in display_color_channels: {e}")

    @staticmethod
    def show_channel_analysis(img, parent_frame):
        """Phân tích nâng cao các kênh màu"""
        ImageProcessor.display_color_channels(img, parent_frame, "RGB")


# ================= MAIN APP ==================
class ModernImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 Ứng dụng Xử lý Ảnh - Chuyển đổi Hệ màu & Nâng cao Chất lượng")
        self.root.geometry("1400x800")
        self.root.configure(bg="#f5f5f5")

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.current_color_space = "RGB"

        self.setup_ui()

    def setup_ui(self):
        header = tk.Label(self.root, text="🎨 ỨNG DỤNG XỬ LÝ ẢNH NÂNG CAO",
                          font=("Segoe UI", 20, "bold"),
                          bg="#4a90e2", fg="white", pady=15)
        header.pack(fill="x")

        container = tk.Frame(self.root, bg="#f5f5f5")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Control Panel
        control_frame = tk.Frame(container, bg="white", width=340, relief="ridge", bd=1)
        control_frame.pack(side="left", fill="y", padx=(0, 10))
        control_frame.pack_propagate(False)

        self.create_controls(control_frame)

        # Right display area
        self.display_frame = tk.Frame(container, bg="#e9eef5", relief="ridge", bd=1)
        self.display_frame.pack(side="right", fill="both", expand=True)
        self.create_display_tabs()

    def create_controls(self, parent):
        ttk.Button(parent, text="📂 Chọn Ảnh", command=self.choose_image).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="💾 Lưu Ảnh", command=self.save_image).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="🔄 Khôi phục Ảnh Gốc", command=self.reset_image).pack(fill="x", pady=8, padx=10)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(parent, text="Chuyển hệ màu:", bg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10)
        self.color_var = tk.StringVar(value="RGB")
        for mode in ["RGB", "HSV", "CMYK", "YCbCr", "GRAY"]:
            ttk.Radiobutton(parent, text=mode, variable=self.color_var, value=mode,
                            command=self.on_color_space_change).pack(anchor="w", padx=20)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(parent, text="Độ sáng:", bg="white").pack(anchor="w", padx=10)
        self.brightness_var = tk.DoubleVar(value=0)
        ttk.Scale(parent, from_=-100, to=100, variable=self.brightness_var, command=self.on_adjustment_change).pack(fill="x", padx=10)

        tk.Label(parent, text="Độ tương phản:", bg="white").pack(anchor="w", padx=10)
        self.contrast_var = tk.DoubleVar(value=1)
        ttk.Scale(parent, from_=0.1, to=3, variable=self.contrast_var, command=self.on_adjustment_change).pack(fill="x", padx=10)

        tk.Label(parent, text="Gamma:", bg="white").pack(anchor="w", padx=10)
        self.gamma_var = tk.DoubleVar(value=1)
        ttk.Scale(parent, from_=0.1, to=3, variable=self.gamma_var, command=self.on_adjustment_change).pack(fill="x", padx=10)

        ttk.Button(parent, text="🧹 Khử Nhiễu", command=self.apply_denoising).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="✨ Làm Nét", command=self.sharpen_image).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="📊 Histogram", command=self.show_histogram).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="🎨 Kênh Màu", command=self.show_color_channels).pack(fill="x", pady=8, padx=10)
        ttk.Button(parent, text="🔍 Phát hiện Biên", command=self.edge_detection).pack(fill="x", pady=8, padx=10)

    def create_display_tabs(self):
        notebook = ttk.Notebook(self.display_frame)
        notebook.pack(fill="both", expand=True)

        self.tab_compare = tk.Frame(notebook, bg="#f5f5f5")
        self.tab_hist = tk.Frame(notebook, bg="#f5f5f5")
        self.tab_channels = tk.Frame(notebook, bg="#f5f5f5")

        notebook.add(self.tab_compare, text="🔄 So sánh Ảnh")
        notebook.add(self.tab_hist, text="📊 Histogram")
        notebook.add(self.tab_channels, text="🎨 Kênh Màu")

        # Frames hiển thị ảnh
        self.orig_label = tk.Label(self.tab_compare, bg="#ddd", text="Ảnh gốc sẽ hiển thị ở đây")
        self.proc_label = tk.Label(self.tab_compare, bg="#ddd", text="Ảnh đã xử lý sẽ hiển thị ở đây")
        self.orig_label.pack(side="left", expand=True, fill="both", padx=10, pady=10)
        self.proc_label.pack(side="right", expand=True, fill="both", padx=10, pady=10)

    # ========== CORE FUNCTIONALITY ==========
    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if not path:
            return
        self.image_path = path
        # Đọc ảnh bằng PIL và chuyển sang numpy array
        pil_image = Image.open(path)
        self.original_image = np.array(pil_image)
        self.processed_image = self.original_image.copy()
        self.update_display()

    def show_image(self, img, label):
        try:
            # Chuyển numpy array sang PIL Image để hiển thị
            if len(img.shape) == 2:  # Grayscale
                pil_img = Image.fromarray(img.astype(np.uint8))
            else:  # Color
                pil_img = Image.fromarray(img.astype(np.uint8))
            
            # KHÔNG resize, giữ nguyên kích thước ảnh thật
            img_tk = ImageTk.PhotoImage(pil_img)

            # Cập nhật hiển thị trên Label
            label.configure(image=img_tk, text="")
            label.image = img_tk  # Giữ reference để ảnh không bị mất

        except Exception as e:
            label.configure(text=f"Lỗi hiển thị: {e}", fg="red")

    def update_display(self):
        if self.original_image is not None:
            self.show_image(self.original_image, self.orig_label)
        if self.processed_image is not None:
            self.show_image(self.processed_image, self.proc_label)

    def on_color_space_change(self):
        if self.processed_image is not None:
            self.current_color_space = self.color_var.get()
            self.processed_image = ImageProcessor.convert_color_space(self.original_image.copy(), self.color_var.get())
            self.update_display()

    def on_adjustment_change(self, event=None):
        if self.original_image is not None:
            self.processed_image = ImageProcessor.adjust_image(
                self.original_image.copy(),
                self.brightness_var.get(),
                self.contrast_var.get(),
                self.gamma_var.get()
            )
            self.update_display()

    def apply_denoising(self):
        if self.processed_image is not None:
            self.processed_image = ImageProcessor.denoise_image(self.processed_image)
            self.update_display()

    def sharpen_image(self):
        if self.processed_image is not None:
            self.processed_image = ImageProcessor.sharpen_image(self.processed_image)
            self.update_display()

    def edge_detection(self):
        if self.processed_image is not None:
            self.processed_image = ImageProcessor.edge_detection(self.processed_image)
            self.update_display()

    def show_histogram(self):
        if self.processed_image is not None:
            ImageProcessor.display_histogram(self.processed_image, self.tab_hist)

    def show_color_channels(self):
        """Hiển thị phân tích kênh màu - ĐÃ SỬA LỖI"""
        if self.processed_image is not None:
            ImageProcessor.display_color_channels(self.processed_image, self.tab_channels, self.current_color_space)
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.color_var.set("RGB")
            self.current_color_space = "RGB"
            self.brightness_var.set(0)
            self.contrast_var.set(1)
            self.gamma_var.set(1)
            self.update_display()

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu!")
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            # Lưu ảnh bằng PIL
            if len(self.processed_image.shape) == 2:
                pil_img = Image.fromarray(self.processed_image.astype(np.uint8))
            else:
                pil_img = Image.fromarray(self.processed_image.astype(np.uint8))
            pil_img.save(path)
            messagebox.showinfo("✅ Thành công", f"Ảnh đã được lưu tại:\n{path}")

    
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = ModernImageApp(root)
    root.mainloop()
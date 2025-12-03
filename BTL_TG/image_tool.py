import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance
import tkinter as tk

class ImageProcessor:
    
    @staticmethod
    def convert_color_space(img, target_space):
        """Chuyển đổi giữa các hệ màu: RGB, HSV, CMYK, YCbCr"""
        try:
            if target_space == "RGB":
                return img.convert('RGB') if img.mode != 'RGB' else img
            elif target_space == "HSV":
                return img.convert('HSV')
            elif target_space == "YCbCr":
                return img.convert('YCbCr')
            elif target_space == "GRAY":
                return img.convert('L')
            elif target_space == "CMYK":
                return img.convert('CMYK')
            return img
        except Exception as e:
            print(f"Lỗi chuyển đổi hệ màu: {e}")
            return img
    
    @staticmethod
    def bgr_to_cmyk(img):
        """Chuyển đổi RGB sang CMYK (PIL đã có sẵn)"""
        try:
            return img.convert('CMYK')
        except Exception as e:
            print(f"Lỗi chuyển đổi CMYK: {e}")
            return img
    
    @staticmethod
    def adjust_image(img, brightness=0, contrast=1.0, gamma=1.0):
        """Điều chỉnh độ sáng, tương phản và gamma"""
        try:
            # Điều chỉnh độ sáng và tương phản
            if brightness != 0:
                enhancer = ImageEnhance.Brightness(img)
                img_adj = enhancer.enhance(1.0 + brightness/100.0)
            else:
                img_adj = img
                
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img_adj)
                img_adj = enhancer.enhance(contrast)
            
            # Gamma correction
            if gamma != 1.0:
                # Áp dụng gamma correction thủ công
                img_array = np.array(img_adj, dtype=np.float32) / 255.0
                img_array = np.power(img_array, 1.0/gamma)
                img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
                img_adj = Image.fromarray(img_array)
                
            return img_adj
        except Exception as e:
            print(f"Lỗi điều chỉnh ảnh: {e}")
            return img
    
    @staticmethod
    def apply_filter(img, filter_type, kernel_size=3):
        """Áp dụng các bộ lọc khác nhau"""
        try:
            if filter_type == "Gaussian":
                return img.filter(ImageFilter.GaussianBlur(radius=kernel_size/2))
            elif filter_type == "Median":
                return img.filter(ImageFilter.MedianFilter(size=kernel_size))
            elif filter_type == "Bilateral":
                # PIL không có bilateral filter, dùng Gaussian thay thế
                return img.filter(ImageFilter.GaussianBlur(radius=kernel_size/3))
            elif filter_type == "Laplacian":
                # Edge enhancement thay cho Laplacian
                return img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif filter_type == "Sobel":
                # Tạo kernel Sobel thủ công
                return ImageProcessor._sobel_filter(img)
            return img
        except Exception as e:
            print(f"Lỗi áp dụng bộ lọc: {e}")
            return img
    
    @staticmethod
    def _sobel_filter(img):
        """Triển khai Sobel filter thủ công"""
        try:
            # Chuyển sang grayscale nếu cần
            if img.mode != 'L':
                gray_img = img.convert('L')
            else:
                gray_img = img
            
            img_array = np.array(gray_img, dtype=np.float32)
            
            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            # Padding ảnh
            padded = np.pad(img_array, 1, mode='constant')
            
            # Áp dụng convolution
            grad_x = np.zeros_like(img_array)
            grad_y = np.zeros_like(img_array)
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    grad_x[i, j] = np.sum(region * sobel_x)
                    grad_y[i, j] = np.sum(region * sobel_y)
            
            # Tính magnitude
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
            
            return Image.fromarray(magnitude).convert('RGB')
        except Exception as e:
            print(f"Lỗi Sobel filter: {e}")
            return img

    @staticmethod
    def denoise_image(img, method='auto', strength='medium'):
        """
        Khử nhiễu ảnh nâng cao với nhiều phương pháp
        """
        try:
            if img is None:
                return img
                
            # Tham số cho các mức độ
            strength_params = {
                'light': {
                    'median_size': 3,
                    'gaussian_radius': 0.8,
                    'sharpen_factor': 0.5
                },
                'medium': {
                    'median_size': 5,
                    'gaussian_radius': 1.2,
                    'sharpen_factor': 0.3
                },
                'strong': {
                    'median_size': 7,
                    'gaussian_radius': 1.6,
                    'sharpen_factor': 0.1
                }
            }
            
            params = strength_params.get(strength, strength_params['medium'])
            
            # Tự động chọn phương pháp
            if method == 'auto':
                method = ImageProcessor._analyze_noise_type(img)
                print(f"🔍 Tự động chọn phương pháp: {method}")
            
            print(f"🛠 Khử nhiễu: {method} | Độ mạnh: {strength}")
            
            if method == 'median':
                return ImageProcessor._median_denoise(img, params)
            elif method == 'gaussian':
                return ImageProcessor._gaussian_denoise(img, params)
            elif method == 'hybrid':
                return ImageProcessor._hybrid_denoise(img, params)
            else:
                # Mặc định dùng median
                return ImageProcessor._median_denoise(img, params)
                
        except Exception as e:
            print(f"Lỗi khử nhiễu: {e}")
            return img

    @staticmethod
    def _analyze_noise_type(img):
        """Phân tích loại nhiễu để chọn phương pháp phù hợp"""
        try:
            gray_img = img.convert('L')
            img_array = np.array(gray_img, dtype=np.float32)
            
            # Tính toán mức độ nhiễu (phương sai của gradient)
            grad_x = np.diff(img_array, axis=1)[:, :-1]
            grad_y = np.diff(img_array, axis=0)[:-1, :]
            noise_level = np.var(grad_x) + np.var(grad_y)
            
            print(f"📊 Mức nhiễu: {noise_level:.1f}")
            
            # Quyết định phương pháp dựa trên phân tích
            if noise_level > 1000:
                return 'hybrid'
            elif noise_level > 500:
                return 'median'
            elif noise_level > 200:
                return 'median'
            else:
                return 'gaussian'
                
        except Exception as e:
            print(f"⚠ Lỗi phân tích nhiễu: {e}")
            return 'median'

    @staticmethod
    def _median_denoise(img, params):
        """Median Filter - Hiệu quả với noise salt & pepper"""
        try:
            return img.filter(ImageFilter.MedianFilter(size=params['median_size']))
        except Exception as e:
            print(f"⚠ Lỗi median: {e}")
            return img

    @staticmethod
    def _gaussian_denoise(img, params):
        """Gaussian Filter - Mịn ảnh, giảm nhiễu Gaussian"""
        try:
            return img.filter(ImageFilter.GaussianBlur(radius=params['gaussian_radius']))
        except Exception as e:
            print(f"⚠ Lỗi gaussian: {e}")
            return img

    @staticmethod
    def _hybrid_denoise(img, params):
        """Kết hợp nhiều phương pháp cho nhiễu nặng"""
        try:
            # Bước 1: Median filter
            step1 = img.filter(ImageFilter.MedianFilter(size=3))
            
            # Bước 2: Gaussian filter
            step2 = step1.filter(ImageFilter.GaussianBlur(radius=1.0))
            
            return step2
        except Exception as e:
            print(f"⚠ Lỗi hybrid: {e}")
            return img

    @staticmethod
    def get_denoise_methods():
        """Trả về danh sách các phương pháp khử nhiễu"""
        return ['auto', 'median', 'gaussian', 'hybrid']

    @staticmethod
    def get_denoise_strengths():
        """Trả về danh sách các mức độ khử nhiễu"""
        return ['light', 'medium', 'strong']

    @staticmethod
    def sharpen_image(img):
        """Làm nét ảnh"""
        try:
            return img.filter(ImageFilter.SHARPEN)
        except Exception as e:
            print(f"Lỗi làm nét ảnh: {e}")
            return img
    
    @staticmethod
    def edge_detection(img):
        """Phát hiện biên cạnh"""
        try:
            gray_img = img.convert('L')
            # Sử dụng FIND_EDGES filter
            edges = gray_img.filter(ImageFilter.FIND_EDGES)
            return edges.convert('RGB')
        except Exception as e:
            print(f"Lỗi phát hiện biên: {e}")
            return img
    
    @staticmethod
    def display_histogram(img, parent_frame):
        """Hiển thị histogram của ảnh"""
        try:
            # Clear previous content
            for widget in parent_frame.winfo_children():
                widget.destroy()
                
            fig, axes = plt.subplots(1, 1, figsize=(12, 4))
            fig.patch.set_facecolor('#f5f5f5')
            
            # Chuyển ảnh sang numpy array
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:  # Color image
                colors = ['b', 'g', 'r']
                channels = ['Blue', 'Green', 'Red']
                for i, color in enumerate(colors):
                    axes.hist(img_array[:,:,i].ravel(), 256, [0,256], color=color, alpha=0.7, label=channels[i])
                axes.legend()
            else:  # Grayscale
                axes.hist(img_array.ravel(), 256, [0,256], color='gray', alpha=0.7)
            
            axes.set_title('Histogram', color='black')
            axes.set_facecolor('#ffffff')
            axes.tick_params(colors='black')
            
            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            print(f"Lỗi hiển thị histogram: {e}")
    
    @staticmethod
    def display_color_channels(img, parent_frame, color_space="RGB"):
        """Hiển thị các kênh màu riêng biệt"""
        try:
            # Clear previous content
            for widget in parent_frame.winfo_children():
                widget.destroy()
            
            # Tạo frame chính với scrollbar
            main_frame = tk.Frame(parent_frame, bg="#f5f5f5")
            main_frame.pack(fill="both", expand=True)
            
            # Tiêu đề
            title_label = tk.Label(main_frame, 
                                 text=f"PHÂN TÍCH KÊNH MÀU - HỆ {color_space}", 
                                 font=("Arial", 14, "bold"),
                                 fg="#2c3e50", bg="#f5f5f5")
            title_label.pack(pady=10)
            
            # Canvas với scrollbar
            canvas = tk.Canvas(main_frame, bg="#f5f5f5", highlightthickness=0)
            scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas, bg="#f5f5f5")
            
            scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True, padx=10)
            scrollbar.pack(side="right", fill="y")
            
            # Xác định kênh màu dựa trên hệ màu
            channels = []
            channel_names = []
            
            if color_space == "RGB":
                if img.mode == 'RGB':
                    r, g, b = img.split()
                    channels = [r, g, b]
                    channel_names = ['Kênh ĐỎ (Red)', 'Kênh XANH LÁ (Green)', 'Kênh XANH DƯƠNG (Blue)']
                else:
                    channels = [img]
                    channel_names = ['Ảnh Grayscale']
                    
            elif color_space == "HSV":
                hsv_img = img.convert('HSV')
                h, s, v = hsv_img.split()
                channels = [h, s, v]
                channel_names = ['Kênh Hue (Màu sắc)', 'Kênh Saturation (Độ bão hòa)', 'Kênh Value (Độ sáng)']
                
            elif color_space == "YCbCr":
                ycbcr_img = img.convert('YCbCr')
                y, cb, cr = ycbcr_img.split()
                channels = [y, cb, cr]
                channel_names = ['Kênh Y (Độ chói)', 'Kênh Cb (Blue Difference)', 'Kênh Cr (Red Difference)']
                
            elif color_space == "CMYK":
                cmyk_img = img.convert('CMYK')
                c, m, y, k = cmyk_img.split()
                channels = [c, m, y, k]
                channel_names = ['Kênh Cyan', 'Kênh Magenta', 'Kênh Yellow', 'Kênh Black']
                
            elif color_space == "GRAY":
                gray_img = img.convert('L')
                channels = [gray_img]
                channel_names = ['Ảnh Grayscale']
            
            # Hiển thị từng kênh
            for i, (channel, name) in enumerate(zip(channels, channel_names)):
                # Tạo frame cho mỗi kênh
                channel_frame = tk.Frame(scroll_frame, bg="white", relief="raised", bd=2)
                channel_frame.pack(fill="x", padx=10, pady=8, ipady=5)
                
                # Tên kênh
                name_label = tk.Label(channel_frame, text=name, font=("Arial", 12, "bold"),
                                    fg="white", bg="#4a90e2", pady=5)
                name_label.pack(fill="x", padx=5, pady=5)
                
                # Nội dung kênh
                content_frame = tk.Frame(channel_frame, bg="white")
                content_frame.pack(fill="x", padx=10, pady=10)
                
                # Chuẩn bị ảnh để hiển thị
                if channel.mode == 'L':  # Grayscale
                    # Tạo ảnh RGB từ grayscale để hiển thị
                    display_img = channel.convert('RGB')
                else:
                    display_img = channel
                
                # Resize ảnh để hiển thị
                max_size = 300
                width, height = display_img.size
                if width > max_size or height > max_size:
                    ratio = min(max_size/width, max_size/height)
                    new_size = (int(width * ratio), int(height * ratio))
                    display_img = display_img.resize(new_size, Image.LANCZOS)
                
                # Chuyển sang PhotoImage
                img_tk = ImageTk.PhotoImage(display_img)
                
                # Hiển thị ảnh
                img_label = tk.Label(content_frame, image=img_tk, bg="white")
                img_label.image = img_tk  # Giữ reference
                img_label.pack(side="left", padx=10)
                
                # Thông tin thống kê
                stats_frame = tk.Frame(content_frame, bg="white")
                stats_frame.pack(side="right", fill="y", padx=10)
                
                channel_array = np.array(channel)
                stats_text = f"""THỐNG KÊ KÊNH:
• Min: {np.min(channel_array):.1f}
• Max: {np.max(channel_array):.1f}
• Mean: {np.mean(channel_array):.1f}
• Std: {np.std(channel_array):.1f}
• Kích thước: {channel_array.shape}"""
                
                stats_label = tk.Label(stats_frame, text=stats_text, 
                                     font=("Consolas", 9), justify="left",
                                     fg="#2c3e50", bg="white")
                stats_label.pack(anchor="w")
                
        except Exception as e:
            error_label = tk.Label(parent_frame, 
                                 text=f"Lỗi hiển thị kênh màu: {str(e)}", 
                                 fg="red", bg="#f5f5f5", font=("Arial", 10))
            error_label.pack(pady=20)
            print(f"Lỗi trong display_color_channels: {e}")
    
    @staticmethod
    def view_color_channels_matplotlib(img, color_space="RGB"):
        """Hiển thị ảnh gốc và các kênh màu bằng matplotlib"""
        try:
            if img is None:
                print("❌ Ảnh không tồn tại hoặc chưa được nạp.")
                return
            
            if img.mode != 'RGB':
                print("⚠ Ảnh không có 3 kênh màu để hiển thị.")
                return
            
            if color_space == "RGB":
                img_array = np.array(img)
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 2, 1)
                plt.imshow(img_array)
                plt.title("Ảnh gốc (RGB)")
                plt.axis('off')

                plt.subplot(2, 2, 2)
                plt.imshow(r, cmap='Reds')
                plt.title("Kênh màu Đỏ (Red)")
                plt.axis('off')

                plt.subplot(2, 2, 3)
                plt.imshow(g, cmap='Greens')
                plt.title("Kênh màu Xanh lá (Green)")
                plt.axis('off')

                plt.subplot(2, 2, 4)
                plt.imshow(b, cmap='Blues')
                plt.title("Kênh màu Xanh dương (Blue)")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Lỗi hiển thị matplotlib: {e}")
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys, image_process

class MainWindow(QMainWindow, image_process.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('image_process_hw')
        self.setupUi(self)    
        
        self.image = None    
        
        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.smooth_filter)
        self.pushButton_3.clicked.connect(self.sharp)
        self.pushButton_4.clicked.connect(self.gaussian)
        self.pushButton_5.clicked.connect(self.lower_pass)
        
        
    #------------------------------------------------------#
    #   Graphic View       
    #------------------------------------------------------#   
    def convert_cvimage_to_pixmap(self, image):
        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            image_bytes = image.tobytes()
            q_image = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            image_bytes = image.tobytes()
            q_image = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        return pixmap  
    
    #------------------------------------------------------#
    #   Clear View       
    #------------------------------------------------------#
    def clear_view(self):
        labels = [self.label_2, self.label_3, self.label_4]
        graphics_views = [self.graphicsView_2, self.graphicsView_3, self.graphicsView_4]
        
        for label in labels:
            label.clear()
        
        for graphics_view in graphics_views:
            if graphics_view.scene() is not None and graphics_view.scene().items():
                graphics_view.scene().clear()
        
    #------------------------------------------------------#
    #   Load Image       
    #------------------------------------------------------#    
    def load_image(self):  
        self.clear_view()
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        
        if file_name:
            print("File name: ", file_name)
            self.image = cv2.imread(file_name)
            pixmap = self.convert_cvimage_to_pixmap(self.image)
            
            scene = QGraphicsScene(self)
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)

            self.graphicsView.setScene(scene)
            self.graphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            
    #------------------------------------------------------#
    #   Display Image       
    #------------------------------------------------------#
    def display_image(self, processed_image, label_text, graphics_view, label):
        pixmap = self.convert_cvimage_to_pixmap(processed_image)

        scene = QGraphicsScene(self)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)

        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

        label.setText(label_text)
            
    #------------------------------------------------------#
    #   Smooth Filter       
    #------------------------------------------------------#
    def gaussian_filter(self, size, sigma=1):
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)
     
    def smooth_filter(self):
        self.clear_view()
        average_filtered = cv2.blur(self.image, (3, 3))
        median_filtered = cv2.medianBlur(self.image, 3)

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        f_transform = np.fft.fft2(gray_img)
        f_transform_shifted = np.fft.fftshift(f_transform)
        
        cutoff_frequency = 30
        f_transform_shifted_filtered = f_transform_shifted.copy()
        mask = np.zeros((self.image.shape[:2]))
        rows, cols = self.image.shape[:2]
        crow, ccol = rows // 2, cols // 2
        mask[crow - cutoff_frequency:crow + cutoff_frequency, ccol - cutoff_frequency:ccol + cutoff_frequency] = self.gaussian_filter(60, sigma=20)
        f_transform_shifted_filtered = f_transform_shifted * mask

        image_filtered = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted_filtered))
        image_filtered = np.abs(image_filtered)
        image_filtered /= np.max(image_filtered)
        image_filtered *= 255

        self.display_image(average_filtered, '1(a) Average Filter', self.graphicsView_2, self.label_2)
        self.display_image(median_filtered, '1(b) Median Filter', self.graphicsView_3, self.label_3)
        self.display_image(image_filtered.astype(np.uint8), '1(b) Fourier Transform', self.graphicsView_4, self.label_4)

    #------------------------------------------------------#
    #   Sharp
    #------------------------------------------------------#        
    def sharp(self):
        self.clear_view()
        if self.image is None:
            print("Image not loaded.")
            return
        
        if len(self.image.shape) > 2:
            grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale_image = self.image

        sobel_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        gradient_magnitude /= np.sqrt(255 ** 2 * 2)
        gradient_magnitude = gradient_magnitude * 255
        
        f_transform = np.fft.fft2(grayscale_image)
        f_transform_shifted = np.fft.fftshift(f_transform)

        rows, cols = grayscale_image.shape
        crow, ccol = rows // 2, cols // 2

        filter_size = 30
        f_transform_shifted[crow - filter_size:crow + filter_size, ccol - filter_size:ccol + filter_size] = 0

        image_fourier_sharp = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted)).real
        image_fourier_sharp = np.abs(image_fourier_sharp)
        image_fourier_sharp /= np.max(image_fourier_sharp)
        image_fourier_sharp *= 255
        
        self.display_image(gradient_magnitude.astype(np.uint8), '2(a) Sobel mask', self.graphicsView_3, self.label_3)
        self.display_image(image_fourier_sharp.astype(np.uint8), '2(b) Fourier Transform', self.graphicsView_4, self.label_4)
        
    #------------------------------------------------------#
    #   Gaussian
    #------------------------------------------------------#
    def apply_filter(self, image, filter_mask):
        return cv2.filter2D(image, -1, filter_mask) 
    
    def gaussian(self):
        self.clear_view()
        filter_size = 5
        sigma = 1.0
        gaussian_mask = self.gaussian_filter(filter_size, sigma)

        low_pass_filtered_image = self.apply_filter(self.image, gaussian_mask)
        
        self.display_image(low_pass_filtered_image, 'Result', self.graphicsView_2, self.label_2)
        
    #------------------------------------------------------#
    #   Lower Pass
    #------------------------------------------------------#
    def lower_pass(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        filter_size = 5
        sigma = 1.0
        gaussian_mask = self.gaussian_filter(filter_size, sigma)

        gaussian_fft = np.fft.fft2(gaussian_mask, s=image_gray.shape)
        image_fft = np.fft.fft2(image_gray)
        smoothed_image_fft = image_fft * gaussian_fft

        smoothed_image = np.fft.ifft2(smoothed_image_fft).real
        smoothed_image = np.abs(smoothed_image)
        smoothed_image /= np.max(smoothed_image)
        smoothed_image *= 255
        
        self.display_image(smoothed_image.astype(np.uint8), 'Result', self.graphicsView_2, self.label_2)
         
               
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
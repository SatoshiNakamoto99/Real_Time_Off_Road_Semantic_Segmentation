
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QComboBox, QFileDialog, QSpinBox, QFrame)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox 
from model import TwinLite as net
from inference_utils import Run_rgb, Run_depth, Run_datafusion

class VideoProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('Video Processing GUI')
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 5px 10px;
                font-size: 14px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QComboBox, QSpinBox {
                background-color: #34495E;
                color: #ECF0F1;
                border: 1px solid #7F8C8D;
                padding: 5px;
                font-size: 14px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(20)

        # Modality selection every time the user selects a modality, 
        # the clear method is called to clear the frame labels
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(['RGB only', 'DEPTH only', 'Data Fusion'])
        self.modality_combo.currentIndexChanged.connect(self.modality_change)
        controls_layout.addWidget(QLabel('Modality:'))
        controls_layout.addWidget(self.modality_combo)


        # RGB/Depth selection buttons
        self.rgb_video_button = QPushButton('Select RGB Video')
        self.rgb_video_button.clicked.connect(self.select_rgb_video)
        controls_layout.addWidget(self.rgb_video_button)

        self.depth_video_button = QPushButton('Select Depth Video')
        self.depth_video_button.clicked.connect(self.select_depth_video)
        controls_layout.addWidget(self.depth_video_button)

        # Download button
        self.download_button = QPushButton('Download Processed Video')
        self.download_button.clicked.connect(self.download_video)
        controls_layout.addWidget(self.download_button)
        # FPS selection
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)
        controls_layout.addWidget(QLabel('FPS:'))
        controls_layout.addWidget(self.fps_spinbox)
        # Model selection
        self.model_button = QPushButton('Select Model')
        self.model_button.clicked.connect(self.select_model)
        controls_layout.addWidget(self.model_button)

        main_layout.addLayout(controls_layout)
        # Frame display
        display_layout = QHBoxLayout()
        text_title =["RGB Frame", "Depth Frame", "Processed Frame"]
        self.rgb_frame_label = QLabel()
        self.depth_frame_label = QLabel()
        self.processed_frame_label = QLabel()

        for label,title in zip([self.rgb_frame_label, self.depth_frame_label, self.processed_frame_label]
                                ,text_title):
            
            #label.setText(title) positioning in the top center
            
            label.setText(title)
            label.setStyleSheet("border: 2px solid #7F8C8D; background-color: #34495E;")
            label.setAlignment(Qt.AlignCenter)
            
            display_layout.addWidget(label)

        main_layout.addLayout(display_layout)
        self.depth_frame_label.hide()
        self.depth_video_button.hide()

        # Start button
        self.start_button = QPushButton('Start Processing')
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
        """)
        main_layout.addWidget(self.start_button)
        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
        """)
        main_layout.addWidget(self.clear_button)
        self.rgb_video_path = ''
        self.depth_video_path = ''
        self.model_path = ''
        self.cap_rgb = None
        self.cap_depth = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def select_rgb_video(self):
        self.clear()
        self.rgb_video_path, _ = QFileDialog.getOpenFileName(self, "Select RGB Video", "", "Video Files (*.mp4 *.mkv)")
        if self.rgb_video_path:
            self.rgb_video_button.setText(f"RGB Video: {self.rgb_video_path.split('/')[-1]}")
    def select_depth_video(self):
        self.clear()
        self.depth_video_path, _ = QFileDialog.getOpenFileName(self, "Select Depth Video", "", "Video Files (*.mp4 *.mkv)")
        if self.depth_video_path:
            self.depth_video_button.setText(f"Depth Video: {self.depth_video_path.split('/')[-1]}")
    def select_model(self):
        self.clear()
        self.model_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pth)")
        if self.model_path:
            self.model_button.setText(f"Model: {self.model_path.split('/')[-1]}")
            # Load the appropriate model here
            if self.modality_combo.currentText() == 'RGB only' or self.modality_combo.currentText() == 'DEPTH only':
                self.model = net.TwinLiteNet().to(self.device)
            else:
                self.model = net.TwinLiteNet_RGBD_Adaptive().to(self.device)
                #self.model = net.TwinLiteNet().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
    def check_modality_for_processing(self):
        modality = self.modality_combo.currentText()
        if modality == 'RGB only':
            if not self.rgb_video_path or not self.model_path:
                self.show_alert("Error", "Please select an RGB video and a model before starting processing.")
                return False, "RGB"
            return True, "RGB"
        
        elif modality == 'DEPTH only':
            if not self.depth_video_path or not self.model_path:
                self.show_alert("Error", "Please select a Depth video and a model before starting processing.")
                return False, "DEPTH"
            return True, "DEPTH"
        
        elif modality == 'Data Fusion':
            if not self.rgb_video_path or not self.depth_video_path or not self.model_path:
                self.show_alert("Error", "Please select RGB, Depth videos and a model before starting processing.")
                return False, "RGBD"
            return True, "RGBD"
    def start_processing(self):
        check, mod = self.check_modality_for_processing()
        if not check:
            return
        
        if mod == 'RGB':
            self.cap_rgb = cv2.VideoCapture(self.rgb_video_path)
            self.cap_depth = None
        elif mod == 'DEPTH':
            self.cap_rgb = None
            self.cap_depth = cv2.VideoCapture(self.depth_video_path)
        else:
            self.cap_rgb = cv2.VideoCapture(self.rgb_video_path)
            self.cap_depth = cv2.VideoCapture(self.depth_video_path)

        self.timer.start(1000 // self.fps_spinbox.value())
    def download_video(self):
        # Verifica se video e modelo sono stati selezionati
        check, mod = self.check_modality_for_processing()
        if not check:
            self.show_alert("Error", "Please select a video and a model before downloading the processed video.")
            return
        if self.cap_rgb:
            self.cap_rgb.release()
        if self.cap_depth:
            self.cap_depth.release()
        # Crea il video da scaricare in base alla modalità selezionata
        # create a video writer object
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Video", "", "Video Files (*.mp4 *.mkv)")
        if not save_path:
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

        if mod == 'RGB':
            out = cv2.VideoWriter(save_path, fourcc, self.fps_spinbox.value(), (1280, 360))
            self.cap_rgb = cv2.VideoCapture(self.rgb_video_path)
            #salva il video originale frame | processed_frame
            while True:
                ret, frame_rgb = self.cap_rgb.read()
                if not ret:
                    break
                processed_frame = Run_rgb(self.model, frame_rgb, self.device)
                #processed_frame = frame_rgb
                frame_rgb = cv2.resize(frame_rgb, (640, 360))
                
                processed_frame = cv2.resize(processed_frame, (640, 360))
                frame = np.hstack((frame_rgb, processed_frame))
                out.write(frame)
            out.release()
            self.show_alert("Success", "The processed video has been saved successfully.")
        elif mod == 'DEPTH':
            out = cv2.VideoWriter(save_path, fourcc, self.fps_spinbox.value(), (1280, 360))
            self.cap_depth = cv2.VideoCapture(self.depth_video_path)
            #salva il video originale frame | processed_frame
            while True:
                ret, frame_depth = self.cap_depth.read()
                if not ret:
                    break
                processed_frame = Run_depth(self.model, frame_depth, self.device)
                #processed_frame = frame_depth
                frame_depth = cv2.resize(frame_depth, (640, 360))
                
                processed_frame = cv2.resize(processed_frame, (640, 360))
                frame = np.hstack((frame_depth, processed_frame))
                out.write(frame)
            out.release()
            self.show_alert("Success", "The processed video has been saved successfully.")
        else:
            out = cv2.VideoWriter(save_path, fourcc, self.fps_spinbox.value(), (1920, 360))
            self.cap_rgb = cv2.VideoCapture(self.rgb_video_path)
            self.cap_depth = cv2.VideoCapture(self.depth_video_path)
            #salva il video originale frame | processed_frame
            while True:
                ret_rgb, frame_rgb = self.cap_rgb.read()
                ret_depth, frame_depth = self.cap_depth.read()
                if not ret_rgb or not ret_depth:
                    break
                processed_frame = Run_datafusion(self.model, frame_depth, frame_rgb, self.device)
                #processed_frame = frame_depth
                frame_rgb = cv2.resize(frame_rgb, (640, 360))
                frame_depth = cv2.resize(frame_depth, (640, 360))
                processed_frame = cv2.resize(processed_frame, (640, 360))
                frame = np.hstack((frame_rgb, frame_depth, processed_frame))
                out.write(frame)
            out.release()
            self.show_alert("Success", "The processed video has been saved successfully.")
    def show_alert(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    def update_frame(self):
        modality = self.modality_combo.currentText()
        ret_rgb, frame_rgb = self.cap_rgb.read() if self.cap_rgb else (False, None)
        ret_depth, frame_depth = self.cap_depth.read() if self.cap_depth else (False, None)

        if modality == 'RGB only' and ret_rgb:
            # Process and display RGB frame
            processed_frame = Run_rgb(self.model, frame_rgb, self.device)
            #processed_frame = frame_rgb
            self.display_frames(frame_rgb, processed_frame)

        elif modality == 'DEPTH only' and ret_depth:
            # Process and display Depth frame

            processed_frame = Run_depth(self.model, frame_depth, self.device)
            #processed_frame = frame_depth
            self.display_frames(frame_depth, processed_frame)

        elif modality == 'Data Fusion' and ret_rgb and ret_depth:
            # Process and display Data Fusion frame
            processed_frame = Run_datafusion(self.model, frame_depth, frame_rgb, self.device)
            #processed_frame = frame_depth
            self.display_frames((frame_rgb, frame_depth), processed_frame)

        else:
            self.timer.stop()
            if self.cap_rgb:
                self.cap_rgb.release()
            if self.cap_depth:
                self.cap_depth.release()
    def display_frames(self, original_frame, processed_frame):
        modality = self.modality_combo.currentText()

        if modality == 'RGB only':

            rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.rgb_frame_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        elif modality == 'DEPTH only':

            depth_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = depth_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(depth_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.depth_frame_label.setPixmap(QPixmap.fromImage(qt_image).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        elif modality == 'Data Fusion':

            frame_rgb, frame_depth = original_frame

            # Display the RGB original frame
            rgb_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image_rgb = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.rgb_frame_label.setPixmap(QPixmap.fromImage(qt_image_rgb).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Display the Depth original frame
            depth_frame = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2RGB)
            h, w, ch = depth_frame.shape
            bytes_per_line = ch * w
            qt_image_depth = QImage(depth_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.depth_frame_label.setPixmap(QPixmap.fromImage(qt_image_depth).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Display the processed frame
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = processed_rgb.shape
        bytes_per_line = ch * w
        qt_image_processed = QImage(processed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.processed_frame_label.setPixmap(QPixmap.fromImage(qt_image_processed).scaled(640, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def clear(self):
        self.rgb_frame_label.clear()
        self.depth_frame_label.clear()
        self.processed_frame_label.clear()
        self.rgb_frame_label.setText("RGB Frame")
        self.depth_frame_label.setText("Depth Frame")
        self.processed_frame_label.setText("Processed Frame")
        self.timer.stop()
        if self.cap_rgb:
            self.cap_rgb.release()
        if self.cap_depth:
            self.cap_depth.release()
        

    def modality_change(self):
        self.clear()
        modality = self.modality_combo.currentText()
        if modality == 'RGB only':
            self.rgb_video_button.show()
            self.depth_video_button.hide()
            self.rgb_frame_label.show()
            self.depth_frame_label.hide()
        elif modality == 'DEPTH only':
            self.rgb_video_button.hide()
            self.depth_video_button.show()
            self.rgb_frame_label.hide()
            self.depth_frame_label.show()
        elif modality == 'Data Fusion':
            self.rgb_video_button.show()
            self.depth_video_button.show()
            self.rgb_frame_label.show()
            self.depth_frame_label.show()
    def closeEvent(self, event):
        if self.cap_rgb:
            self.cap_rgb.release()
        if self.cap_depth:
            self.cap_depth.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = VideoProcessingGUI()
    gui.show()
    sys.exit(app.exec_())



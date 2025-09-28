import sys
import os
import cv2
import time
import uuid
import mss
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QVBoxLayout, QWidget, QHBoxLayout, QListWidget, QMessageBox,
                             QComboBox, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QWheelEvent
from PyQt5.QtCore import Qt, QTimer
from ultralytics import YOLO
import numpy as np


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self._zoom = 1.0
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid gray")

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self._zoom *= 1.1
        else:
            self._zoom /= 1.1
        self.update_image()

    def update_image(self):
        if hasattr(self, 'image'):
            resized = cv2.resize(self.image, None, fx=self._zoom, fy=self._zoom)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qimg))

    def set_image(self, image):
        self.image = image
        self._zoom = 1.0
        self.update_image()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection App")
        self.setWindowIcon(QIcon("icons/app.png"))
        self.resize(1500, 800)

        self.model_paths = {
            "YOLOv8n-basic": "YOLO/yolov8n.pt",
            "YOLOv8s-basic": "YOLO/yolov8s.pt",
            "YOLOv8n-custom": "runs/detect/yolov8n_custom/weights/best.pt",
            "YOLOv8s-custom": "runs/detect/yolov8s_custom/weights/best.pt"
        }
        self.model = YOLO(self.model_paths["YOLOv8n-basic"])
        self.image_label = ImageLabel()
        self.results_list = QListWidget()
        self.detected_objects = []

        self.model_selector = QComboBox()
        self.model_selector.addItems(self.model_paths.keys())
        self.model_selector.currentIndexChanged.connect(self.change_model)

        self.open_image_btn = QPushButton("Відкрити зображення")
        self.open_image_btn.setIcon(QIcon("icons/image.png"))
        self.open_image_btn.clicked.connect(self.load_image)

        self.open_video_btn = QPushButton("Відкрити відео")
        self.open_video_btn.setIcon(QIcon("icons/video.png"))
        self.open_video_btn.clicked.connect(self.load_video)

        self.stream_btn = QPushButton("Стрім (IP / дрон)")
        self.stream_btn.setIcon(QIcon("icons/stream.png"))
        self.stream_btn.clicked.connect(self.start_stream)

        self.save_btn = QPushButton("Зберегти результати")
        self.save_btn.setIcon(QIcon("icons/save.png"))
        self.save_btn.clicked.connect(self.save_results)

        self.pause_btn = QPushButton("Пауза")
        self.pause_btn.clicked.connect(self.toggle_pause)

        self.forward_btn = QPushButton(">>")
        self.forward_btn.clicked.connect(self.skip_forward)

        self.backward_btn = QPushButton("<<")
        self.backward_btn.clicked.connect(self.skip_backward)

        self.exit_btn = QPushButton("Вийти")
        self.exit_btn.setIcon(QIcon("icons/exit.png"))
        self.exit_btn.clicked.connect(self.close)

        for btn in [self.open_image_btn, self.open_video_btn, self.stream_btn, self.save_btn,
                    self.pause_btn, self.forward_btn, self.backward_btn, self.exit_btn]:
            btn.setStyleSheet("font-size: 16px; padding: 8px;")

        buttons = QVBoxLayout()
        buttons.addWidget(QLabel("Оберіть модель:"))
        buttons.addWidget(self.model_selector)
        for btn in [self.open_image_btn, self.open_video_btn, self.stream_btn,
                    self.save_btn, self.pause_btn, self.forward_btn,
                    self.backward_btn, self.exit_btn]:
            buttons.addWidget(btn)

        right_layout = QVBoxLayout()
        right_layout.addLayout(buttons)
        right_layout.addWidget(QLabel("Результати"))
        right_layout.addWidget(self.results_list)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.paused = False
        self.output_writer = None
        self.output_dir = None
        self.video_path = None
        self.start_time = None

    def change_model(self):
        model_name = self.model_selector.currentText()
        self.model = YOLO(self.model_paths[model_name])

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Вибрати зображення")
        if path:
            # Перевірка розширення
            if not any(path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]):
                QMessageBox.warning(self, "Помилка формату", "Файл не є зображенням.")
                return
            image = cv2.imread(path)
            if image is None:
                QMessageBox.critical(self, "Помилка", "Не вдалося завантажити зображення.")
                return
            results = self.model(image)[0]
            image = self.display_results(image, results)
            self.image_label.set_image(image)

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Вибрати відео")
        if self.video_path:
            # Перевірка розширення
            if not any(self.video_path.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
                QMessageBox.warning(self, "Помилка формату", "Файл не є відео.")
                return
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Помилка", "Не вдалося відкрити відеофайл.")
                return
            self.prepare_output()
            self.timer.start(30)
            self.start_time = time.time()

    def start_screen_capture(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.timer.timeout.disconnect()
        self.timer.timeout.connect(self.update_screen_frame)
        self.timer.start(30)
        self.prepare_output()
        self.start_time = time.time()

    def start_stream(self):
        choice, ok = QInputDialog.getItem(
            self, "Тип стріму", "Оберіть джерело:",
            ["Ввести IP-адресу", "Захоплення екрана"],
            editable=False
        )

        if ok:
            if choice == "Ввести IP-адресу":
                ip, ok = QInputDialog.getText(self, "Введення IP", "Введіть посилання на стрім:")
                if ok and ip:
                    self.cap = cv2.VideoCapture(ip)
                    if not self.cap.isOpened():
                        QMessageBox.critical(self, "Помилка", "Не вдалося відкрити стрім.")
                        return
                    self.prepare_output()
                    self.timer.start(30)
            elif choice == "Захоплення екрана":
                self.start_screen_capture()

    def prepare_output(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("results", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        elif hasattr(self, "monitor"):
            width = self.monitor["width"]
            height = self.monitor["height"]
            fps = 30
        else:
            QMessageBox.warning(self, "Помилка", "Неможливо визначити розміри відео.")
            return

        self.output_writer = cv2.VideoWriter(
            os.path.join(self.output_dir, "output.mp4"), fourcc, fps, (width, height)
        )
        self.detected_objects.clear()

    def update_frame(self):
        if self.paused or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            if self.cap:
                self.cap.release()
            if self.output_writer:
                self.output_writer.release()
            return

        results = self.model(frame)[0]
        frame = self.display_results(frame, results)

        if self.output_writer:
            self.output_writer.write(frame)

        fps_text = f"FPS: {1 / (time.time() - self.start_time):.2f}"
        self.start_time = time.time()
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.image_label.set_image(frame)

    def display_results(self, frame, results):
        self.results_list.clear()
        for r in results.boxes:
            cls_id = int(r.cls[0].item())
            label = self.model.names[cls_id]
            conf = r.conf[0].item()
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            color = (0, 255, 0) if conf >= 0.6 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            self.results_list.addItem(f"{label}: {conf:.2f}")
            self.detected_objects.append(f"{label}: {conf:.2f}")
        return frame

    def save_results(self):
        if not self.output_dir:
            QMessageBox.warning(self, "Помилка", "Немає результатів для збереження")
            return
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, "w") as f:
            for obj in self.detected_objects:
                f.write(obj + "\n")
        QMessageBox.information(self, "Збережено", f"Результати збережено в:\n{self.output_dir}")

    def toggle_pause(self):
        self.paused = not self.paused

    def skip_forward(self):
        if self.cap and self.video_path:
            frame_no = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 30)

    def skip_backward(self):
        if self.cap and self.video_path:
            frame_no = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 30))

    def update_screen_frame(self):
        img = np.array(self.sct.grab(self.monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = self.model(frame)[0]
        frame = self.display_results(frame, results)

        if self.output_writer:
            self.output_writer.write(frame)

        fps_text = f"FPS: {1 / (time.time() - self.start_time):.2f}"
        self.start_time = time.time()
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        self.image_label.set_image(frame)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
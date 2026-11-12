import sys
import os
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from PIL import Image, ImageDraw, ImageFont


class AdminClient(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("道路病害监控大屏 - 独立推理端")
        self.resize(1000, 800)
        self.shared_dir = 'shared_results'
        self.last_processed_file = ""

        # 【关键】PyQt 自己加载模型
        print("大屏端正在加载模型...")
        self.model = torch.hub.load('./', 'custom', path='weights/best_fixed.pt', source='local')
        self.model.conf = 0.35
        self.model.iou = 0.45
        print("大屏端模型加载完毕！")

        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #121212; color: white;")
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("🌐 道路病害智能监测总控大屏")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #00FF00; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        main_display_frame = QFrame()
        main_display_frame.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333; border-radius: 12px;")
        main_display_layout = QVBoxLayout(main_display_frame)

        self.img_label = QLabel("等待终端上传原图...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("color: #666; font-size: 20px;")
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_display_layout.addWidget(self.img_label)
        main_layout.addWidget(main_display_frame, stretch=4)

        self.status_label = QLabel("系统就绪 | 独立推理监听中...")
        self.status_label.setStyleSheet(
            "color: #888; font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_for_new_images)
        self.timer.start(500)

    def check_for_new_images(self):
        if not os.path.exists(self.shared_dir): return
        files = [f for f in os.listdir(self.shared_dir) if f.endswith('.jpg')]
        if not files: return

        files.sort()
        latest_file = files[-1]

        if latest_file != self.last_processed_file:
            self.last_processed_file = latest_file

            latest_path = os.path.join(self.shared_dir, latest_file)
            self.process_and_show(latest_path)

            self.status_label.setText(f"📡 接收成功并完成推理 | 最新: {latest_file}")
            self.status_label.setStyleSheet(
                "color: #00FF00; font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")

    def process_and_show(self, file_path):
        # 读取原图
        img = cv2.imread(file_path)
        if img is None: return

        # 推理
        results = self.model(img)
        df = results.pandas().xyxy[0]

        # 画图
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size=24)
        except:
            font = ImageFont.load_default()

        for index, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cls_name = row['name']
            conf = row['confidence']

            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
            label = f"{cls_name} {conf:.2f}"
            text_bbox = draw.textbbox((x1, y1 - 28), label, font=font)
            draw.rectangle(text_bbox, fill=(0, 255, 0))
            draw.text((x1, y1 - 28), label, font=font, fill=(0, 0, 0))

        # 显示到 UI
        q_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 这里不重新推理，只做简单的控件重置，避免耗费资源
        pass


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    ex = AdminClient()
    ex.show()
    sys.exit(app.exec_())
from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

class UI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Laptop and Monitor Recognizer")
        self.setGeometry(100, 100, 800, 600)
        
        self.image_path = None
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.central_widget.setStyleSheet("background-color: black;")

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.setup_title()
        self.setup_image_label()
        self.setup_result_label()
        self.setup_buttons()

    def setup_title(self):
        self.label_title = QLabel("Laptop and Monitor Recognition")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_title.setStyleSheet("color: red; font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        self.layout.addWidget(self.label_title)

    def setup_image_label(self):
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_image)

    def setup_result_label(self):
        self.label_result = QLabel()
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setStyleSheet("color: white; font-size: 18px;")
        self.layout.addWidget(self.label_result)

    def setup_buttons(self):
        self.button_upload = QPushButton("Upload Image")
        self.layout.addWidget(self.button_upload)

        self.button_train = QPushButton("Train Model")
        self.layout.addWidget(self.button_train)

        self.button_predict = QPushButton("Predict Image")
        self.layout.addWidget(self.button_predict)

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((400, 400), Image.LANCZOS)
        image = image.convert("RGBA")
        
        data = image.tobytes("raw", "RGBA")
        qim = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qim)
        
        self.label_image.setPixmap(pixmap)

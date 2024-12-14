import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPropertyAnimation
from train import TrainWindow

class ArrowWidget(QWidget):
    def __init__(self, parent=None):
        super(ArrowWidget, self).__init__(parent)
        self.setGeometry(0, 0, 100, 100)
        self.setStyleSheet("background-color: transparent;")

    def move_left_animation(self):
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(500)
        anim.setStartValue(QRect(0, 0, 100, 100))
        anim.setEndValue(QRect(-50, 0, 100, 100))
        anim.start(QPropertyAnimation.DeleteWhenStopped)

class PredictWindow(QMainWindow):
    def __init__(self, parent=None):
        super(PredictWindow, self).__init__(parent)
        self.setWindowTitle("Predict Image")
        self.setGeometry(100, 100, 400, 300)

        self.btnUploadImage = QPushButton("Upload Image", self)
        self.btnUploadImage.clicked.connect(self.upload_image)
        self.btnUploadImage.setStyleSheet("QPushButton { background-color: #DCDCDC; color: white; border: 2px solid navy; border-radius: 10px; }"
                                          "QPushButton:hover { background-color: #a9a9a9; }")

        self.lblImagePath = QLabel("", self)
        self.lblImagePath.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.btnUploadImage)
        layout.addWidget(self.lblImagePath)
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def upload_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if fileName:
            self.lblImagePath.setText(f"Selected file: {fileName}")

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 600, 400)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.title_label = QLabel("Object Identifier", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 76px; font-weight: bold; font-family: Arial, sans-serif; margin-top: 20px;")

        self.btnTrain = QPushButton("Train", self)
        self.btnTrain.clicked.connect(self.open_train_window)
        self.btnTrain.setStyleSheet("QPushButton { font-weight: bold; font-family: Arial, sans-serif; font-size: 30px; color: white; background-color: navy; border: none; }"
                                    "QPushButton:hover { background-color: darkblue; }")

        self.btnPredict = QPushButton("Predict", self)
        self.btnPredict.clicked.connect(self.open_predict_window)
        self.btnPredict.setStyleSheet("QPushButton { font-weight: bold; font-family: Arial, sans-serif; font-size: 30px; color: white; background-color: navy; border: none; }"
                                      "QPushButton:hover { background-color: darkblue; }")

        button_size = (250, 100)
        self.btnTrain.setFixedSize(*button_size)
        self.btnPredict.setFixedSize(*button_size)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.btnTrain)
        buttonLayout.addWidget(self.btnPredict)
        buttonLayout.setAlignment(Qt.AlignCenter)

        self.setStyleSheet("QMainWindow { background-color: #DCDCDC; border-top: 120px solid navy; }")

        layout = QVBoxLayout(self.centralWidget)
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        layout.addLayout(buttonLayout)

        self.arrowWidget = ArrowWidget(self.centralWidget)
        layout.addWidget(self.arrowWidget, alignment=Qt.AlignTop | Qt.AlignLeft)
        
    def open_train_window(self):
        self.train_window = TrainWindow(self)
        self.train_window.closed.connect(self.move_arrow_left)
        self.train_window.show()

    def open_predict_window(self):
        self.predict_window = PredictWindow(self)
        self.predict_window.show()
    
    def move_arrow_left(self):
        self.arrowWidget.move_left_animation()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

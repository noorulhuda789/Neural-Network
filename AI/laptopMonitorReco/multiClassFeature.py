import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QRect, QPropertyAnimation
import numpy as np
from UL.train import TrainWindow
from DataStore.loadingMultiData import *
from models.MultiModel import *

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

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Main Window")
        self.setGeometry(300, 300, 800, 700)

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.title_label = QLabel("Object Identifier", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 76px; font-weight: bold; font-family: Arial, sans-serif; margin-top: 20px; color:white;")

        self.btnTrain = QPushButton("Train", self)
        self.btnTrain.clicked.connect(self.open_train_window)
        self.btnTrain.setStyleSheet("QPushButton { font-weight: bold; font-family: Arial, sans-serif; font-size: 30px; color: white; background-color: navy; border: none; }"
                                    "QPushButton:hover { background-color: darkblue; }")

        self.btnPredict = QPushButton("Predict", self)
        self.btnPredict.clicked.connect(self.upload_image_and_predict)
        self.btnPredict.setStyleSheet("QPushButton { font-weight: bold; font-family: Arial, sans-serif; font-size: 30px; color: white; background-color: navy; border: none; }"
                                      "QPushButton:hover { background-color: darkblue; }")

        button_size = (250, 100)
        self.btnTrain.setFixedSize(*button_size)
        self.btnPredict.setFixedSize(*button_size)

        self.lblImagePath = QLabel("", self)
        self.lblImagePath.setAlignment(Qt.AlignCenter)
        self.lblImagePath.setWordWrap(True)

        self.image_label = QLabel("", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.prediction_label = QLabel("Prediction: None", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.btnTrain)
        buttonLayout.addWidget(self.btnPredict)
        buttonLayout.setAlignment(Qt.AlignCenter)

        self.setStyleSheet("QMainWindow { background-color: #DCDCDC; border-top: 150px solid navy; }")

        layout = QVBoxLayout(self.centralWidget)
        layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.lblImagePath)
        layout.addWidget(self.image_label)
        layout.addWidget(self.prediction_label)
        layout.addLayout(buttonLayout)

        self.arrowWidget = ArrowWidget(self.centralWidget)
        layout.addWidget(self.arrowWidget, alignment=Qt.AlignTop | Qt.AlignLeft)

    def open_train_window(self):
        self.train_window = TrainWindow(self)
        self.train_window.closed.connect(self.move_arrow_left)
        self.train_window.show()
        np.random.seed(1)

    def upload_image_and_predict(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.xpm *.jpg *.jpeg);;All Files (*)", options=options)
        if fileName:
            #self.lblImagePath.setText(f"Selected file: {fileName}")
            pixmap = QPixmap(fileName)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
            self.show_prediction(fileName)

    def show_prediction(self, file_path):
        
        parameters=load_parameters()
        class_names = loadClasses()
        prediction = predict_image(parameters,file_path,class_names)
        self.prediction_label.setText(f"Prediction: {prediction}")

    def move_arrow_left(self):
        self.arrowWidget.move_left_animation()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

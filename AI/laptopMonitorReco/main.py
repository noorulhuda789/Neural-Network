import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import numpy as np
from Training import *
import cProfile
import pstats
from UL.appUI import UI
from PyQt5.QtCore import Qt

class LaptopMonitorRecognizer(UI):
    def __init__(self):
        super().__init__()
        
        self.button_upload.clicked.connect(self.upload_image)
        self.button_upload.setStyleSheet("color: white; font-size: 18px;")
        self.button_upload.setCursor(Qt.PointingHandCursor)

        self.button_train.clicked.connect(self.train_model)
        self.button_train.setStyleSheet("color: white; font-size: 18px;")
        self.button_train.setCursor(Qt.PointingHandCursor)

        self.button_predict.clicked.connect(self.predict_image)
        self.button_predict.setStyleSheet("color: white; font-size: 18px;")
        self.button_predict.setCursor(Qt.PointingHandCursor)

    def upload_image(self):
        self.label_result.clear()
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if self.image_path:
            self.display_image(self.image_path)

    def train_model(self):
        np.random.seed(1)
        train_x, train_y = load_data()
        dimensions = [12288, 120, 60, 5, 1]
        L_layer_model(train_x, train_y, dimensions, num_iterations=2500, print_cost=True)

    def predict_image(self):
        if self.image_path:
            parameters = load_parameters()
            prediction = predict_image(parameters, self.image_path)
            self.label_result.setText(str(prediction))
        else:
            self.label_result.setText("Please upload an image first.")

def profile_code():
    cProfile.run('re.compile("foo|bar")', 'restats')
    with open('restats', 'rb') as f:
        stats = pstats.Stats('restats')
        stats.strip_dirs()
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()

def main():
    app = QApplication(sys.argv)
    window = LaptopMonitorRecognizer()
    window.show()
    app.exec_()
    profile_code()

if __name__ == "__main__":
    main()

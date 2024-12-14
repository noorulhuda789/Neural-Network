import sys
import pickle
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from DataStore.loadingMultiData import loadData, saveClasses
from models.MultiModel import L_layer_model

class TrainWindow(QDialog):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super(TrainWindow, self).__init__(parent)
        self.setWindowTitle("Train Model")
        self.setGeometry(400, 400, 800, 600)

        self.layout = QVBoxLayout(self)

        self.leftArrowButton = QPushButton("←", self)
        self.leftArrowButton.clicked.connect(self.close_window)
        self.leftArrowButton.setStyleSheet(
            "QPushButton { background-color: #DCDCDC; color: darkblue; border: 5px solid navy; border-radius: 10px; }"
            "QPushButton:hover { background-color: darkblue; color: white; }")
        self.layout.addWidget(self.leftArrowButton, alignment=Qt.AlignTop | Qt.AlignLeft)

        self.titleLabel = QLabel("Train Model", self)
        self.titleLabel.setStyleSheet("font-size: 70px; font-weight: bold; color: black;")
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.titleLabel)

        self.inputBox = QLineEdit(self)
        self.inputBox.setPlaceholderText("Enter number of Inputs:")
        self.inputBox.textChanged.connect(self.generate_textboxes)
        self.layout.addWidget(self.inputBox)

        self.textboxLayout = QVBoxLayout()
        self.layout.addLayout(self.textboxLayout)

        self.trainButton = QPushButton("Train", self)
        self.trainButton.clicked.connect(self.train_model)
        self.trainButton.setStyleSheet(
            "QPushButton { background-color: navy; color: white; border: 5px solid navy; border-radius: 10px; font-size: 30px; font-weight: bold;}"
            "QPushButton:hover { background-color: darkblue; }")
        self.layout.addWidget(self.trainButton)

        self.setLayout(self.layout)
        self.selected_folders = []  
        self.textboxes = []  

    def close_window(self):
        self.closed.emit()
        self.close()

    def generate_textboxes(self, text=None):
        if text is None:
            text = self.inputBox.text()

        try:
            count = int(text)
        except ValueError:
            self.clear_textboxes()
            return

        self.clear_textboxes()

        for i in range(count):
            textboxLayout = QHBoxLayout()

            textbox = QLineEdit(self)
            textbox.setPlaceholderText(f"Label {i + 1} ")
            self.textboxes.append(textbox)

            uploadFolderButton = QPushButton("Upload Folder", self)
            uploadFolderButton.clicked.connect(lambda checked, text=textbox, layout=textboxLayout: self.upload_folder(text, layout))
            uploadFolderButton.setStyleSheet(
                "QPushButton { background-color: navy; color: white; border: 2px solid navy; border-radius: 10px; font-size: 20px; }"
                "QPushButton:hover { background-color: darkblue; }")

            textboxLayout.addWidget(textbox)
            textboxLayout.addWidget(uploadFolderButton)

            self.textboxLayout.addLayout(textboxLayout)

    def upload_folder(self, textbox, layout):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", "", options=options)
        if folder:
            # Check if there's already an address text box in the layout
            existing_address_box = None
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if isinstance(widget, QLineEdit) and widget.isReadOnly():
                    existing_address_box = widget
                    break

            if existing_address_box:
                # Update the existing address box with the new folder path
                existing_address_box.setText(folder)
            else:
                # Create a new address box and add it to the layout
                addressTextbox = QLineEdit(self)
                addressTextbox.setText(folder)
                addressTextbox.setReadOnly(True)
                layout.addWidget(addressTextbox)

            # Update the selected folders list
            textbox_index = self.textboxes.index(textbox)
            if len(self.selected_folders) > textbox_index:
                self.selected_folders[textbox_index] = folder
            else:
                self.selected_folders.append(folder)
            
            print(self.selected_folders)

    def train_model(self):
        if len(self.selected_folders) > 0:
            try:
                textbox_values = self.get_textbox_values()
                if any(value.strip() == "" for value in textbox_values):
                    self.show_warning("Please fill in all text boxes before training.")
                    return
                print("Training model...")
                train_x, train_y = loadData(px=64, folders=self.selected_folders)

                print(f"train_x shape: {train_x.shape}")
                print(f"train_y shape: {train_y.shape}")

                num_textboxes = int(self.inputBox.text())
                dimensions = [12288, 180, 60, 25, num_textboxes]
                L_layer_model(train_x, train_y, dimensions, num_iterations=2500, print_cost=True)
                print("Selected folders:", self.selected_folders)
                print("Network dimensions:", dimensions)

                # Save textbox inputs using pickle
                values = self.get_textbox_values()
                saveClasses(values)
                    
                #self.save_textbox_inputs()

                QMessageBox.information(self, "Success", "Model training completed successfully!")
                self.closed.emit()
                self.close()
            except Exception as e:
                self.show_error(f"An error occurred during training: {e}")
        else:
            self.show_warning("Please select folders before training.")

    def get_textbox_values(self):
        return [textbox.text() for textbox in self.textboxes]

    def show_error(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()

    def show_warning(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("No Folders Selected")
        msg_box.setText(message)
        msg_box.exec_()

    def clear_textboxes(self):
        self.textboxes.clear()
        self.selected_folders.clear()
        for i in reversed(range(self.textboxLayout.count())):
            widget = self.textboxLayout.itemAt(i).layout()
            if widget is not None:
                for j in reversed(range(widget.count())):
                    item = widget.itemAt(j)
                    if item.widget():
                        item.widget().deleteLater()
                widget.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    train_window = TrainWindow()
    train_window.show()
    sys.exit(app.exec_())

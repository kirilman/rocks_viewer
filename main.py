import os
import sys


from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from PyQt5 import QtGui, QtCore,QtWidgets

from PyQt5.QtWidgets import QRubberBand, QLabel, QApplication, QWidget, QMainWindow, QMenu, QAction, QFileDialog, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QRect
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import numpy as np
from asbestutills.plotter.plotting import plot_bboxs, plot_obounding_box
import qimage2ndarray
import cv2
from utills import qimage2array
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from model import Model
from utills import  prediction_to_detection, collect_max_size_from_detection
import supervision as sv
import ultralytics
import time
GEOMETRY_SIZE = (720,720)

PATH2MODELS = Path("./models/")
class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self._createActions()
        self._createMenuBar()
        self.path2model = list(Path(PATH2MODELS).glob("*"))[0]
        #Предсказания модели для разных значений conf
        self._dict_predicts = {}
        # Snip...
        self.h_layout = QHBoxLayout()

        image_label = QLabel(self)
        image_label.setScaledContents(True)
        #гистограмма
        self.hist_canvas = FigureCanvas(Figure())
        self.hist_canvas.figure.set_figheight(6)
        self.hist_canvas.figure.set_figwidth(10)
        self.ax_hist = self.hist_canvas.figure.add_subplot(121)
        self.ax_pdf = self.hist_canvas.figure.add_subplot(122)  # Ось для функции плотности

        self.slider = QtWidgets.QSlider(self)
        self.slider.setStyleSheet("""
            QSlider{
                background: #E3DEE2;
            }
            QSlider::groove:horizontal {  
                height: 10px;
                margin: 0px;
                border-radius: 5px;
                background: #B0AEB1;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 1px solid #E3DEE2;
                width: 17px;
                margin: -5px 0; 
                border-radius: 8px;
            }
            QSlider::sub-page:qlineargradient {
                background: #3B99FC;
                border-radius: 5px;
            }
        """)
        self.MIN_CONF = 0.05
        self.MAX_CONF = 0.8
        self.CONF_STEP = 0.2
        n = len(np.arange(self.MIN_CONF, self.MAX_CONF, self.CONF_STEP))
        self.slider.setMinimum(0)
        self.slider.setMaximum(n)
        self.slider.setTickInterval(1)
        self.slider.setValue(3)
        self.h_layout.addWidget(image_label)
        self.h_layout.addWidget(self.slider)
        self.h_layout.addWidget(self.hist_canvas)
        self.slider.valueChanged.connect(self.slider_value_changed)

        # self.setLayout(self.h_layout)
        # print(pixmap.width()/2)
        #
        # self.resize(int(pixmap.width()/100), int(pixmap.height()/100))
        # self.resize(100,100)
        #модель
        button_score = QPushButton()
        button_score.setText('Прогнозирование')
        combo_box = QComboBox(self)
        for file in os.listdir(PATH2MODELS):
            combo_box.addItem(file)
        button_score.clicked.connect(self.predict_varied)
        button_clear = QPushButton()
        button_clear.setText("Почистить распределение")
        button_clear.clicked.connect(self.clear_predict)
        button_image = QPushButton()
        button_image.setText("Улучшить качество изображения")
        button_image.clicked.connect(self.preprocess_image)
        combo_box.currentIndexChanged.connect(self.on_combobox_changed)

        self.v_layout = QVBoxLayout()
        self.v_layout.addWidget(button_score)
        self.v_layout.addWidget(button_clear)
        self.v_layout.addWidget(button_image)
        self.v_layout.addWidget(combo_box)
        self.v_layout.addLayout(self.h_layout)
        widget = QWidget()
        widget.setLayout(self.v_layout)
        self.setCentralWidget(widget)
        self.load_model()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.openAction)
        fileMenu.addAction(self.saveAction)
        fileMenu.addAction(self.exitAction)

        menuBar.addMenu(fileMenu)
        # Creating menus using a title
        editMenu = menuBar.addMenu("Edit")
        helpMenu = menuBar.addMenu("Help")
        editMenu.addAction(self.predictAction)


    def _createActions(self):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New")
        # Creating actions using the second constructor
        self.openAction = QAction("&Open...", self)
        self.saveAction = QAction("&Save", self)
        self.exitAction = QAction("&Exit", self)
        self.copyAction = QAction("&Copy", self)
        self.pasteAction = QAction("&Paste", self)
        self.predictAction = QAction("&Predict", self)

        self.cutAction = QAction("C&ut", self)
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)
        #обработчики
        self.openAction.triggered.connect(self.open_file)
        self.saveAction.triggered.connect(self.save_image)
        self.predictAction.triggered.connect(self.predict)


    def open_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open file")
        print(fname)
        if fname:
            pixmap = QPixmap(fname)

            pixmap = pixmap.scaled(*GEOMETRY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.h_layout.itemAt(0).widget().setPixmap(pixmap)
            print(self.h_layout.itemAt(0).widget().size()/2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.source_pixmap = pixmap.copy()

    def resize_images(self):
        # Масштабируем каждое изображение в QLabel
        for widget in self.h_layout.widget().count():
            if isinstance(widget, QLabel):
                pixmap = widget.pixmap()
                scaled_pixmap = pixmap.scaled(GEOMETRY_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                widget.setPixmap(scaled_pixmap)

    def load_model(self):
        self.model = Model()
        self.model.load(self.path2model)

    def predict(self, **kwargs):
        pixmap = self.h_layout.itemAt(0).widget().pixmap()
        qimage = pixmap.toImage()
        img = qimage2array(qimage)
        img = img.astype(np.uint8)
        t = time.time()
        out = self.model.predict(img, **kwargs)
        print(time.time() - t)
        image_box = self.draw_boxes(img, prediction_to_detection(out))
        qimage = qimage2ndarray.array2qimage(image_box)
        self.h_layout.itemAt(0).widget().setPixmap(QPixmap.fromImage(qimage))

    def predict_varied(self):
        if not self.source_pixmap:
            return
        pixmap = self.source_pixmap.copy()
        qimage = pixmap.toImage()
        image = qimage2array(qimage)
        self._conf_values = {}

        for i, conf in enumerate(np.arange(self.MIN_CONF, self.MAX_CONF, self.CONF_STEP)):
            self.model.load(self.path2model, conf )
            out = self.model.predict(image, imgsz=640, conf=conf)
            if isinstance(out, list):
                self._dict_predicts[conf] = out[0]
            else:
                self._dict_predicts[conf] = out
            self._conf_values[i] = conf

    def slider_value_changed(self, value):
        """
            Обработка измения позиция slider для порога вероятности сети
        """

        if not value in self._conf_values or not self.source_pixmap:
            return
        conf = self._conf_values[value]

        predict = prediction_to_detection(self._dict_predicts[conf])
        d = collect_max_size_from_detection(predict)
        N = len(d)
        # N = predict[0].obb.xyxyxyxyn.shape[0]
        # d = predict[0].obb.xywhr[:,[2,3]].max(axis=1).values.detach().cpu().numpy()
        print(value, conf, N)
        self.draw_hist(d, conf)
        pixmap = self.source_pixmap.copy()
        qimage = pixmap.toImage()
        img = qimage2array(qimage)
        image_box = self.draw_boxes(img, predict)
        qimage = qimage2ndarray.array2qimage(image_box)
        self.h_layout.itemAt(0).widget().setPixmap(QPixmap.fromImage(qimage))

    def draw_boxes(self, image, detections):
        """
        detections sv.Detection:
        """
        if len(detections) == 0:
            return print(detections)
        if isinstance(detections, list):
            detections = detections[0]
        image = image.copy()
        size = max(image.shape)
        t = int(size * 0.0025)
        if detections.mask is None:
            annotator = sv.OrientedBoxAnnotator()
            annotated_frame = annotator.annotate(
                scene=image.copy().astype(np.uint8),
                detections=detections,
            )
        else:
            annotator = sv.MaskAnnotator()
            print(type(detections.mask[0][0][0]), type(image[0][0][0]))
            annotated_frame = annotator.annotate(
                scene=image.copy().astype(np.uint8),
                detections=detections,
                # custom_color_lookup = np.array([123,0,255])
                # skip_label=True,
            )
        print(len(detections))

        # if anno.boxes:
        #     if len(anno.boxes.xyxyn) > 0:
        #         image_with_bbox = plot_bboxs(image, anno.boxes.xyxyn, t)
        # elif anno.obb:
        #     if len(anno.obb.xyxyxyxyn) > 0:
        #         N = anno.obb.xyxyxyxyn.shape[0]
        #         image_with_bbox = plot_obounding_box(
        #             image, anno.obb.xyxyxyxyn.reshape(N, -1).detach().cpu().numpy(), t, color = [255,0,0]
        #         )
        return annotated_frame

    def draw_hist(self, arr, conf_value):
        self.ax_hist.clear()
        self.ax_pdf.clear()
        mean = arr.mean()
        var  = arr.var()
        # Построение функции распределения
        sorted_data = np.sort(arr)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        self.ax_pdf.plot(sorted_data, yvals)
        self.ax_hist.hist(arr, bins = 30, edgecolor='black')
        self.ax_hist.text(0,100, str(mean))
        self.ax_hist.text(0, 200, str(var))
        self.ax_hist.set_title(f'Гистограмма, N = {len(arr)}, conf = {conf_value}')
        self.hist_canvas.draw()

    def clear_predict(self):
        current_index = self.slider.value()
        conf = self._conf_values[current_index]
        anno = self._dict_predicts[conf]
        d = anno[0].obb.xywhr[:,[2,3]].max(axis=1).values.detach().cpu().numpy()
        q = np.quantile(d,0.99)
        self.draw_hist(d[d<q])

    def preprocess_image(self):
        pixmap = self.source_pixmap.copy()
        qimage = pixmap.toImage()
        image = qimage2array(qimage)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpen_img = cv2.filter2D(image, -1, kernel)
        qimage = qimage2ndarray.array2qimage(sharpen_img)
        pixmap = QPixmap.fromImage(qimage)
        self.h_layout.itemAt(0).widget().setPixmap(pixmap)
        self.source_pixmap = pixmap.copy()

    def on_combobox_changed(self, index):
        # Получаем текст выбранного элемента
        selected_text = self.v_layout.itemAt(3).widget().itemText(index)
        self.path2model = PATH2MODELS / selected_text
        self.load_model()

    def save_image(self):
        pixmap = self.h_layout.itemAt(0).widget().pixmap()
        qimage = pixmap.toImage()
        img = qimage2array(qimage)
        img = img.astype(np.uint8)
        cv2.imwrite('save_image.jpeg', img)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import os
import cv2
import supervision as sv
import ultralytics
import detectron2
class Model:
    def __init__(self, task = "box"):
        self.model = ''
        self._task = task
    @property
    def task(self):
        return self._task
    @task.setter
    def task(self, value):
        if value not in ['box', 'bbox', 'segm']:
            raise ValueError("Invalid value")
        self._task = value

    def load(self, path2model, score = 0.2):
        dist_cfg = torch.load(path2model, map_location = 'cpu')
        if 'train_args' in dist_cfg:
            self.model = YOLO(model=path2model)
            self._task  = dist_cfg['train_args']['task']
            self.model.to('cpu')
        else:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
            # cfg.SOLVER.IMS_PER_BATCH = 8
            # cfg.SOLVER.BASE_LR = 0.00025
            # cfg.SOLVER.MAX_ITER = 5000  # We found that with a patience of 500, training will early stop before 10,000 iterations
            # cfg.SOLVER.CHECKPOINT_PERIOD = 1000
            cfg.SOLVER.STEPS = []
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            cfg.TEST.DETECTIONS_PER_IMAGE = 1000
            cfg.MODEL.DEVICE = 'cpu'
            cfg.MODEL.WEIGHTS = os.path.join(path2model)  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score)  # set a custom testing threshold
            self.model = DefaultPredictor(cfg)
            self._task = 'segm'
        del dist_cfg

    def predict(self, image, *args, **kwargs):
        if issubclass(self.model.__class__, ultralytics.models.yolo.model.YOLO):
            p = self.model.predict(image, *args, **kwargs, max_det = 2000)
        elif issubclass(self.model.__class__, detectron2.engine.defaults.DefaultPredictor):
            p = self.model(image)
        return p

# model = Model()
# model.load('./models/cascade-mask.pth', )
# model.load('./models/yolo_obb.pt', )
# print(type(model.model))

# img = cv2.imread('738165-product4201463.jpeg')
# p = model(img)
# print(p)
# if 'instances' in p:
#     print(type(p['instances']), issubclass(p['instances'].__class__, detectron2.structures.instances.Instances))
# # detections = sv.Detections.from_ultralytics(p[0])
# print(type(p), p.__class__)
# print(issubclass(p[0].__class__, ultralytics.engine.results.Results))

# bounding_box_annotator = sv.OrientedBoxAnnotator()
# annotated_frame = bounding_box_annotator.annotate(
#     scene=img.copy(),
#     detections=detections
# )
# fig = plt.figure()
# plt.imshow(annotated_frame)
# cv2.imwrite('anno.jpeg', annotated_frame)
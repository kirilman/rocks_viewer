import cv2
import numpy as np
import qimage2ndarray
import supervision as sv
from supervision import mask_to_polygons
import ultralytics
import detectron2
from typing import Union

def max_distance(x_coords, y_coords):
    r = (x_coords[:, np.newaxis] - x_coords) ** 2 + (
        y_coords[:, np.newaxis] - y_coords
    ) ** 2
    return np.sqrt(r.max())

def letterbox_image(image, expected_size=(736, 736)):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_img = np.full((eh, ew, 3), 128, dtype="uint8")
    # fill new image with the resized image and centered it
    new_img[
        (eh - nh) // 2 : (eh - nh) // 2 + nh, (ew - nw) // 2 : (ew - nw) // 2 + nw, :
    ] = image.copy()
    step_h = (eh - nh) // 2
    step_w = (ew - nw) // 2
    return new_img, nh, nw, step_h, step_w, scale


def qimage2array(qimage):
    arr = qimage2ndarray.recarray_view(qimage)
    shape = arr['r'].shape
    img = np.zeros((*shape, 3))
    img[:, :, 0] = arr['r']
    img[:, :, 1] = arr['g']
    img[:, :, 2] = arr['b']
    return img

def prediction_to_detection(prediction):
    """
        Convert to sv.Detections format
    """
    if isinstance(prediction, list):
        prediction = prediction[0]
    if len(prediction) == 0:
        return []
    if 'instances' in prediction:  # detecton2
        if issubclass(prediction['instances'].__class__, detectron2.structures.instances.Instances):
            detections = sv.Detections.from_detectron2(prediction)

    else:
        if issubclass(prediction.__class__, ultralytics.engine.results.Results):
            detections = sv.Detections.from_ultralytics(prediction)
    return detections

def collect_max_size_from_detection(detection:sv.Detections)-> np.ndarray:
    max_sizes = []
    if not detection.mask is None:
        for m in detection.mask:
            coords = mask_to_polygons(m)[0]
            d = max_distance(coords[:,0], coords[:,1])
            max_sizes.append(d)
    else:
        for m in detection.data['xyxyxyxy']:
            coords = m.reshape(-1)
            dx = np.sqrt((coords[0] - coords[2]) ** 2 + (coords[1] - coords[3]) ** 2)
            dy = np.sqrt((coords[2] - coords[4]) ** 2 + (coords[3] - coords[5]) ** 2)
            max_sizes.append(max(dx, dy))
    return np.array(max_sizes)
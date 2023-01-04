# Setup detectron2 logger
import torch, detectron2

# import some common libraries
import cv2
import numpy as np
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import Boxes
from detectron2.utils.visualizer import ColorMode, Visualizer

#custom scripts
from utils.formatter import bitmask_to_polygons

import warnings
warnings.filterwarnings("ignore")


#Paths
UPLOAD_FOLDER_PATH = "_uploads"
RESULT_FOLDER_PATH="_results"

#Detectron2 Model Configs
MODEL_CONFIG_PATH = "seg_model_weight_and_config/22_12_2022/config.yaml"
MODEL_WEIGHT_PATH = "seg_model_weight_and_config/22_12_2022/model_final.pth"
MODEL_SCORE_THRESHOLD = 0.6 # set a custom testing threshold
THING_CLASSES = [
                    "_background_",
                    "News",
                    "Corner",
                    "Tag",
                    "Puzzle",
                    "Advertisement",
                    "Announcement",
                    "Broadcast",
                    "StockMarket"
                ]

class Segmentation:
    def __init__(self):
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.UPLOAD_FOLDER = UPLOAD_FOLDER_PATH

        self.cfg = get_cfg()
        self.cfg.merge_from_file(MODEL_CONFIG_PATH)
        self.cfg.MODEL.WEIGHTS = MODEL_WEIGHT_PATH
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MODEL_SCORE_THRESHOLD  

        self.predictor = DefaultPredictor(self.cfg)

        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

        self.metadata.thing_classes = THING_CLASSES

        self.instance_mode = ColorMode.IMAGE


    def extract_instances(self, instances):
        boxes = instances.pred_boxes
        print(f"instances: {len(boxes)}")
        if isinstance(boxes, Boxes):
            boxes = boxes.tensor.cpu().numpy()
        else:
            boxes = np.asarray(boxes)

        scores = instances.scores
        pred_classes = instances.pred_classes

        labels = [self.metadata.thing_classes[i] for i in pred_classes]
        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

        #convert from bitmap mask to polygonal
        bit_masks=instances.pred_masks.cpu().numpy().astype("uint8")
        polygons = [bitmask_to_polygons(bit_mask) for bit_mask in bit_masks]


        return boxes, pred_classes, scores, labels, polygons

    def prediction(self, file_name):
        print("file_name:",file_name)

        image = cv2.imread(f"{UPLOAD_FOLDER_PATH}//{file_name}")
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        predictions = self.predictor(image)
        
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode, scale=1.2)

        instances = predictions["instances"].to("cpu")
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        boxes, pred_classes, scores, labels, polygons = self.extract_instances(predictions["instances"])

        vis_output.save(f"{RESULT_FOLDER_PATH}//{file_name}")


        result_data = {
            "filename":file_name,
            "boxes":  boxes.tolist(),
            "boxes_count":  len(boxes),
            "scores": scores.tolist(),
            "classes": pred_classes.tolist(),
            "labels": self.metadata.thing_classes,
            "polygons": polygons
        }
        

        return result_data
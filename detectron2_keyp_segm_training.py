# Setup detectron2 logger
import torch, torchvision
print(torch.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow

from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import numpy as np 
import skimage.io as io 
import pylab,json 
from tempfile import NamedTemporaryFile
import pycocotools.mask as mask_util
import os

from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

import json
import os
 
raw_dir = "./"
splits = ["val","test","train"]
label_dir = 'test/test.json'

for split in splits:
  with open(os.path.join(raw_dir,split,split+".json"),'r') as coco_file:
    coco_labels = json.load(coco_file)

  anns = coco_labels['annotations']
  for ann in anns:
    bbox = ann['bbox']
    ann['keypoints'] = [bbox[0]+bbox[2]//2,bbox[1]+bbox[3]//2,2]
    ann['num_keypoints'] = 1
  
  with open(os.path.join(raw_dir,split,split+"2.json"), 'w') as file:
      json.dump(coco_labels, file)

if __name__ == '__main__':
	register_coco_instances("skku_unloading_coco_train", {}, "./train/train2.json", "./train/")
	skku_train_metadata = MetadataCatalog.get("skku_unloading_coco_train")
	skku_train_dataset_dicts = DatasetCatalog.get("skku_unloading_coco_train")

	register_coco_instances("skku_unloading_coco_test", {}, "./test/test2.json", "./test/")
	skku_test_metadata = MetadataCatalog.get("skku_unloading_coco_test")
	skku_test_dataset_dicts = DatasetCatalog.get("skku_unloading_coco_test")

	register_coco_instances("skku_unloading_coco_val", {}, "./val/val2.json", "./val/")
	skku_val_metadata = MetadataCatalog.get("skku_unloading_coco_val")
	skku_val_dataset_dicts = DatasetCatalog.get("skku_unloading_coco_val")

	skku_test_metadata.keypoint_names = ['grasp']
	skku_train_metadata.keypoint_names = ['grasp']
	skku_val_metadata.keypoint_names = ['grasp']

	skku_test_metadata.keypoint_flip_map = []
	skku_train_metadata.keypoint_flip_map = []
	skku_val_metadata.keypoint_flip_map = []

	#import random

	#for d in random.sample(skku_train_dataset_dicts, 3):
	#    img = cv2.imread(d["file_name"])
	#    visualizer = Visualizer(img[:, :, ::-1], metadata=skku_train_metadata, scale=0.35)
	#    vis = visualizer.draw_dataset_dict(d)
	#    cv2_imshow(vis.get_image()[:, :, ::-1])

	from detectron2.engine import DefaultTrainer
	from detectron2.config import get_cfg
	import os

	# Evaluation code from
	#https://github.com/facebookresearch/detectron2/issues/810#issuecomment-596194293

	# More detailed implementation at
	#https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
	##or
	#https://tshafer.com/blog/2020/06/detectron2-eval-loss

	from detectron2.engine import HookBase
	from detectron2.data import build_detection_train_loader
	import detectron2.utils.comm as comm
	import torch
	from detectron2.evaluation import COCOEvaluator, inference_on_dataset


	class MyTrainer(DefaultTrainer):
	  @classmethod
	  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
	    if output_folder is None:
	      output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
	    return COCOEvaluator(dataset_name, cfg, True, output_folder)

	class ValidationLoss(HookBase):
	    def __init__(self, cfg):
	        super().__init__()
	        self.cfg = cfg.clone()
	        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
	        self._loader = iter(build_detection_train_loader(self.cfg))
	        
	    def after_step(self):
	        data = next(self._loader)
	        with torch.no_grad():
	            loss_dict = self.trainer.model(data)
	            
	            losses = sum(loss_dict.values())
	            assert torch.isfinite(losses).all(), loss_dict

	            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
	                                 comm.reduce_dict(loss_dict).items()}
	            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
	            if comm.is_main_process():
	                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
	                                                 **loss_dict_reduced)

	cfg = get_cfg()
	cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # for instance segmentation
	#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("skku_unloading_coco_train",)
	cfg.DATASETS.TEST = ("skku_unloading_coco_test",)   # no metrics implemented for this dataset
	cfg.DATASETS.VAL = ("skku_unloading_coco_val",)   # no metrics implemented for this dataset
	
	cfg.TEST.EVAL_PERIOD = 500
	cfg.TEST.KEYPOINT_OKS_SIGMAS = [1.0]
	
	cfg.DATALOADER.NUM_WORKERS = 4
	
	#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # initialize from model zoo
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 4 classes (box, icebox, pouch, sack)
	#   ENABLE KEYPOINT REGRESSION
	cfg.MODEL.KEYPOINT_ON = True
	cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 1
	cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0
	#   ENABLE INSTANCE SEGMENTATION
	cfg.MODEL.MASK_ON =  True
	cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 0.5
	
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.02
	cfg.SOLVER.CHECKPOINT_PERIOD = 5000
	# ********* Learning rate calc: https://github.com/facebookresearch/detectron2/issues/1128#issuecomment-774175041
	num_gpu = 1
	bs = (num_gpu * 2)
	cfg.SOLVER.BASE_LR = 0.01 * bs / 128  # pick a good LR
	# ********
	cfg.SOLVER.MAX_ITER = 9000   # 300 iterations seems good enough, but you can certainly train longer


	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = MyTrainer(cfg) #DefaultTrainer(cfg)
	val_loss = ValidationLoss(cfg)  
	trainer.register_hooks([val_loss])
	# swap the order of PeriodicWriter and ValidationLoss
	trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
	trainer.resume_or_load(resume=False)
	trainer.train()

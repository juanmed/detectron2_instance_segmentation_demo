# Setup detectron2 logger
import torch, torchvision
print(torch.__version__)

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import numpy as np 
import skimage.io as io 
import pylab,json 
from tempfile import NamedTemporaryFile
import pycocotools.mask as mask_util
import os
import cv2
import matplotlib.pyplot as plt

def xyxy2xywh(bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ] 

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd, bbox):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    #print(len(contours))
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        if len(segmentation)>4:
            segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    #x, y, max_x, max_y = multi_poly.bounds
    #width = max_x - x
    #height = max_y - y
    #bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return segmentations

    # format all outputs into a list of dicts compatible with COCO

if __name__ == '__main__':

    register_coco_instances("skku_unloading_coco_test", {}, "./skku_unloading_coco_test/trainval.json", "./skku_unloading_coco_test/images/")
    skku_test_metadata = MetadataCatalog.get("skku_unloading_coco_test")
    skku_test_dataset_dicts = DatasetCatalog.get("skku_unloading_coco_test")

    cfg = get_cfg()
    cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("skku_unloading_coco_train",)
    cfg.DATASETS.TEST = ("skku_unloading_coco_val",)   # no metrics implemented for this dataset
    cfg.DATASETS.VAL = ("skku_unloading_coco_val",)   # no metrics implemented for this dataset
    cfg.TEST.EVAL_PERIOD = 200
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 2500   # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 3 classes (data, fig, hazelnut)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("skku_unloading_coco_test", )
    predictor = DefaultPredictor(cfg)



    #Inspired from
    # https://www.immersivelimit.com/create-coco-annotations-from-scratch

    """
    detection_res = []
    is_crowd = 0
    for k, d in enumerate(skku_test_dataset_dicts):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        outputs = outputs["instances"].to("cpu")

        bboxes = outputs.pred_boxes
        scores = outputs.scores
        classes = outputs.pred_classes
        masks = outputs.pred_masks

        for i,(bbox, score, class_, mask) in enumerate(zip(bboxes, scores, classes, masks)):

            annotation = create_sub_mask_annotation(mask.numpy(), d["image_id"], class_, i, is_crowd, xyxy2xywh(bbox))
            #annotations.append(annotation)

            detection_res.append({
                'score': score.item(),
                'category_id': class_.item(),
                'bbox': xyxy2xywh(bbox),
                'image_id': d["image_id"],
                'segmentation': annotation
            })

    print(detection_res)
    """

    from detectron2.utils.visualizer import ColorMode
    import random

    for d in random.sample(skku_test_dataset_dicts, 5):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=skku_test_metadata, 
                       scale=0.4, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(v.get_image()[:, :, ::-1])
        print(d["file_name"])
        fig.savefig(d["file_name"].split("/")[-1])


    # json file in coco format, original annotation data
    anno_file = './skku_unloading_coco_test/trainval.json'
    coco_gt = COCO(anno_file)
     
     # Use GT box as prediction box for calculation, the purpose is to get detection_res

    """
    detection_res = []
    with open(anno_file, 'r') as f:
        json_file = json.load(f)
    annotations = json_file['annotations']
    detection_res = []
    for i, anno in enumerate(annotations):
        detection_res.append({
            'score': 1,
            'category_id': anno['category_id'],
            'bbox': anno['bbox'],
            'image_id': anno['image_id']
        })
        if i < 1 :
          print( anno['category_id'], anno['image_id'])
    """
    """
    with NamedTemporaryFile(suffix='.json') as tf:
             # Due to subsequent needs, first convert detection_res to binary and then write it to the json file
        content = json.dumps(detection_res).encode(encoding='utf-8')
        tf.write(content)
        res_path = tf.name
     
             # loadRes will generate a new COCO type instance based on coco_gt and return
        coco_dt = coco_gt.loadRes(res_path)
     
        cocoEval = COCOeval(coco_gt, coco_dt, 'segm')  # use 'bbox' for bbox mAP or 'segm' for instance segmentation mAP
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    print(cocoEval.stats)

    with NamedTemporaryFile(suffix='.json') as tf:
             # Due to subsequent needs, first convert detection_res to binary and then write it to the json file
        content = json.dumps(detection_res).encode(encoding='utf-8')
        tf.write(content)
        res_path = tf.name
     
             # loadRes will generate a new COCO type instance based on coco_gt and return
        coco_dt = coco_gt.loadRes(res_path)
     
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')  # use 'bbox' for bbox mAP or 'segm' for instance segmentation mAP
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()     

    print(cocoEval.stats)

    import time
    times = []
    for i in range(1000):
        start_time = time.time()
        outputs = predictor(im)
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print("Average(sec):{:.4f},fps:{:.4f}".format(mean_delta, fps))
    """
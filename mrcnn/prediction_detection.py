import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import glob
import os.path as osp

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize_edited as visualize
from mrcnn.visualize_edited import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import skimage.draw
import cv2

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

from mrcnn.config import Config

class ObjectDetectionConfig(Config):
    """Configuration for training on the buildings and ground detection.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "objectDetection"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + building-ground classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class InferenceConfig(ObjectDetectionConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# my object detection config:
config = InferenceConfig()
data_DIR_images = "F:\\MaskRCNN\\Mask_RCNN\\myproject\\objectDetection\\finalDatasets\\valing\\images\\"
data_DIR_lables = "F:\\MaskRCNN\\Mask_RCNN\\myproject\\objectDetection\\finalDatasets\\valing\\labels\\"
data_DIR_thermals = "F:\\MaskRCNN\\Mask_RCNN\\myproject\\objectDetection\\finalDatasets\\valing\\thermals\\"
data_DIR_predicts = "F:\\MaskRCNN\\Mask_RCNN\\myproject\\objectDetection\\finalDatasets\\valing\\predicts\\"


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()



DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

class_names = ['background','building_roof', 'ground_cars', 'building_facade', 'ground_cars', 'building_roof']

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)



names = [x for x in os.listdir(data_DIR_images) if ".jpg" in x] #new

for name in names:
    name = name.split(".")[0]

    image = skimage.io.imread(data_DIR_images + name+".jpg")
    thermal = skimage.io.imread(data_DIR_thermals + name+".jpg")

    image = np.concatenate((image, thermal), axis=2)

    gt_image = cv2.imread(data_DIR_lables+name+".png")
    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results

    r = results[0]
    pred_image = visualize.return_save(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Predictions")

    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
    pred_image = pred_image[:, :, ::-1]
    # pred_image = np.array(pred_image,dtype=np.uint8)
    # pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

    plt.imshow(pred_image)
    plt.show()
    plt.imshow(image)
    plt.show()
    plt.imshow(gt_image)
    plt.show()

    print(gt_image[900,400])
    print(pred_image[1000,400])
    # cv2.imwrite(data_DIR_predicts+name+".png", pred_image)
    skimage.io.imsave(data_DIR_predicts+name+".png", pred_image)

"""
Mask R-CNN
Train on the object detection dataset.
The dataset has five classes, also have the background.
1, building_roof
2, building_facade
3, building_equipment
4, ground_cars
5, ground_equipment

background->> ground_roads

Maybe later will implement color splash effect.


Original Written by Waleed Abdulla
Edited by Yu Hou

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 objectDetect.py train --dataset=/path/to/balloon/dataset --weights=coco

    *** in my code ***
    train --dataset=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets --weights=coco
    train --dataset=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\reducedDatasets --weights=coco


    # Resume training a model that you had trained earlier
    python3 objectDetect.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 objectDetect.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    mycode
    splash --weights=last --image=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\reducedDatasets\valing\images\20181210_093540_631_R_2.jpg

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

    #??splash --weights=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\newcoco.h5 --image=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\val

    # Run COCO evaluatoin on the last model you trained

    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
    evaluate --dataset=F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\reducedDatasets --weights=last
"""

"""
Import the useful libraries
"""
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib
import matplotlib.pyplot as plt
import re
import time
import random

CHANGE_MODEL = True
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
if CHANGE_MODEL:
	from mrcnn import model_my as modellib, utils
else:
	from mrcnn import model as modellib, utils


# add more functions
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_objectdetection_0024.h5")
ADD_THERMAL_CHANNEL = True

'''
############################################################
#  Configurations
############################################################
'''
class ObjectDetectionConfig(Config):
    """Configuration for training on the buildings and ground detection.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "objectDetection"

    GPU_COUNT = 2

    if ADD_THERMAL_CHANNEL:
    	IMAGE_CHANNEL_COUNT = 4
    	MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 115.6]) # maybe change for later

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + building-ground classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 250

    BACKBONE = 'resnet50'    

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 512

    TRAIN_ROIS_PER_IMAGE = 128

    MAX_GT_INSTANCES = 10

    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    RPN_ANCHOR_SCALES = (32, 64, 128, 256)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


'''
############################################################
#  This is for visualization
############################################################
'''


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


'''
############################################################
#  Dataset
############################################################
'''

class ObjectDetectionDataset(utils.Dataset):

    def load_objects(self, dataset_dir, subset):
        """Load a subset of the Building-ground elements dataset.
        important variances:

        1-> dataset_dir: Root directory of the dataset.
        2-> subset: Subset to load: train or val

        """
        # Add classes. We have five classes to add.
        
        self.add_class("objectDetection", 1, "ground_cars")
        self.add_class("objectDetection", 2, "building_facade")
        self.add_class("objectDetection", 3, "building_equipment")
        self.add_class("objectDetection", 4, "building_roof")
        self.add_class("objectDetection", 5, "ground_equipment")


        # Train or validation dataset?
        assert subset in ["training", "valing"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\training'
        # dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\valing'
        # annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))
        annotationsPath = os.path.join(dataset_dir, "label") #new
        labelNames = [x for x in os.listdir(annotationsPath) if ".png" in x] #new

        '''
        Add images - New Version
        '''

        # Add images
        print("start to load images")

        # to reduce the dataset for multiple training use
        # middle = int(len(labelNames) / 3)
        # print("middle",middle)
        # labelNames = labelNames[middle:]

        for name in labelNames:

            # load_mask() needs the image size to convert polygons to masks, the new is to give the
            # path directly to polygons
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.

            imageName = "/rgb/" + name[:-3] + "jpg"
            labelName = "/label/"+ name

            # # get paths
            image_path = dataset_dir + imageName
            label_path = dataset_dir + labelName

            # # get images
            label_image = skimage.io.imread(label_path)
            # image = skimage.io.imread(image_path)

            height, width = label_image.shape[:2]

            self.add_image(
                "objectDetection",
                image_id=imageName,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=label_image)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        XXXXX class_ids: a 1D array of class IDs of the instance masks.
        class_ids: a 1D array of class IDs of the instance masks.

        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "objectDetection":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        '''
        new Version
        '''
        label_image = info["polygons"]
        '''
        show the image to check
        '''
        # imgplot = plt.imshow(label_image)
        # plt.show()



        '''
        # 1, building_roof , red
        # 2, building_facade , green
        # 3, building_equipment , brown
        # 4, ground_cars , blue
        # 5, ground_equipment [128   0 128] pink
        '''


        classList = {1: np.array([128, 0, 0]),
                     2: np.array([0, 128, 0]),
                     3: np.array([128, 128, 0]),
                     4: np.array([0, 0, 128]),
                     5: np.array([128, 0, 128]),
                     }


        hasPolygonClass = []

        for element in classList:

            image = label_image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.where(image == classList[element], 1, 255)
            image = np.sum(image, axis=-1)
            image = np.where(image == 3, 1, 0)
            if np.sum(image) != 0:
                hasPolygonClass.append(element)


        mask = np.zeros([info["height"], info["width"], len(hasPolygonClass)], dtype=np.uint8)
        class_ids = []

        for i in range(len(hasPolygonClass)):
            image = label_image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.where(image == classList[hasPolygonClass[i]], 1, 255)
            image = np.sum(image, axis=-1)
            image = np.where(image == 3, 1, 0)
            # check if load the mask right
            # plt.imshow(image, cmap=plt.get_cmap('gray'))
            # plt.title(hasPolygonClass[i], fontsize=9)
            # plt.show()

            mask[:, :, i] = image
            class_ids.append(hasPolygonClass[i])

        mask = np.array(mask)
        class_ids = np.array(class_ids)
        # print("mask",mask.shape)
        # print("classid",class_ids)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "objectDetection":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
            or [H, W, 4] array
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # thermal channel added
        if ADD_THERMAL_CHANNEL:
            thermal_path = self.image_info[image_id]['path'].replace('rgb', 'thermal')
            thermal_image = skimage.io.imread(thermal_path)           
            thermal_image = thermal_image[:,:,0].reshape((image.shape[0], image.shape[1], 1))
            image = np.concatenate((image,thermal_image),axis=2)
            return image

        # If grayscale. Convert to RGB for consistency.
        # if image.ndim != 3:
        #     image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4 and not ADD_THERMAL_CHANNEL:
            image = image[..., :3]
        return image


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ObjectDetectionDataset()    # This is a class
    dataset_train.load_objects(args.dataset, "training")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ObjectDetectionDataset()  # This is a class
    dataset_val.load_objects(args.dataset, "valing")
    dataset_val.prepare()

    '''
    show and display random samples
    '''
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    print("####")
    print("all image loaded", dataset_train.image_ids)
    print("images selected to show", image_ids)
    for image_id in image_ids:
        # print(image_id)
        image = dataset_train.load_image(image_id)
        print(image.shape)

        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img_s = [image]
        # titles = ['img']
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
        # print(dataset_train.class_names)
        # print(class_ids)
        # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names, limit=5)


    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training **ALL** networks!")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='all')


'''
Not sure if will be used, but keep it here
'''
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash



'''
Not sure if will be used, but keep it here
'''
def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


'''
############################################################
#  Training
############################################################
'''


if __name__ == '__main__':
    import argparse
    import tensorflow as tf

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    print("GPU Available: ", tf.test.is_gpu_available())

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ObjectDetectionConfig()
    else:
        class InferenceConfig(ObjectDetectionConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights:
	    if args.weights.lower() == "coco":
	        weights_path = COCO_WEIGHTS_PATH
	        # Download weights file
	        if not os.path.exists(weights_path):
	            utils.download_trained_weights(weights_path)
	    elif args.weights.lower() == "last":
	        # Find last trained weights
	        weights_path = model.find_last()
	    elif args.weights.lower() == "imagenet":
	        # Start from ImageNet trained weights
	        weights_path = model.get_imagenet_weights()
	    else:
	        weights_path = args.weights

    	# Load weights
	    print("Loading weights ", weights_path)
	    if args.weights.lower() == "coco":
	        # Exclude the last layers because they require a matching
	        # number of classes
	        model.load_weights(weights_path, by_name=True, exclude=[
	            "mrcnn_class_logits", "mrcnn_bbox_fc",
	            "mrcnn_bbox", "mrcnn_mask"])
	    else:
	    	if ADD_THERMAL_CHANNEL:
	    		model.load_weights(weights_path, by_name=True, exclude=['conv1'])
	    	elif CHANGE_MODEL:
	    		include = ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 
	    		'fpn_p2', 'fpn_p3', 'fpn_p4', 'rpn_model', 'rpn_conv_shared', 'rpn_class_raw', 
	    		'rpn_bbox_pred', 'mrcnn_mask_conv1', 'mrcnn_mask_bn1', 'mrcnn_mask_conv2', 
	    		'mrcnn_mask_bn2', 'mrcnn_class_conv1', 'mrcnn_class_bn1', 'mrcnn_mask_conv3', 
	    		'mrcnn_mask_bn3', 'mrcnn_class_conv2', 'mrcnn_class_bn2', 'mrcnn_mask_conv4', 
	    		'mrcnn_mask_bn4', 'mrcnn_bbox_fc, mrcnn_mask_deconv' ,' mrcnn_class_logits', 'mrcnn_mask']
	    		model.load_weights(weights_path, by_name=True, include=include)
	    	else:
	    		model.load_weights(weights_path, by_name=True)


    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "evaluate":

        '''
        actually these are done above
        '''
        # Create model object in inference mode.
        # model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        # model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
        # print(weights_path)
        # model.load_weights(weights_path, by_name=True)

        '''
        '''
        class_names = ['bg', 'ground_cars', 'building_facade', 'building_equipment', 'building_roof', 'ground_equipment']
        # Load a random image from the images folder
        IMAGE_DIR = os.path.join(args.dataset, "rgb")
        file_names = next(os.walk(IMAGE_DIR))[2]
        print(file_names[:10])
        # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

        for name in file_names:
            image = skimage.io.imread(os.path.join(IMAGE_DIR, name))
            # thermal channel added
            if ADD_THERMAL_CHANNEL:
                thermal_path = os.path.join(args.dataset, "thermal")
                thermal_path = os.path.join(thermal_path, name)
                thermal_image = skimage.io.imread(thermal_path)           
                thermal_image = thermal_image[:,:,0].reshape((image.shape[0], image.shape[1], 1))
                image = np.concatenate((image,thermal_image),axis=2)

            # Run detection
            print(image.shape)
            results = model.detect([image], verbose=1)

            # Visualize results
            r = results[0]
            print(r['rois'].shape, len(r['class_ids']))
            visualize.display_instances(image[..., :3], r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

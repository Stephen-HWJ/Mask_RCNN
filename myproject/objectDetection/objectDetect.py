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

    # Resume training a model that you had trained earlier
    python3 objectDetect.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 objectDetect.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

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


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# add more functions
from mrcnn import visualize
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 2

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + building-ground classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

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
        # self.add_class("building_roof",         1, "building_roof")
        # self.add_class("building_facade",       2, "building_facade")
        # self.add_class("building_equipment",    3, "building_equipment")
        # self.add_class("ground_cars",           4, "ground_cars")
        # self.add_class("ground_equipment",      5, "ground_equipment")

        self.add_class("objectDetection", 1, "building_roof")
        self.add_class("objectDetection", 2, "building_facade")
        self.add_class("objectDetection", 3, "building_equipment")
        self.add_class("objectDetection", 4, "ground_cars")
        self.add_class("objectDetection", 5, "ground_equipment")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\training'
        # dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\valing'
        annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))
        annotationsPath = os.path.join(dataset_dir, "labels") #new
        labelNames = [x for x in os.listdir(annotationsPath) if ".png" in x] #new

        '''
        The annotation is a dict:
        dict_keys(['_via_attributes', '_via_settings', '_via_img_metadata'])
        <1> annotations['_via_attributes']
        index one stores a dictionary describing the options for classes:
        1, building_roof
        2, building_facade
        3, building_equipment
        4, ground_cars
        5, ground_equipment

        <2> annotation['_via_img_metadata']
        index two stores a dictionary describing the datasets of all annotations:
            #   {'20181210_093147_331_R.JPG3510965':

                    {   'filename': '28503151_5b5b7ec140_b.jpg',
                        'regions': 
                            [   {
                                    'region_attributes': { {'objects'}: '3'},
                                    'shape_attributes': 
                                    {
                                        'all_points_x': [...],
                                        'all_points_y': [...],
                                        'name': 'polygon'}
                                    }
                                },  # this is the first region
                                {
                                    'region_attributes': { {'objects'}: '3'},
                                    'shape_attributes': 
                                    {
                                        'all_points_x': [...],
                                        'all_points_y': [...],
                                        'name': 'polygon'}
                                    }
                                },  # this is the second region
                                ... more regions ...
                            ] # regions are stored in a list[]  

                        'size': 100202
                    }
                  '20181210_093147_331_R.JPG3510965': {}
                  '20181210_093147_331_R.JPG3510965': {}
                  ... more images ...
                 }
        <3> annotation['_via_settings']
        Dont worry about this dictionary, this is configuration about the annotation software.     

        '''

        annotations = annotations['_via_img_metadata']  # we just extract the metadata
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.

        annotations = [a for a in annotations if a['regions']]
        # print(annotations[0])

        '''
        Add images-Old Version
        
        '''
        # # Add images
        # for a in annotations:
        #     # print(a)
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #
        #     '''
        #     Below is an example version:::
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']]
        #     '''
        #
        #     # polygons = [r['shape_attributes'] for r in a['regions']]
        #     '''
        #     Pay attention here, the example version will put polygons to "polygons=a" in "self.add_image"
        #     but here needs some changes, we will give a to "self.add_image" directly, cuz we will use a later.
        #     '''
        #
        #     # print(polygons)
        #
        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     '''
        #     show the image to check
        #     '''
        #     # imgplot = plt.imshow(image)
        #     # plt.show()
        #     height, width = image.shape[:2]
        #
        #     self.add_image(
        #         "objectDetection",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=a)


        '''
        Add images - New Version
        '''

        # Add images
        for name in labelNames:

            # load_mask() needs the image size to convert polygons to masks, the new is to give the
            # path directly to polygons
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            imageName = name.split(".")[0] + ".jpg"
            image_path = os.path.join(dataset_dir, imageName)

            label_path = os.path.join(dataset_dir, name)
            label_image = skimage.io.imread(label_path)

            image = skimage.io.imread(image_path)
            '''
            show the image to check
            '''
            # imgplot = plt.imshow(image)
            # plt.show()
            height, width = image.shape[:2]

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
        old version
        '''
        #
        # '''
        # # Get the x, y coordinaets of points of the polygons that make up
        # # the outline of each object instance. These are stores in the
        # # shape_attributes (see json format above)
        # '''
        # a = info["polygons"]
        # polygons = [r['shape_attributes'] for r in a['regions']]
        # classes = [r['region_attributes'] for r in a['regions']]
        #
        # mask = np.zeros([info["height"], info["width"], len(polygons)],
        #                 dtype=np.uint8)
        #
        # for i, p in enumerate(polygons):
        #     # Get indexes of pixels inside the polygon and set them to 1
        #     rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        #     mask[rr, cc, i] = 1
        #
        # # Return mask, and array of class IDs of each instance. Since we have
        # # one class ID only, we return an array of 1s XXXXX
        # # Actually this is not an array of 1s. This should be adjusted.
        #
        # # class_ids = np.ones([mask.shape[-1]], dtype=np.int32) # This will not be used, cuz not all are 1.
        # class_ids = np.array([int(r['Objects']) for r in classes])


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

        classList = {[128,   0,   0]: 1,
                     [0,   128,   0]: 2,
                     [128, 128,   0]: 3,
                     [0,     0, 128]: 4,
                     [128,   0, 128]: 5,
                     }

        mask = np.zeros([info["height"], info["width"], len(classList.keys())], dtype=np.uint8)

        for element in classList:

            image = label_image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.where(image == classList[element], 1, 255)
            image = np.sum(image, axis=-1)
            image = np.where(image == 3, 1, 0)
            # # check if load the mask right
            # plt.imshow(image, cmap=plt.get_cmap('gray'))
            # plt.show()

            mask[:, :, element - 1] = image

        class_ids = np.array([int(r['Objects']) for r in classes])


        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "objectDetection":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ObjectDetectionDataset()    # This is a class
    dataset_train.load_objects(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ObjectDetectionDataset()  # This is a class
    dataset_val.load_objects(args.dataset, "val")
    dataset_val.prepare()

    '''
    show and display random samples
    '''
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    print("####")
    print(dataset_train.image_ids)
    print(image_ids)
    for image_id in image_ids:
        print(image_id)
        image = dataset_train.load_image(image_id)

        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


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
    parser.add_argument('--weights', required=True,
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
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    # elif args.command == "splash":
    #     detect_and_color_splash(model, image_path=args.image,
    #                             video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))

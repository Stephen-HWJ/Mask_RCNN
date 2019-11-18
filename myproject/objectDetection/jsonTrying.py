import json
import os
import skimage
import matplotlib.pyplot as plt
import numpy as np

dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\train'
dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject\objectDetection\objectsDatasets\\val'
annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))
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

annotations = annotations['_via_img_metadata'] # we just extract the metadata
annotations = list(annotations.values())  # don't need the dict keys

# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.

annotations = [a for a in annotations if a['regions']]
# print(annotations[0])

# Add images
for a in annotations:
    # print(a)
    # Get the x, y coordinaets of points of the polygons that make up
    # the outline of each object instance. These are stores in the
    # shape_attributes (see json format above)
    # The if condition is needed to support VIA versions 1.x and 2.x.
    '''
    if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in a['regions']]
    '''

    # print(a['regions'])
    polygons = [r['shape_attributes'] for r in a['regions']]
    classes = [r['region_attributes'] for r in a['regions']]    ### not know if the order matters???????

    # print(len(polygons))
    # print(len(classes))
    # print(polygons)
    # print(classes)


    # load_mask() needs the image size to convert polygons to masks.
    # Unfortunately, VIA doesn't include it in JSON, so we must read
    # the image. This is only managable since the dataset is tiny.
    image_path = os.path.join(dataset_dir, a['filename'])
    image = skimage.io.imread(image_path)
    '''
    show the image to check
    '''
    # imgplot = plt.imshow(image)
    # plt.show()
    height, width = image.shape[:2]

    # self.add_image(
    #     "balloon",
    #     image_id=a['filename'],  # use file name as a unique image id
    #     path=image_path,
    #     width=width, height=height,
    #     polygons=polygons)
    '''
    load masks
    '''
    # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
    #                 dtype=np.uint8)
    mask = np.zeros([height, width, len(polygons)],dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1


    print(classes)
    class_ids = np.ones([mask.shape[-1]], dtype=np.int32)
    classes = np.array([int(r['Objects']) for r in classes])
    print(classes)
    print(class_ids)



    imgplot = plt.imshow(mask[:,:,0])
    plt.show()
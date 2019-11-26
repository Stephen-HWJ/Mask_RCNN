
import json
import os
import numpy as np
import csv


if __name__ == '__main__':
    original = "referenceSW.csv"
    target = "convertannotationsSW.csv"
    result = "convertannotationsSW1.csv"
    alldata=[]
    dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject'
    path = os.path.join(dataset_dir, original)

    import csv

    fileXsizePair = {}
    with open(path, mode='r') as csv_file:
        original_reader = csv.DictReader(csv_file)


        for row in original_reader:
            fileXsizePair[row['filename']]= row['file_size']
    print(fileXsizePair)

    path = os.path.join(dataset_dir, target)

    temp = []


    with open(path, mode='r') as csv_file:
        target_reader = csv.DictReader(csv_file)
        filenamestemp = target_reader.fieldnames

        for row in target_reader:

            name = row["filename"]
            row["file_size"]=fileXsizePair[name]

            temp.append(row)

    path = os.path.join(dataset_dir, result)
    with open(path, mode='w') as csv_file:
        # print(filenamestemp)
        filenamestemp = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes','region_attributes']
        writer = csv.DictWriter(csv_file, fieldnames=filenamestemp)

        writer.writeheader()
        for row in temp:
            writer.writerow(row)



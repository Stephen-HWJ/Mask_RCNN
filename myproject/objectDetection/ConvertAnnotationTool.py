
import json
import os
import numpy as np
import csv

def findTheCode (name):
    if name == "building_roof":
        return 1
    if name == "building_facade":
        return 2
    if name == "building_equipment":
        return 3
    if name == "ground_cars":
        return 4
    if name == "ground_equipment":
        return 5
def giveFileName(name):
    list = name.split("/")
    return list[-1]


if __name__ == '__main__':
    original = "annotationsSW.csv"
    converted = "convertannotationsSW.csv"
    alldata=[]
    dataset_dir = 'F:\MaskRCNN\Mask_RCNN\myproject'
    path = os.path.join(dataset_dir, original)

    import json

    with open(path) as file:  # Use file to refer to the file object
        lines = file.readlines()[1:]

        for line in lines:
            # print(line)
            transDict = {"filename": None, "region_count": None,
                         "Object": {"ID": [], "all_points_x": [], "all_points_y": []}}

            filename = giveFileName(str(line).split(',')[0])
            transDict["filename"]=filename

            content = ",".join(line.split(',')[1:])
            # print(content)

            json_acceptable_string = content.replace("'", "\"")
            objects = json.loads(json_acceptable_string)
            transDict["region_count"]=len(objects)



            for object in objects:



                if object["type"] == "rotated_box":
                    # print(object)
                    transDict["Object"]["ID"].append(findTheCode(object["label"]))  # each object has a type id

                    h = object["coordinates"]["height"]
                    w = object["coordinates"]["width"]
                    x = object["coordinates"]["cx"]
                    y = object["coordinates"]["cy"]
                    rot = object["coordinates"]["rot"]

                    allx = [x, x + w,   x + w, x]
                    ally = [y, y,       y + h, y + h]

                    transDict["Object"]["all_points_x"].append(allx)
                    transDict["Object"]["all_points_y"].append(ally)

                elif object["type"] == "polygon":
                    transDict["Object"]["ID"].append(findTheCode(object["label"]))  # each object has a type id
                    number = int(len(object["coordinates"])/2)

                    allx = [0]*number
                    ally = [0]*number
                    for key in object["coordinates"].keys():
                        if key[0] == 'x':
                            location=int(key[1:])
                            allx[location - 1]=object["coordinates"][key]
                        if key[0] == 'y':
                            location=int(key[1:])
                            ally[location - 1] = object["coordinates"][key]

                    transDict["Object"]["all_points_x"].append(allx)
                    transDict["Object"]["all_points_y"].append(ally)



                elif object["type"] == "rectangle":

                    transDict["Object"]["ID"].append(findTheCode(object["label"]))  # each object has a type id
                    number = 4      # because this is rectangle, it has four points

                    h = object["coordinates"]["height"]
                    w = object["coordinates"]["width"]
                    x = object["coordinates"]["x"]
                    y = object["coordinates"]["y"]
                    allx = [x, x+w, x, x+w]
                    ally = [y, y,   y+h,y+h]

                    transDict["Object"]["all_points_x"].append(allx)
                    transDict["Object"]["all_points_y"].append(ally)

                elif object["type"] == "line":

                    transDict["Object"]["ID"].append(findTheCode(object["label"]))  # each object has a type id
                    number = int(len(object["coordinates"]) / 2)

                    allx = [0] * number
                    ally = [0] * number
                    for key in object["coordinates"].keys():
                        if key[0] == 'x':
                            location = int(key[1:])
                            allx[location - 1] = object["coordinates"][key]
                        if key[0] == 'y':
                            location = int(key[1:])
                            ally[location - 1] = object["coordinates"][key]

                    transDict["Object"]["all_points_x"].append(allx)
                    transDict["Object"]["all_points_y"].append(ally)

            # print(len(transDict["Object"]["ID"]),len(transDict["Object"]["all_points_x"]))
            alldata.append(transDict)


    path = os.path.join(dataset_dir, converted)


    with open(path,"w") as f:
        # writer = csv.writer(f)
        # writer.writerows("filename","file_size","file_attributes","region_count","region_id","region_shape_attributes","region_attributes")
        f.writelines("filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes\n")
        # print(alldata)


        for element in alldata:


            # print(element)
            filename = element["filename"]
            file_size = ' '
            file_attributes = {}
            region_count = element["region_count"]
            name = "polygon"
            for i in range(region_count):

                region_id = i
                region_attributes = {"Objects": str(element["Object"]["ID"][i])}
                all_points_x = element["Object"]["all_points_x"][i]
                all_points_y = element["Object"]["all_points_y"][i]

                region_shape_attributes = "\"" + str({"name": name,"all_points_x": all_points_x, "all_points_y": all_points_y}).replace("\'","\"\"") + "\""
                region_attributes = "\"" + str(region_attributes).replace("\'","\"\"") + "\""
                string = ",".join([filename, file_size, str(file_attributes), str(region_count), str(region_id), region_shape_attributes,region_attributes])
                f.writelines(string)
                f.writelines("\n")




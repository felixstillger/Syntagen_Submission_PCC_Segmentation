classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

palette = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


# important vor voc prompts:
add_classes=["tv",]
add_palette=[[0, 64, 128],]


pascal_id2color = {}
for key in range(21):
    #class_index = classes.index(key)
    pascal_id2color[classes[key]] = palette[key]
    
pascal_name2id = {}
for key in range(21):
    #class_index = classes.index(key)
    pascal_name2id[classes[key]] = key 
    
# extended class assignment    
pascal_name2id["table"]=pascal_name2id["diningtable"]
pascal_name2id["tv"]=pascal_name2id["tvmonitor"]
pascal_name2id["monitor"]=pascal_name2id["tvmonitor"]
pascal_name2id["plant"]=pascal_name2id["pottedplant"]
pascal_name2id["pot"]=pascal_name2id["pottedplant"]
pascal_name2id["plane"]=pascal_name2id["aeroplane"]
pascal_name2id["airplane"]=pascal_name2id["aeroplane"]
    
pascal_palette = [value for color in palette for value in color]
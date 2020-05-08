import json

train_filename = 'D:/电影缓存/数据集/COCO2017/2017/annotations_trainval2017/annotations/instances_train2017.json'
val_filename = 'D:/电影缓存/数据集/COCO2017/2017/annotations_trainval2017/annotations/instances_val2017.json'
#['info', 'images', 'licenses', 'categories', 'annotations']
with open(train_filename) as f:
    data = json.load(f)
    print('train')
    print(data.keys())
    print(len(data['annotations']))
    print(len(data['images']))
    print(len(data['categories']))

with open(val_filename) as f:
    data = json.load(f)
    print('val')
    print(data.keys())
    print(len(data['annotations']))
    print(len(data['images']))
    print(len(data['categories']))

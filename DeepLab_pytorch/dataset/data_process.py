from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch


def _resize_data(src_img, src_seg, resized_width):
    pad_w = int(resized_width - src_img.shape[0])
    pad_h = int(resized_width - src_seg.shape[1])
    resized_img = np.pad(src_img, ((0, pad_w), (0, pad_h), (0, 0)), mode='constant')
    resized_seg = np.pad(src_seg, ((0, pad_w), (0, pad_h)), mode='constant')
    resized_img = resized_img / 255.0
    resized_img = resized_img - np.mean(resized_img)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img.transpose((2, 0, 1))

    resized_img = torch.from_numpy(resized_img)
    resized_seg = torch.from_numpy(resized_seg)

    return resized_img, resized_seg


class VOCDataset(Dataset):
    def __init__(self, file_list, image_folder, seg_folder, resized_width, transform=None):
        self.transform = transform
        self.file_list = file_list
        self.image_folder = image_folder
        self.seg_folder = seg_folder
        self.resized_width = resized_width
        self.filename_list = [x.strip('\n') for x in open(file_list, 'r')]

    def __getitem__(self, index):
        image_path = self.image_folder + str(self.filename_list[index]) + '.jpg'
        seg_path = self.seg_folder + str(self.filename_list[index]) + '.png'
        image = Image.open(image_path)
        seg = Image.open(seg_path)
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image)
        seg = np.array(seg)
        if image.shape[0] != seg.shape[0] or image.shape[1] != seg.shape[1]:
            raise RuntimeError('Shape mismatched between image and label.')
        return _resize_data(image, seg, self.resized_width)

    def __len__(self):
        return len(self.filename_list)


def get_loader(data_list, image_folder, seg_folder, width, batch_size, num_workers, shuffle=True):
    train_dataset = VOCDataset(data_list, image_folder, seg_folder, width)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                             pin_memory=True)
    return data_loader


def test():
    data_list = 'data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    image_folder = 'data/VOC_2012/VOCdevkit/VOC2012/JPEGImages/'
    seg_folder = 'data/VOC_2012/VOCdevkit/VOC2012/SegmentationClass/'
    train_dataset = VOCDataset(data_list, image_folder, seg_folder, 512)
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False)
    data_iter = iter(train_data_loader)
    image, seg = next(data_iter)
    print('image shape: ', end='')
    print(image.shape)
    print('image type: ', end='')
    print(type(image.shape))
    print('seg shape: ', end='')
    print(seg.shape)
    print('seg type: ', end='')
    print(type(seg.shape))


def count():
    data_list = 'data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
    image_folder = 'data/VOC_2012/VOCdevkit/VOC2012/JPEGImages/'
    seg_folder = 'data/VOC_2012/VOCdevkit/VOC2012/SegmentationClass/'
    train_dataset = VOCDataset(data_list, image_folder, seg_folder, 512)
    train_data_loader = DataLoader(train_dataset, batch_size=150, num_workers=4, shuffle=True)
    data_iter = iter(train_data_loader)
    _, seg = next(data_iter)
    seg = seg.numpy()
    cnt = np.zeros(256)
    import matplotlib.pyplot as plt

    for i in tqdm(range(150)):
        for j in range(512):
            for k in range(512):
                color = seg[i][j][k]
                if cnt[color] == 0:
                    cnt[color] = 1
    for i in range(256):
        if cnt[i] != 0:
            print(i, cnt[i])


if __name__ == '__main__':
    count()

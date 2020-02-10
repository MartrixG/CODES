import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def show_seg(seg):
    plt.imshow(seg, cmap='hot')
    plt.axis('off')
    plt.show()


def parse_function(serialized):
    features = tf.parse_single_example(
        serialized,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'segmentation': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        }
    )
    height = features['height']
    width = features['width']
    features['image_raw'] = tf.decode_raw(features['image_raw'], tf.uint8)
    features['segmentation'] = tf.decode_raw(features['segmentation'], tf.uint8)
    features['image_raw'] = tf.reshape(features['image_raw'], [height, width, 3])
    features['segmentation'] = tf.reshape(features['segmentation'], [height, width, 1])
    return features


def get_data(record_file, trained_image_width, shuffle=True):
    filenames = [record_file + '/' + i for i in os.listdir(record_file)]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_function)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    iterator = dataset.make_one_shot_iterator()
    tf_features = iterator.get_next()
    all_image = []
    all_seg = []
    print("Loading data.")
    with tf.Session() as sess:
        while True:
            try:
                image, seg = sess.run([tf_features['image_raw'], tf_features['segmentation']])
                all_image.append(image)
                seg = np.squeeze(seg, axis=2)
                all_seg.append(seg)
            except tf.errors.OutOfRangeError:
                print("Loaded data.")
                print("Total pictures:%d" % (len(all_image)))
                break
    return _resize_data(all_image, all_seg, trained_image_width)


def _resize_data(src_image, src_seg, trained_image_width):
    num = len(src_image)
    for i in range(num):
        # Image scaling
        # w, h, _ = src_image[i].shape
        # r = float(trained_image_width) / np.max([w, h])
        # resize_image.append(np.array(Image.fromarray(src_image[i]).resize((int(r * h), int(r * w)))))
        # resize_seg.append(np.array(Image.fromarray(src_seg[i], mode='L').resize((int(r * h), int(r * w)))))
        # pad_w = int(trained_image_width - resize_image[i].shape[0])
        # pad_h = int(trained_image_width - resize_image[i].shape[1])
        pad_w = int(trained_image_width - src_image[i].shape[0])
        pad_h = int(trained_image_width - src_seg[i].shape[1])
        src_image[i] = np.pad(src_image[i], ((0, pad_w), (0, pad_h), (0, 0)), mode='constant')
        src_seg[i] = np.pad(src_seg[i], ((0, pad_w), (0, pad_h)), mode='constant')
    return np.array(src_image), np.array(src_seg)


def test(filename, sha):
    train_input, train_output = get_data(filename, sha, shuffle=False)
    print(len(train_input))
    print(train_input[0].shape)
    print(type(train_input[0]))
#    show_image(train_input[0])
    matplotlib.image.imsave('input_lin.jpg', train_input[0])

    print(train_output[0].shape)
    print(type(train_output[0]))
#    show_seg(train_output[0])
    matplotlib.image.imsave('out_put_lin.png', train_output[0], cmap='hot')


if __name__ == "__main__":
    test("data/VOC_2012/tfrecord/train", 512)

import os
import sys
import math
import tensorflow as tf
import argparse
from dataset import image_reader
from six.moves import range

parser = argparse.ArgumentParser(description='args for making tfrecords')
parser.add_argument('--image_folder', '-f', type=str,
                    default='data/VOC_2012/VOCdevkit/VOC2012/JPEGImages')

parser.add_argument('--semantic_segmentation_folder', '-s',
                    default='data/VOC_2012/VOCdevkit/VOC2012/SegmentationClassRaw')

parser.add_argument('--list_folder', '-l',
                    default='data/VOC_2012/VOCdevkit/VOC2012/ImageSets/Segmentation')

parser.add_argument('--train_record', '-t', type=str, default="data/VOC_2012/tfrecord/train")

parser.add_argument('--val_record', '-v', type=str, default="data/VOC_2012/tfrecord/val")

parser.add_argument('--trainval_record', '-tv', type=str, default='data/VOC_2012/tfrecord/trainval')

args = parser.parse_args()

_NUM_SHARDS = 4


def switch_case(name):
    file_path = {
        'train': args.train_record,
        'val': args.val_record,
        'trainval': args.trainval_record
    }
    return file_path.get(name)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _load_images(dataset_split):
    dataset_name = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset_name)
    filename_list = [x.strip('\n') for x in open(dataset_split, 'r')]
    images_num = len(filename_list)

    return dataset_name, filename_list, images_num


def _convert_dataset(dataset, filenames, num_images):
    """Converts the specified dataset split to TFRecord format.

    Args:
    dataset_split: The dataset split (e.g., train, test).

    Raises:
    RuntimeError: If loaded image and label have different shape.
    """
    num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

    jpg_reader = image_reader.ImageReader('jpg', channels=3)
    png_reader = image_reader.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = switch_case(dataset) + '/%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()
                # Read the image.
                image_format = "jpg"
                label_format = "png"
                image_filename = args.image_folder + '/' + filenames[i] + '.' + image_format
                image_data = tf.gfile.GFile(image_filename, 'rb').read()
                height, width = jpg_reader.read_image_dims(image_data)
                image_raw = jpg_reader.decode_image(image_data).tostring()
                # Read the semantic segmentation annotation.
                seg_filename = args.semantic_segmentation_folder + '/' + filenames[i] + '.' + label_format
                seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
                seg_height, seg_width = png_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                segmentation = png_reader.decode_image(seg_data).tostring()
                # Convert to tf example.
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image_raw': _bytes_list_feature(image_raw),
                    'segmentation': _bytes_list_feature(segmentation),
                    'height': _int64_feature(height),
                    'width': _int64_feature(width)
                }))
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    dataset_splits = tf.gfile.Glob(os.path.join(args.list_folder, '*.txt'))
    for each_dataset in dataset_splits:
        _dataset, _filenames, _num_images = _load_images(each_dataset)
        _convert_dataset(_dataset, _filenames, _num_images)
    print("All data has been converted to tfrecord.")

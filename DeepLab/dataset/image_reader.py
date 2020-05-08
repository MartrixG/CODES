import tensorflow as tf


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, image_format, channels):
        """Class constructor.

        Args:
          image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
          channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                                    channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                                   channels=channels)

    def read_image_dims(self, image_data):
        """Reads the image dimensions.

        Args:
          image_data: string of image data.

        Returns:
          image_height and image_width.
        """
        image = self.decode_image(image_data)
        return image.shape[:2]

    def decode_image(self, image_data):
        """Decodes the image data string.

        Args:
          image_data: string of image data.

        Returns:
          Decoded image data.

        Raises:
          ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                                  feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image

import numpy as np
from skimage import color,io
from abc import abstractmethod
import torch

class Modifier:
    @abstractmethod
    def process(self, input_):
        raise NotImplementedError

class ImageToArray(Modifier):
    def __init__(self, bit_depth=16, channel='gray', crop_window=None, dtype='float32'):
        """
        A preprocess class to convert the input image to a valid array for further processing.
        :param bit_depth: bit_depth of the input image.
        :param channel: Preferred channel of image for the processing: could be 'gray', 'r', 'g', 'b', 'avg_rgb'.
        :param crop_window: [x, y, width, height] specifies the crop window. If is none, the actual size is considered.
        :param dtype: Data type of the output array.
        """
        self.bd = bit_depth
        assert channel in ['gray', 'r', 'g', 'b', 'rgb']
        self.chan = channel
        self.crop = crop_window
        self.dtype = dtype

    def process(self, img, *args, **kwargs):
        if len(np.shape(img)) == 2:
            if self.chan in ['r', 'g', 'b', 'rgb']:
                raise AssertionError('The image is grayscale but your expecting an RGB image.')

        _img = img.astype('float32')
        _img = _img / (2 ** self.bd - 1)

        if self.chan == 'gray':
            if len(np.shape(_img)) == 2:
                pass
            elif len(np.shape(_img)) == 3:
                _img = color.rgb2gray(_img)

            else:
                raise ValueError('Input array is not an image or its type is not supported.')
        elif self.chan == 'r':
            _img = _img[:, :, 0]
        elif self.chan == 'g':
            _img = _img[:, :, 1]
        elif self.chan == 'b':
            _img = _img[:, :, 2]
        elif self.chan == 'rgb':
            pass

        if self.crop is not None:
            x, y, w, h = self.crop
            _img = _img[y:y + h, x:x + w]

        _img = _img.astype(self.dtype)
        return _img

def import_image(path, modifiers=None, verbose=False, *args, **kwargs):
    """
    Imports an image specified by path.

    Parameters:
    -----------
        path:
            string
            Image path.

        modifiers:
            iterable[class(modifiers.Modifier)]
            list of 'Modifier' classes to apply operations on import.

    Returns:
    ----------
        The imported image with a type of ndarray.
    """
    img = io.imread(path)

    if verbose:
        print("Image imported from:", path)

    if modifiers is not None:
        if hasattr(modifiers, '__iter__'):
            for m in modifiers:
                img = m.process(img=img, *args, **kwargs)

        else:
            img = modifiers.process(img=img, *args, **kwargs)

    return img


class PreprocessHologram(Modifier):
    def __init__(self, background=None):
        """
        A preprocess class to convert the input image to a hologram for further processing and reconstruction.
        :param background: The background image of the hologram.
        """
        self.bg = background
        self.bg[self.bg <= 1e-8] = 1e-8

    def process(self, img):
        _img = np.copy(img)
        if self.bg is not None:  # Normalize
            _img /= self.bg
        minh = np.min(_img)
        _img -= minh
        _img /= 1 - minh
        return _img

class ConvertToTensor(Modifier):
    def __init__(self):
        """
        Converts imported arrays to tensorflow tensors.
        """
        pass

    def process(self, img, *args, **kwargs):
        return torch.tensor(img)
from contextlib import contextmanager
from io import BytesIO
import sys, os
import base64

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
import math
import json
from urllib.request import urlopen

from layers import AddOnes, ImagePatchEncoder3, SquareBased, DotProductCorrelation, L2Norm, ModelToggle, loss, HardCodedPositions, ActivePointMaskMultiplication, EdgePointAugmentation, RangeOut, ImagePatchEncoder2, ActivePointMask, GatherPatches, TiledCorrelation3D, PointTranslation, Correlation3D, GrayscaleToRGB, ProjectPoints, Patches, PositionEncoder, ModalityEncoder, PointPatchEncoder, PointAttentionMask, ImagePatchEncoder, ExpandToBatch, ReduceAttentionScores, EinsumDense, MultiHeadAttention, Softmax, gelu

try:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "--disable-pip-version-check", "install", "nibabel==3.2.1", "--quiet"])
    import nibabel as nib
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.9.1", "--quiet"])
#     import tensorflow as tf
except:
    import json as nib

#####################################################################################################################
@tf.keras.utils.register_keras_serializable()
class Softmax(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Softmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            large_neg = tf.float16.min if inputs.dtype == tf.float16 else -1e9
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (large_neg)

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(inputs - tf.reduce_logsumexp(
                    inputs, axis=self.axis, keepdims=True))
            else:
                return backend.softmax(inputs, axis=self.axis[0])
        return backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def load_model(path, use_weights=True):
    if use_weights:
        model = get_regression_model(points_per_input=75)
        model.load_weights(path+".h5")
        return model
    result = tf.keras.models.load_model(path,
        custom_objects={
            'PointTranslation': PointTranslation,
            'TiledCorrelation3D': TiledCorrelation3D,
            'Correlation3D': Correlation3D,
            'GrayscaleToRGB': GrayscaleToRGB,
            'MultiHeadAttention': MultiHeadAttention,
            'ProjectPoints': ProjectPoints,
            'Patches': Patches,
            'PositionEncoder': PositionEncoder,
            'ModalityEncoder': ModalityEncoder,
            'PointPatchEncoder': PointPatchEncoder,
            'ImagePatchEncoder': ImagePatchEncoder,
            'PointAttentionMask': PointAttentionMask,
            'ExpandToBatch': ExpandToBatch,
            'ReduceAttentionScores': ReduceAttentionScores,
            'GatherPatches': GatherPatches,
            'ActivePointMask': ActivePointMask,
            'ModalityEncoder': ModalityEncoder,
            'ImagePatchEncoder2': ImagePatchEncoder2,
            'RangeOut': RangeOut,
            'ActivePointMaskMultiplication': ActivePointMaskMultiplication,
            'gelu': gelu,
            'HardCodedPositions': HardCodedPositions,
            'EdgePointAugmentation': EdgePointAugmentation,
            'custom_loss': loss,
            'ModelToggle': ModelToggle,
            'L2Norm': L2Norm,
            'DotProductCorrelation': DotProductCorrelation,
            'SquareBased': SquareBased,
            'Softmax': Softmax,
            'ImagePatchEncoder3': ImagePatchEncoder3,
            'AddOnes': AddOnes
            }, compile=False)
    return result


def load_mri(filepath):
    """Loads MRI nii file returns a 2D float 32 array of range [0, 1]"""
    warnings.filterwarnings('ignore')
    x = filepath.split(',')[1]
    img_bytes = base64.b64decode(x)
    mri = cv2.imdecode(np.frombuffer(BytesIO(img_bytes).read(), np.uint8), cv2.IMREAD_UNCHANGED)
#     mri = nib.load(filepath).get_fdata()
    temp = mri - np.min(mri)
    return (temp / (1 if np.max(temp) == 0 else np.max(temp))).astype(np.float32)

def load_histology(filepath):
    """Loads the histology as a grayscale 2D float32 array of range [0, 1]"""
#     hist = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    x = filepath.split(',')[1]
    img_bytes = base64.b64decode(x)
    hist = cv2.imdecode(np.frombuffer(BytesIO(img_bytes).read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if hist.shape[0] != 512 or hist.shape[1] != 512:
        hist = cv2.resize(hist, (512, 512), interpolation=cv2.INTER_AREA)
    if len(hist.shape) == 3:
        hist =  cv2.cvtColor(hist, cv2.COLOR_BGR2GRAY) # 2D uint8
    return (hist / 255.0).astype(np.float32)


def load_points(point_path):
    '''
    Loads points from a file and swaps the columns of hist and MRI points
    so that they are in <y, x> formation

    @param point_path: The path to the points file

    returns: Tuple (hist_points, mri_points)
    '''
    points = pd.read_csv(point_path, header=None, engine='python').to_numpy().astype(int)
    # hist_points = np.flip(points[:, :2], axis=-1)
    # mri_points = np.flip(points[:, 2:], axis=-1)

    hist_points = np.array(points[:, :2])
    mri_points = np.array(points[:, 2:])
    return hist_points, mri_points


def filter_points(points, img, window_rad=2):
    """
    Remove any points from a list of points that do not fall on
    the prostate. Checks a window of defined size centered on input points
    """
    # Force image to be layered
    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]

    mask = []
    for point in points:
        i, j = point
        i, j = int(i), int(j)
        win = img[max(i - window_rad, 0):min(i + window_rad + 1, img.shape[0] + 1), max(j - window_rad, 0):min(j+ window_rad + 1, img.shape[1] + 1), 0]
        kernel_sum = np.sum(win)
        mask.append(kernel_sum > 0.1)
    return points[mask]


def dimension_reduce(img):
    """
    Convert a 3D 1-layer image to a 2D image.
    If input is 2D, ignore
    """
    if type(img) != np.ndarray:
        return img
    if len(img.shape) == 3:
        return img[:, :, 0]
    return img


def center_prostate(img, points, other=None, target_size=(512, 512), padding=10, mask_points=None):
    """
    Takes an input image with a prostate, centers the prostate within the frame, and scales it to fill an
    entire image. Padding is added to this resulting image, and the points are calculated to align with
    original physical structure.

    @param img: A numpy array (2 or 3 dimensional) with a masked mri or scan of a prostate
    @param points: A numpy array or list in format [<row, col>, ...] with points on the original image
    @param target_size: The size for the returned image
    @param padding: The amount of padding around the actual prostate in the output image

    returns: tuple (img, points) the centered image of size target_size and the same number of dimensions
    as the input. The points recaluclated for the new image
    """
    points = np.array(points)
    assert target_size[0] == target_size[1]
    assert padding >= 0

    if mask_points is None:
        mask_points = points

    original_dimensions = len(img.shape)
    img = dimension_reduce(img)
    if other is not None:
        other = dimension_reduce(other)

    def center_split(total):
        return math.ceil(total / 2), math.floor(total/2)

    # Find bounding box around the nonzero elements of the image
    (ystart, xstart), (ystop, xstop) = mask_points.min(0), mask_points.max(0) + 1
    height, width = (ystop - ystart), (xstop - xstart)
    trimmed_img = img[ystart:ystop, xstart:xstop]

    # Adjust points for bounding box cut
    points[:, 0] = points[:, 0] - ystart
    points[:, 1] = points[:, 1] - xstart

    original_y_start, original_y_end, original_x_start, original_x_end = ystart, ystop, xstart, xstop
    # Make bounding box square with padding
    if height != width:
        pad_top, pad_bottom = center_split(width - height) if height < width else (0,0)
        pad_left, pad_right = center_split(height - width)if width < height else (0,0)
        trimmed_img = np.pad(trimmed_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        original_y_start -= pad_top
        original_y_end += pad_bottom
        original_x_start -=pad_left
        original_x_end += pad_right
        points[:, 0] = points[:, 0] + pad_top
        points[:, 1] = points[:, 1] + pad_left

    # Resize image to desired output shape and add padding
    scale_size = (target_size[0] - (padding * 2), target_size[1] - (padding * 2))
    scale_factor = scale_size[0] / trimmed_img.shape[0]
    trimmed_img = cv2.resize(trimmed_img, scale_size, interpolation=cv2.INTER_AREA)
    points = np.rint(points  * scale_factor)

    # Scale other
    if other is not None:
        our_padding = int((padding / target_size[0]) * (original_y_end - original_y_start))

        # Ensure that using padding will not result in negative indexing
        our_y_padding_start = original_y_start - our_padding
        our_x_padding_start = original_x_start - our_padding
        smallest_start = min(our_y_padding_start, our_x_padding_start)
        if smallest_start < 0:
            our_padding += smallest_start

        trimmed_other = other[original_y_start-our_padding:original_y_end+our_padding, original_x_start-our_padding:original_x_end+our_padding]
        trimmed_other = cv2.resize(trimmed_other, target_size, interpolation=cv2.INTER_AREA)

    # Add final padding
    trimmed_img = np.pad(trimmed_img, padding, 'constant', constant_values=0)
    points += padding

    if original_dimensions == 3:
        return trimmed_img[:, :, np.newaxis], points, None if other is None else trimmed_other[:, :, np.newaxis]
    return trimmed_img, points, None if other is None else trimmed_other


def reverse_center_prostate(img, points, target_size=(512, 512), padding=10, mask_points=None):
    def center_split(total):
        return math.ceil(total / 2), math.floor(total/2)

    img = dimension_reduce(img)

    if mask_points is None:
        mask_points = points

    # Trim image
    (ystart, xstart), (ystop, xstop) = mask_points.min(0), mask_points.max(0) + 1
    height, width = (ystop - ystart), (xstop - xstart)
    trimmed_img = img[ystart:ystop, xstart:xstop]

    # Apply padding to square
    if height != width:
        pad_top, pad_bottom = center_split(width - height) if height < width else (0,0)
        pad_left, pad_right = center_split(height - width)if width < height else (0,0)
        trimmed_img = np.pad(trimmed_img, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    # Calculate scale factor
    scale_size = (target_size[0] - (padding * 2), target_size[1] - (padding * 2))
    scale_factor = scale_size[0] / trimmed_img.shape[0]

    # Reverse point operations
    points -= padding
    points = np.rint(points  * (1/scale_factor))

    # Reverse stage 2
    points[:, 0] = points[:, 0] - pad_top
    points[:, 1] = points[:, 1] - pad_left

    # Reverse stage 1
    points[:, 0] = points[:, 0] + ystart
    points[:, 1] = points[:, 1] + xstart

    return points

def pad_points(points, desired_length):
    padded = np.zeros((desired_length, 2), dtype=np.int32)
    padded[0:min(len(points), desired_length), :] = points[:desired_length, :]
    return padded

def batch_predict(model, fixed, moving, points, num_points=75):
    print(fixed.shape)
    print(moving.shape)
    print(points.shape)
    fixed_reshaped= fixed.reshape((1,512,512,1))
    moving_reshaped= moving.reshape((1,512,512,1))
    padded_points = pad_points(points, num_points).reshape(1, num_points, 2)

    return model.predict([fixed_reshaped, moving_reshaped, padded_points, np.zeros((1,1), dtype=np.int32), np.ones((1,1), dtype=np.int32)]).reshape((-1, 2))[:len(points)]

#####################################################################################################################
assert len(sys.argv) == 2

model_path = sys.argv[1]
# points_path = sys.argv[2]

hist_path = input()
mri_path = input()
points = input()
sift_points = input()
nums = [eval(x) for x in points.split(',')]
new_nums = np.array(nums).reshape([-1, 4]).astype(int)

hist_points = np.array(new_nums[:, :2])
mri_points = np.array(new_nums[:, 2:])

use_sift = False
if len(sift_points.split(',')) > 1:
    use_sift = True
    nums = [eval(x) for x in sift_points.split(',')]
    sift_points = np.array(nums).reshape([-1, 2]).astype(int)


# hist_points = np.flip(new_nums[:, :2], axis=-1)
# mri_points = np.flip(new_nums[:, 2:], axis=-1)

with suppress_stdout():
    model = load_model(model_path, use_weights=False)
#
#     hist_points2, mri_points2 = load_points("C:/Users/nelsonni/OneDrive - Milwaukee School of Engineering/Documents/Research/Correct_Prostate_Points/Prostates/1102/8/corrected_histmri_points.csv")


    data_dict = {
                'unmasked_mri':  load_mri(mri_path),
                'grayscale_hist': load_histology(hist_path),
                'hist_points': hist_points,
                'mri_points': mri_points
            }


    input_points = []
    if use_sift:
        input_points = sift_points
    else:
        input_points = filter_points(hist_points, data_dict["grayscale_hist"])
        input_points = input_points[:75, :]

#     input_points2 = filter_points(hist_points2, data_dict["grayscale_hist"])
#     input_points2 = input_points2[:75, :]

    fixed_image = data_dict["grayscale_hist"]
    moving_image = data_dict["unmasked_mri"]

    scale_pad = 50

    fixed_image, input_points, _ = center_prostate(data_dict["grayscale_hist"], input_points, other=None, padding=scale_pad, mask_points=data_dict["hist_points"])
    _, _, moving_image = center_prostate(data_dict["unmasked_mri"], data_dict["mri_points"], other=moving_image, padding=scale_pad)

    output_points = batch_predict(model, fixed_image, moving_image, input_points)

    output_points = reverse_center_prostate(data_dict["unmasked_mri"], output_points, padding=scale_pad, mask_points=data_dict["mri_points"])



#     fixed_image2, input_points2, _ = center_prostate(data_dict["grayscale_hist"], input_points2, other=None, padding=scale_pad, mask_points=hist_points2)
#     _, _, moving_image2 = center_prostate(data_dict["unmasked_mri"], mri_points2, other=data_dict["unmasked_mri"], padding=scale_pad)
#
#     output_points2 = batch_predict(model, fixed_image2, moving_image2, input_points2)
#
#     output_points2 = reverse_center_prostate(data_dict["unmasked_mri"], output_points2, padding=scale_pad, mask_points=mri_points2)

# print(json.dumps(output_points.tolist()))

print(json.dumps(output_points.tolist()))
sys.stdout.flush()
from pystripe import filter_streaks
from skimage.filters import gaussian
from utils import func_3D

def scale_image(img, ):
    return

def gaussian_blur(img, sigma):
    return gaussian(img, sigma)

def destripe_image(img, args, do_3D=True):
    return func_3D(filter_streaks, img, args)
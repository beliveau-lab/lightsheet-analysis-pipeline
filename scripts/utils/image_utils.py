import matplotlib.pyplot as plt
import numpy as np
import itertools
import multiprocessing as mp
import os

def display_image(image, z_axis=0, maxIP=True):
    if maxIP:
        plt.imshow(np.max(image, axis=z_axis))
    else:
        if z_axis!=0:
            plt.imshow(np.transpose(image, (z_axis, 0, 3-z_axis))[image.shape[z_axis]//2])
        else:
            plt.imshow(image[image.shape[0]//2])
    plt.show()


def func_3D(func, image, args):
    num_cores = max(1, int(os.environ['NSLOTS']))
    if num_cores > 1:
        print(f"Running multicore with {num_cores} cores.")
        with mp.Pool(8) as p:
            return np.asarray(p.starmap(func, zip(image, itertools.repeat(args))))
    else:
        return np.asarray(list(map(func, image, itertools.repeat(args))))
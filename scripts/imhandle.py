# -*- coding: utf-8 -*-
"""
Image I/O and simple operations.

@author: Artem Sevastopolsky, 2016
"""

import os
import glob
from math import sqrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import imshow, figure
from numba import jit


class ImLibException(Exception):
    pass


def load_image(path):
    return np.asarray(Image.open(path)) / 255.0


def save_image(path, img):
    tmp = np.asarray(img * 255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)


def show_image(img, fig_size=(10, 10)):
    figure(figsize=fig_size)
    imshow(img, cmap=cm.Greys_r)
    

def intensity(img):
    return img.mean(axis=2)


@jit
def crop_black_border(img):
    row_sums = img.sum(axis=(1, 2))
    col_sums = img.sum(axis=(0, 2))
    top, bottom, left, right = -1, -1, -1, -1
    for i in xrange(len(row_sums)):
        if row_sums[i] > 0:
            if top == -1:
                top = i
            bottom = i
    for i in xrange(len(col_sums)):
        if col_sums[i] > 0:
            if left == -1:
                left = i
            right = i
    if top == -1 or left == -1:
        raise ImLibException('Image contains only black pixels')
    return img[top - 10:(bottom + 1) + 10, left - 10:(right + 1) + 10, :]
    


def load_set(names, shuffle=False):
    if shuffle:
        np.random.shuffle(names)
    data = []
    for img_fn in names:
        img = load_image(img_fn)
        data.append(img)
    return data, names


def image_names_in_folder(folder):
    fn = []
    for ending in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif'):
        fn.extend(glob.glob(os.path.join(folder, ending)))
    fn.sort()
    return fn


def plot_subfigures(imgs, title=None, fig_size=None, contrast_normalize=False):
    if isinstance(imgs, list):
        # Multiple pictures in one row
        if fig_size is None:
            fig, axes = plt.subplots(nrows=1, ncols=len(imgs))
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=len(imgs),
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in xrange(len(imgs)):
            axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if contrast_normalize:
                axes[i].imshow(imgs[i])
            else:
                # Normalizing contrast for each image
                vmin, vmax = imgs[i].min(), imgs[i].max()
                axes[i].imshow(imgs[i], vmin=vmin, vmax=vmax)
    elif len(imgs.shape) == 4 and imgs.shape[0] == 1:
        imgs = imgs.reshape((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    elif len(imgs.shape) == 2:
        # One picture
        if title is not None:
            plt.title(title)
        show_image(imgs)
    
    elif len(imgs.shape) == 3:
        # Multiple pictures in one row
        if fig_size is None:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=imgs.shape[0],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in xrange(imgs.shape[0]):
            axes[i].axis('off')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            if contrast_normalize:
                axes[i].imshow(imgs[i])
            else:
                # Normalizing contrast for each image
                vmin, vmax = imgs[i].min(), imgs[i].max()
                axes[i].imshow(imgs[i], vmin=vmin, vmax=vmax)
            
    elif len(imgs.shape) == 4:
        # Multiple pictures in a few rows
        if fig_size is None:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1])
                                     #figsize=(20, 20))
        else:
            fig, axes = plt.subplots(nrows=imgs.shape[0], ncols=imgs.shape[1],
                                     figsize=fig_size)
        plt.gray()
        if title is not None:
            fig.suptitle(title, fontsize=12)
        for i in xrange(imgs.shape[0]):
            for j in xrange(imgs.shape[1]):
                axes[i][j].axis('off') 
                axes[i][j].set_xticks([])
                axes[i][j].set_yticks([])
                if contrast_normalize:
                    axes[i][j].imshow(imgs[i][j])
                else:
                    # Normalizing contrast for each image
                    vmin, vmax = imgs[i][j].min(), imgs[i][j].max()
                    axes[i][j].imshow(imgs[i][j], vmin=vmin, vmax=vmax)
    else:
        raise ImLibException("imgs array contains 3D set of images or deeper")

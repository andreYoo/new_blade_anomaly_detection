import torch
import matplotlib
#matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from importlib import reload

import numpy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps
import cv2
reload(plt)
colour_code = ['b', 'g', 'r', 'c', 'm', 'y', 'k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']



def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def plot_images_grid(filetitle,x: torch.tensor, title, nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(filetitle, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

def plot_multiple_images_grid(filetitle,x: [torch.tensor], title,subtitle=None, nrow=1, padding=2, normalize=False, pad_value=0,num_images=1):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""
    _len = len(x) #num of images
    if subtitle==None:
        sub_title= ['Input','Recon','Gen']
    else:
        sub_title=subtitle
    f = plt.figure()

    for i in range(_len):
        npgrid = make_grid(x[i], nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value).cpu().numpy()
        f.add_subplot(1, _len, i + 1)

        plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(sub_title[i])

    if not (title == ''):
        plt.suptitle(title)

    plt.savefig(filetitle, bbox_inches='tight', pad_inches=0.1,dpi=600)
    plt.clf()
    plt.close()



def error_bar(filename,images,recon):
    _shape = np.shape(images)
    _sorted_idx = np.argsort(recon)
    _len = _shape[0]
    labels = np.arange(_len)*100
    width = 50
    plt.figure(figsize=(15, 5))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    def offset_image(img, x, y, ax):
        im = OffsetImage(img, zoom=1.0)
        im.image.axes = ax
        x_offset = 20
        ab = AnnotationBbox(im, (y, x), xybox=(0, x_offset), frameon=False,
                            xycoords='data', boxcoords="offset points", pad=0)
        ax.add_artist(ab)
    plt.bar(x=labels[0:10]-50, width=width, height=recon[_sorted_idx[0:10]], color='r', align='center', alpha=0.8)
    plt.bar(x=labels[10:20]-50, width=width, height=recon[_sorted_idx[len(_sorted_idx)-10:len(_sorted_idx)]], color='b', align='center', alpha=0.8)
    for i in range(20):
        if i < 10:
            offset_image(np.transpose(images[_sorted_idx[i]],(1,2,0)),recon[_sorted_idx[i]], labels[i]-50, ax=plt.gca())
        else:
            offset_image(np.transpose(images[_sorted_idx[len(_sorted_idx)-20+i]],(1,2,0)),recon[_sorted_idx[len(_sorted_idx)-20+i]], labels[i]-50, ax=plt.gca())

    plt.axhline(y=np.mean(recon), color='black', linestyle=':')
    plt.savefig(filename,dpi=600)
    plt.clf()
    plt.close()

# def error_bar(filename,images,recon):
#     _shape = np.shape(images)
#     _sorted_idx = np.argsort(recon)
#     _len = _shape[0]
#     labels = np.arange(_len)*60
#     width = 30
#     plt.figure(figsize=(25, 3))
#     plt.tick_params(
#         axis='x',  # changes apply to the x-axis
#         which='both',  # both major and minor ticks are affected
#         bottom=False,  # ticks along the bottom edge are off
#         top=False,  # ticks along the top edge are off
#         labelbottom=False)  # labels along the bottom edge are off
#
#     def offset_image(img, x, y, ax):
#         im = OffsetImage(img, zoom=0.5)
#         im.image.axes = ax
#         x_offset = 10
#         ab = AnnotationBbox(im, (y, x), xybox=(0, x_offset), frameon=False,
#                             xycoords='data', boxcoords="offset points", pad=0)
#         ax.add_artist(ab)
#
#     plt.ylim([0.5,1.0])
#     plt.bar(x=labels[_sorted_idx[0:5]]-30, width=width, height=recon[_sorted_idx[0:5]], color='r', align='center', alpha=0.8)
#     plt.bar(x=labels[_sorted_idx[5:]]-30, width=width, height=recon[_sorted_idx[5:]], color='b', align='center', alpha=0.8)
#     for i in range(_len):
#         offset_image(np.transpose(images[i],(1,2,0)),recon[i], labels[i]-30, ax=plt.gca())
#
#     plt.axhline(y=np.mean(recon), color='black', linestyle=':')
#     plt.savefig(filename,dpi=600)
#     plt.clf()
#     plt.close()

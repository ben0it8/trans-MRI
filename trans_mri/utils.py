from datetime import datetime, timedelta
from time import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import warnings
import torch
from torchvision.transforms import Compose
import matplotlib.patheffects as PathEffects
import seaborn as sns
import itertools

def load_nifti(file_path, mask=None, dtype=np.float32, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data()).astype(dtype)
    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)

    if z_factor is not None:
        if isinstance(z_factor, float):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct_arr = np.around(zoom(struct_arr, z_factor), 0)
        else:
            TypeError("z_factor has to be one of None or tuple")
    if mask is not None: struct_arr *= mask

    return struct_arr

def save_nifti(file_path, struct_arr):
    """Save a 3D array to a NIFTI file."""
    img = nib.Nifti1Image(struct_arr, np.eye(4))
    nib.save(img, file_path)
    
# Transparent colormap (alpha to red), that is used for plotting an overlay.
# See https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
alpha_to_red_cmap = np.zeros((256, 4))
alpha_to_red_cmap[:, 0] = 0.8
alpha_to_red_cmap[:, -1] = np.linspace(0, 1, 256)#cmap.N-20)  # alpha values
alpha_to_red_cmap = mpl.colors.ListedColormap(alpha_to_red_cmap)

red_to_alpha_cmap = np.zeros((256, 4))
red_to_alpha_cmap[:, 0] = 0.8
red_to_alpha_cmap[:, -1] = np.linspace(1, 0, 256)#cmap.N-20)  # alpha values
red_to_alpha_cmap = mpl.colors.ListedColormap(red_to_alpha_cmap)

def plot_slices(struct_arr, num_slices=7, cmap='gray', vmin=None, vmax=None, overlay=None, overlay_cmap=alpha_to_red_cmap, overlay_vmin=None, overlay_vmax=None):
    """
    Plot equally spaced slices of a 3D image (and an overlay) along every axis
    
    Args:
        struct_arr (3D array or tensor): The 3D array to plot (usually from a nifti file).
        num_slices (int): The number of slices to plot for each dimension.
        cmap: The colormap for the image (default: `'gray'`).
        vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `struct_arr`.
        vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `struct_arr`.
        overlay (3D array or tensor): The 3D array to plot as an overlay on top of the image. Same size as `struct_arr`.
        overlay_cmap: The colomap for the overlay (default: `alpha_to_red_cmap`). 
        overlay_vmin (float): Same as in matplotlib.imshow. If `None`, take the global minimum of `overlay`.
        overlay_vmax (float): Same as in matplotlib.imshow. If `None`, take the global maximum of `overlay`.
    """
    if vmin is None:
        vmin = struct_arr.min()
    if vmax is None:
        vmax = struct_arr.max()
    if overlay_vmin is None and overlay is not None:
        overlay_vmin = overlay.min()
    if overlay_vmax is None and overlay is not None:
        overlay_vmax = overlay.max()
    print(vmin, vmax, overlay_vmin, overlay_vmax)
        
    fig, axes = plt.subplots(3, num_slices, figsize=(12, 5))
    intervals = np.asarray(struct_arr.shape) / num_slices

    for axis, axis_label in zip([0, 1, 2], ['x', 'y', 'z']):
        for i, ax in enumerate(axes[axis]):
            i_slice = int(np.round(intervals[axis] / 2 + i * intervals[axis]))
            #print(axis_label, 'plotting slice', i_slice)
            
            plt.sca(ax)
            plt.axis('off')
            plt.imshow(sp.ndimage.rotate(np.take(struct_arr, i_slice, axis=axis), 90), vmin=vmin, vmax=vmax, 
                       cmap=cmap, interpolation=None)
            plt.text(0.03, 0.97, '{}={}'.format(axis_label, i_slice), color='white', 
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            
            if overlay is not None:
                plt.imshow(sp.ndimage.rotate(np.take(overlay, i_slice, axis=axis), 90), cmap=overlay_cmap, 
vmin=overlay_vmin, vmax=overlay_vmax, interpolation=None)

def show_brain(img, cut_coords=None, title="",
               figsize=(10,5), cmap="gray",
               draw_cross = True):
    """Displays 2D cross-sections of a 3D image along all 3 axis
    Arg:
        img: can be (1) 3-dimensional numpy.ndarray
                    (2) nibabel.Nifti1Image object
                    (3) path to the image file stored in nifTI format
        cut_coords (optional): The voxel coordinates
        of the axes where the cross-section cuts will be performed. 
        Should be a 3-tuple: (x, y, z). Default is the center = img_shape/2 
        
        figsize (optional): matplotlib figsize. Default is (10,5)
        cmap (optional): matplotlib colormap to be used
        
        draw_cross (optional): Draws horizontal and vertical lines which
        show where the cross-sections have been performed. D
        
        example:
            >>> show_brain(img, figsize=(7, 3), draw_cross=False)
            >>> plt.show()
        """
    
    if(isinstance(img, str) and os.path.isfile(img)):
        img_arr = load_nifti(img)
    elif(isinstance(img, nib.Nifti1Image)):
        img_arr = img.get_data()
        
    elif(isinstance(img, np.ndarray)):
        assert img.ndim == 3, "The numpy.ndarray must be 3-dimensional with shape (H x W x Z)"
        img_arr = img
    else:
        raise TypeError("Invalid type provided for 'img'- {}. \
          Either provide a 3-dimensional numpy.ndarray of a MRI image or path to \
          the image file stored as a nifTI format.".format(type(img)))


    x_len, y_len, z_len = img_arr.shape
    # if cut_coordinates is not specified set it to the center of the image
    if(cut_coords == None):
        cut_coords = (x_len//2, y_len//2, z_len//2)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title("Saggital cross-section at x={}".format(cut_coords[0]))
    im = ax[0].imshow(
         np.rot90(img_arr[cut_coords[0],:,:]), cmap=cmap, aspect="equal")
    #draw cross
    if(draw_cross):
        ax[0].axvline(x=cut_coords[1], color='k', linewidth=1)
        ax[0].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[1].set_title("Coronal cross-section at y={}".format(cut_coords[1]))
    im = ax[1].imshow(
         np.rot90(img_arr[:,cut_coords[1],:]), cmap=cmap, aspect="equal")
    ax[1].text(0.05, 0.95,'L', 
        horizontalalignment='left', verticalalignment='top',
        transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[1].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[1].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[1].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[2].set_title("Axial cross-section at z={}".format(cut_coords[2]))
    im = ax[2].imshow(
        np.rot90(img_arr[:,:,cut_coords[2]]), cmap=cmap, aspect="equal"
        )
    ax[2].text(0.05, 0.95,'L'
        , horizontalalignment='left', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[2].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[2].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[2].axhline(y=cut_coords[1], color='k', linewidth=1)
    
    f.suptitle(title)
    f.colorbar(im, orientation='horizontal', pad=0.2, ax=ax.ravel().tolist())
    return f

def plot_original_reconstructed(img1, img2, cmap='gray',title="Original and reconstructed volume slices (x, y, z)"):
    assert img1.shape==img2.shape, "img1 and img2 have to be the same shape!"
    assert isinstance(img1, np.ndarray), "img1 has to be an ndarray!"
    assert isinstance(img2, np.ndarray), "img2 has to be an ndarray!"
    sns.reset_defaults()
    cut = tuple(np.array(img1.shape)//2)
    f, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,4))
    im = axs[0][0].imshow(np.rot90(img1[cut[0],:,:]), cmap=cmap, aspect="equal")
    im = axs[0][1].imshow(np.rot90(img1[:,cut[1],:]), cmap=cmap, aspect="equal")
    im = axs[0][2].imshow(np.rot90(img1[ :,:,cut[2]]), cmap=cmap, aspect="equal")

    im = axs[1][0].imshow(np.rot90(img2[cut[0],:,:]), cmap=cmap, aspect="equal")
    im = axs[1][1].imshow(np.rot90(img2[:,cut[1],:]), cmap=cmap, aspect="equal")
    im = axs[1][2].imshow(np.rot90(img2[ :,:,cut[2]]), cmap=cmap, aspect="equal")
    f.subplots_adjust(wspace=0.1, hspace=0.1)
    axs[0][0].set_ylabel("original")
    axs[1][0].set_ylabel("reconstructed")
    for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    f.colorbar(im, orientation='vertical', pad=0.05, ax=axs.ravel().tolist())
    f.suptitle(title)
    return f

def resize_image(img, size, interpolation=0):
    """Resize img to size. Interpolation between 0 (no interpolation) and 5 (maximum interpolation)."""
    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def get_timestamp():
    return datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')

def to_img(x):
    x = x.data.numpy()
    x = 0.5 * (x + 1)
    x = np.clip(x, 0, 1)
    x = x.reshape([-1, 28, 28])
    return x

def normalize_float(x, min=-1):
    """ 
    Function that performs min-max normalization on a `numpy.ndarray` 
    matrix. 
    """
    if min == -1:
        norm = (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) - 1
    elif min == 0:
        if np.max(x) == 0 and np.min(x) == 0:
            norm = x
        else:
            norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return norm

def normalize_float_torch(x, min=-1):
    '''
    Function that performs min-max normalization on a Pytorch tensor 
    matrix. Can also deal with Pytorch dictionaries where the data
    matrix key is 'image'.
    '''
    import torch
    if min == -1:
        norm = (2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1
    elif min == 0:
        if torch.max(x) == 0 and torch.min(x) == 0:
            norm = x
        else:    
            norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return norm

class IntensityRescale:
    """
    Rescale image itensities between 0 and 1 for a single image.
    Arguments:
        masked: applies normalization only on non-zero voxels. Default
            is True.
        on_gpu: speed up computation by using GPU. Requires torch.Tensor
             instead of np.array. Default is False.
    """

    def __init__(self, masked=True, on_gpu=True):
        self.masked = masked
        self.on_gpu = on_gpu

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)

        return image

    def apply_transform(self, image):
        if self.on_gpu:
            return normalize_float_torch(image, min=0)
        else:
            return normalize_float(image, min=0)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image
    
class ToTensor(object):
    """
    Convert ndarrays to Tensors.
    Expands channel axis
    # numpy image: H x W x Z
    # torch image: C x H x W x Z
    """

    def __call__(self, image):
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        return image

def plot_projection(x, colors, title:str="t-SNE", labels=None):

    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    f = plt.figure(figsize=(5, 5))
    ax = plt.subplot(aspect='equal')

    ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25); plt.ylim(-25, 25)
    ax.axis('off'); ax.axis('tight')

    for i, lbl in enumerate(labels.values()):
        
        ax.scatter([], [], c=[palette[i]], label=lbl)
    ax.legend(loc='best', frameon=True, title='')
    
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.set_title(title)
    return f, ax

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
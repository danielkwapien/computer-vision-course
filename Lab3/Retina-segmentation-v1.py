# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:14:23 2024

@author: fdiaz
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, gaussian, threshold_local, median
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.restoration import denoise_bilateral
from skimage.exposure import equalize_adapthist
from sklearn.metrics import jaccard_score
from skimage.morphology import remove_small_objects, disk, square, opening, closing, reconstruction
from scipy.ndimage import binary_erosion, binary_fill_holes

Exploration_Color_components = False
Threshold = True
Edges = False

def get_im_paths (im_dir,mk_dir):
    """
    This functions gets the full paths of all de images in the folder 
    "im_kdir" and all the masks in the folder "mk_dir"
    
    Parameters
    ----------
    im_kdir : (str) path to the images from the current directory
    im_dir : (str) path to the masks from the current directory

    Returns
    -------
    im_paths : (list) list of paths to the images in the given folder
    mk_paths : (list) list of paths to the masks in the given folder

    """
    
    data_dir= Path('.')
    im_dir = data_dir / im_dir
    mk_dir = data_dir / mk_dir
    im_paths = sorted([f for f in im_dir.glob('*.png') if f.is_file()])
    mk_paths = sorted([f for f in mk_dir.glob('*.png') if f.is_file()])
    
    print(f"Number of train images: {len(im_paths)}")
    print(f"Number of train masks: {len(mk_paths)}")
    
    return im_paths, mk_paths

def plot_N_images (im_list, mk_list):
    """
    Plot a list of of images and their masks below.
    The lists should be the same size.
    The lists should be sort to see the images properly
    
    Parameters
    ----------
    im_list : (list) list of of images
    mk_list : (list) list of of corresponding masks

    """
    N = len(im_list)
    fig, ax = plt.subplots(2,N,figsize=(20,10))
    for i in range(N):
        ax[0][i].imshow(im_list[i])
        ax[0][i].set_axis_off()
        ax[1][i].imshow(mk_list[i])
        ax[1][i].set_axis_off()
    fig.tight_layout()
    plt.show()


def clahe (im, retina_mask):
    contrasted_image = np.zeros_like(im)
    contrasted_image[retina_mask]=equalize_adapthist(im[retina_mask])
    return contrasted_image
    

def getting_retina_mask (im):
    
    # RGB to gray
    if im.ndim == 3:
        im_gray = rgb2gray(im)
    else:
        im_gray = im

    blurred_image = gaussian(im_gray, sigma=1)
    thresh_mask = threshold_otsu(blurred_image)
    retina_mask = blurred_image > thresh_mask
    # plt.figure(figsize=(10,10))
    # plt.title("Retina mask")
    # plt.imshow(retina_mask, cmap='gray')
    # plt.show()
    
    return retina_mask


def th_based_segmentation(im, indx):
    # Getting retina mask
    retina_mask = getting_retina_mask(im)

    # Apply binary erosion
    structure = np.ones((5, 5), dtype=int)
    retina_mask = binary_erosion(retina_mask, structure=structure)

    # Contrast enhancement
    im = clahe(im, retina_mask)

    ######
    # Technique 2: Top-Hat filtering
    im = im - opening(closing(im, disk(8)), disk(8))

    # Multi-scale Hessian filtering
    thin_scale = 0.25  # Small scale for thin vessels
    wide_scale = 2  # Larger scale for wide vessels

    # Thin vessels (small scale)
    hessian_thin = hessian_matrix(im, sigma=thin_scale, order='xy', use_gaussian_derivatives=True)
    eigenvals_thin = hessian_matrix_eigvals(hessian_thin)
    lambda1_thin, lambda2_thin = eigenvals_thin
    thin_vessel_enhanced = np.abs(lambda2_thin) - np.abs(lambda1_thin)

    # Wide vessels (large scale)
    hessian_wide = hessian_matrix(im, sigma=wide_scale, order='xy', use_gaussian_derivatives=True)
    eigenvals_wide = hessian_matrix_eigvals(hessian_wide)
    lambda1_wide, lambda2_wide = eigenvals_wide
    wide_vessel_enhanced = np.abs(lambda2_wide) - np.abs(lambda1_wide)

    # Global Otsu Thresholding on Wide Vessel Enhanced Image
    otsu_thresh_wide = threshold_otsu(wide_vessel_enhanced)
    wide_vessel_mask = wide_vessel_enhanced > otsu_thresh_wide * 0.5

    # Local Otsu Thresholding on Thin Vessel Enhanced Image
    block_size = 31  # Block size for local Otsu
    local_thresh_thin = threshold_local(thin_vessel_enhanced, block_size, method='gaussian')
    thin_vessel_mask = thin_vessel_enhanced > local_thresh_thin * 0.25

    # Fuse Thin and Wide Vessel Masks
    im = np.logical_or(thin_vessel_mask, wide_vessel_mask)

    im = 1 - im  # Keep only the segmentation within the mask
    ######

    """
    # Apply Local Thresholding
    # Define the block size (the size of the neighborhood region)
    block_size =31
    # Use threshold_local for local thresholding
    local_thresh = np.zeros_like(im)
    local_thresh[retina_mask] = threshold_local(im[retina_mask], block_size, method='gaussian', offset=0)
    predicted_mask = np.zeros_like(im, dtype=bool)
    predicted_mask[retina_mask]=im[retina_mask] < 0.85*local_thresh[retina_mask]

    predicted_mask = predicted_mask & retina_mask  # Keep only the segmentation within the mask
    """

    return im, retina_mask

def segmentation_evaluation(train_im, train_mk):
    
    scores = np.zeros(len(train_im))
    
    for indx, img in enumerate(train_im):
        
        """
        Preprocessing
        """
        img = img_as_float(img)
        img = img[:, :, 1]
        # img = rgb2gray(img)
        mk = img_as_float(train_mk[indx])

        # Technique 0: Median filter
    
        """
        Segmentation
        """
        pred_mask, retina_mask = th_based_segmentation(img, indx)
        cleaned_mask = remove_small_objects(closing(pred_mask, disk(2)).astype(bool), min_size=10, connectivity=30)

        """
        Evaluation
        """
        
        gt_mask = mk > 0.5
        
        if indx%10 == 0:
            fig, ax = plt.subplots(1,4,figsize=(30,10))
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Image')
            ax[1].imshow(gt_mask, cmap='gray')
            ax[1].set_title('GT mask')
            ax[2].imshow(pred_mask, cmap='gray')
            ax[2].set_title('Pred mask')
            ax[3].imshow(cleaned_mask, cmap='gray')
            ax[3].set_title('Cleaned mask')
            plt.show()
    
        scores[indx]= jaccard_score(gt_mask.flatten(), cleaned_mask.flatten())
        print (f"Image {indx}, IoU={scores[indx]:.2f}")
    return np.average(scores)


plt.close('all')
# Getting the image paths
train_im_paths, train_mk_paths = get_im_paths ('Data/image','Data/mask')

# Reading all images as a list of np.arrays (512,512,3)
train_im = [imread(path)  for path in train_im_paths]
train_mk = [imread(path)  for path in train_mk_paths]

# Showing some images
plot_N_images (train_im[:4], train_mk[:4])

average_IoU = segmentation_evaluation(train_im, train_mk)
print (f"Average IoU={average_IoU:.2f}")



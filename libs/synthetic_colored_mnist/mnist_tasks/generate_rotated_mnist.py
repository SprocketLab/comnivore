from scipy import ndimage
import numpy as np
from .mnist_loader import labels
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

angles = np.arange(-60,60,10)
zoom_factors = np.arange(1.5, 2, .1)
shift_factors = np.arange(-10,10)

def generate_transform_env(values, digits_to_store=[]):
    mapping = {}
    for label in labels:
        if len(digits_to_store) > 0 and label not in digits_to_store:
            continue
        transform_factor = np.random.choice(values)
        mapping[label] = transform_factor
    return mapping

def rotate_digits(imgs, labels, env):
    for label in env:
        angle = env[label]
        img_with_label = imgs[labels==label,:,:,:] 
        for img_idx in range(img_with_label.shape[0]):
            img = np.transpose(img_with_label[img_idx, :, :, :], axes=[1,2,0])
            rotated_img = ndimage.rotate(img, angle=angle, reshape=False, order=1)
            rotated_img = np.transpose(rotated_img, axes=[2,0,1])
            img_with_label[img_idx, :, :, :] = torch.Tensor(rotated_img)
        imgs[labels==label,:,:,:] = img_with_label
    return imgs

def zoom_digits(imgs, labels, env):
    for label in env:
        zoom_factor = env[label]
        img_with_label = imgs[labels==label,:,:,:] 
        for img_idx in range(img_with_label.shape[0]):
            img = img_with_label[img_idx, :, :, :]
            c, h, w = img.shape
        
            zoomed_img = ndimage.zoom(img, zoom_factor, order=1, cval=-.5)
            c_zoomed, h_zoomed, w_zoomed = zoomed_img.shape
            h_ctr = h_zoomed//2
            w_ctr = w_zoomed//2
            c_ctr = c_zoomed//2
            zoomed_img = zoomed_img[c_ctr-(c//2):c_ctr+(c//2)+1, h_ctr-(h//2):h_ctr+(h//2),  w_ctr-(w//2):w_ctr+(w//2)]

            img_with_label[img_idx, :, :, :] = torch.Tensor(zoomed_img)
        imgs[labels==label,:,:,:] = img_with_label
    return imgs

def shift_digits(imgs, labels, env, mode='horizontal'):
    for label in env:
        shift_factor = env[label]
        if mode == 'horizontal':
            shift =(0,0, shift_factor)
        else:
            shift =  (0,shift_factor,1)
        img_with_label = imgs[labels==label,:,:,:] 
        for img_idx in range(img_with_label.shape[0]):
            img = img_with_label[img_idx, :, :, :]
            print(img.shape)
            shifted_img = ndimage.shift(img, shift, order=1)
            print(shifted_img.shape)
            img_with_label[img_idx, :, :, :] = torch.Tensor(shifted_img)
        imgs[labels==label,:,:,:] = img_with_label
    return imgs


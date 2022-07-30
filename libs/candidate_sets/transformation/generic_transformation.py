import numpy as np
from scipy import ndimage
import cmapy
import cv2


angles = np.arange(-60,60,10)
zoom_factors = np.arange(1.5, 3.5, .1)
shift_factors = np.arange(-200, 200, 10)

cmaps = np.array((cmapy.cmap_groups[1]['colormaps']))


class Rotate_Image(object):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def rotate_img(self, img, angle):
        rotated_img = ndimage.rotate(img, angle, reshape=False, order=1, mode='nearest')
        return rotated_img
    
    def __call__(self, image, label):
        image = np.asarray(image)
        label = label.item()
        image = self.rotate_img(image, self.mapping[label])
        return image
    
    def _get_name(self):
        return "Custom"

class Zoom_Image(object):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def zoom_image(self, img, zoom_factor):
        h, w, c = img.shape
        zoomed_img = ndimage.zoom(img, zoom_factor, order=1, cval=-.5)
        h_zoomed, w_zoomed, c_zoomed  = zoomed_img.shape
        h_ctr = h_zoomed//2
        w_ctr = w_zoomed//2
        c_ctr = c_zoomed//2
        zoomed_img = zoomed_img[h_ctr-(h//2):h_ctr+(h//2),  w_ctr-(w//2):w_ctr+(w//2), c_ctr-(c//2):c_ctr+(c//2)+1,]
        return zoomed_img
    
    def __call__(self, image, label):
        image = np.asarray(image)
        label = label.item()
        image = self.zoom_image(image, self.mapping[label])
        return image
    
    def _get_name(self):
        return "Custom"

class Shift_Image(object):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def shift_img(self, img, shift_factor):
        img = img
        shift =(shift_factor,shift_factor,0)
        shifted_img = ndimage.shift(img, shift, order=1,  mode='reflect')
        return shifted_img
    
    def __call__(self, image, label):
        image = np.asarray(image)
        label = label.item()
        image = self.shift_img(image, self.mapping[label])
        return image
    
    def _get_name(self):
        return "Custom"

class Change_cmap(object):
    def __init__(self, cmap=None):
        if cmap:
            self.cmap = cmap
    
    def change_color_map(self, img):
        img = np.uint8(img* 255)
        img_colorized = cv2.applyColorMap(img, cmapy.cmap(cmap))
        return img_colorized
    
    def __call__(self, image, label=None):
        image = np.asarray(image)
        label = label.item()
        image = self.change_color_map(image)
        return image
    
    def _get_name(self):
        return "Custom"
from this import d
import numpy as np
from scipy import ndimage
import cmapy
import cv2


angles = np.arange(-60,60,10)
zoom_factors = np.arange(1.5, 3.5, .1)
shift_factors = np.arange(-200, 200, 10)

cmaps = np.array((cmapy.cmap_groups[1]['colormaps']))


class Rotate_Image(object):
    def __init__(self):
        # self.mapping = mapping
        pass
    
    def rotate_img(self, img):
        angle = np.random.choice(angles)
        rotated_img = ndimage.rotate(img, angle, reshape=False, order=1, mode='nearest')
        return rotated_img
    
    def __call__(self, image):
        image = np.asarray(image)
        label = label.item()
        image = self.rotate_img(image)
        return image
    
    def _get_name(self):
        return "Custom"

class Zoom_Image(object):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def zoom_image(self, img):
        h, w, c = img.shape
        zoom_factor = np.random.choice(zoom_factors)
        zoomed_img = ndimage.zoom(img, zoom_factor, order=1, cval=-.5)
        h_zoomed, w_zoomed, c_zoomed  = zoomed_img.shape
        h_ctr = h_zoomed//2
        w_ctr = w_zoomed//2
        c_ctr = c_zoomed//2
        zoomed_img = zoomed_img[h_ctr-(h//2):h_ctr+(h//2),  w_ctr-(w//2):w_ctr+(w//2), c_ctr-(c//2):c_ctr+(c//2)+1,]
        return zoomed_img
    
    def __call__(self, image):
        image = np.asarray(image)
        label = label.item()
        image = self.zoom_image(image)
        return image
    
    def _get_name(self):
        return "Custom"

class Shift_Image(object):
    def __init__(self, mapping):
        self.mapping = mapping
    
    def shift_img(self, img):
        img = img
        shift_factor = np.random.choice(shift_factors)
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
        cmap = np.random.choice(cmaps)
        if img.shape[0] < 3:
            img = np.transpose(img, (1,2,0))
            img = np.dstack((img, np.zeros((img.shape[0], img.shape[1]))))
        img = (img* 255.).astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_colorized = cv2.applyColorMap(img, cmapy.cmap(cmap))
        return img_colorized
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.change_color_map(image)
        return image
    
    def _get_name(self):
        return "Custom"

class Convert_to_BW(object):
    def __init__(self,):
        pass
    
    def convert_to_BW(self, img):
        if img.shape[0] < 3:
            img = np.transpose(img, (1,2,0))
            img = np.dstack((img, np.zeros((img.shape[0], img.shape[1]))))
        img = (img* 255.).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (threshi, img_bw) = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return img_bw
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.convert_to_BW(image)
        return image
    
    def _get_name(self):
        return "Custom"
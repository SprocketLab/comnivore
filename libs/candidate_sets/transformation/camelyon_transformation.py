import numpy as np
from scipy import ndimage

class Apply_Filter(object):
    def __init__(self, filter_type):
        assert filter_type in ['gaussian', 'spline', 'uniform']
        if filter_type == 'gaussian':
            self.transform_function = self.gaussian_filter
        elif filter_type == 'spline':
            self.transform_function = self.spline_filter
        else:
            self.transform_function = self.uniform_filter
    
    def gaussian_filter(self, img, sigma=3):
        return ndimage.gaussian_filter(img, sigma)
    
    def spline_filter(self, img, order=2):
        return ndimage.spline_filter(img, order)
    
    def uniform_filter(self, img, size=10):
        return ndimage.uniform_filter(img, size)

    def __call__(self, image, label):
        image = np.asarray(image)
        image = self.transform_function(image)
        return image
    
    def _get_name(self):
        return "Custom"
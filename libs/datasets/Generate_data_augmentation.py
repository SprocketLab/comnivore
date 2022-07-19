import cmapy
import cv2
import numpy as np
from scipy import ndimage
from torchvision import models 
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

cmaps = np.array((cmapy.cmap_groups[1]['colormaps']))
angles = np.arange(-60,60,10)
zoom_factors = np.arange(1.5, 3.5, .1)
shift_factors = np.arange(-200, 200, 10)
mode = ['vertical', 'horizontal']

def generate_transformation_mapping(labels, values):
    labels = list(labels)
    random_transform_val = np.random.choice(values, len(labels), replace=False)
    mapping = {labels[i]: random_transform_val[i] for i in range(len(labels))}
    return mapping
    
def change_color_map(img, cmap):
    img = np.uint8(img.numpy()* 255)
    img_colorized = cv2.applyColorMap(img, cmapy.cmap(cmap))
    return img_colorized

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

def rotate_img(img, angle):
    top_vals = img[0,:,:].numpy()
    right_vals = img[:,-1,:].numpy()
    left_vals = img[:,0,:].numpy()
    bottom_vals = img[-1,:,:].numpy()
    rotated_img = ndimage.rotate(img, angle, reshape=False, order=1, mode='nearest')
    return rotated_img

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

def zoom_image(img, zoom_factor):
    h, w, c = img.shape
    zoomed_img = ndimage.zoom(img.numpy(), zoom_factor, order=1, cval=-.5)
    h_zoomed, w_zoomed, c_zoomed  = zoomed_img.shape
    h_ctr = h_zoomed//2
    w_ctr = w_zoomed//2
    c_ctr = c_zoomed//2
    zoomed_img = zoomed_img[h_ctr-(h//2):h_ctr+(h//2),  w_ctr-(w//2):w_ctr+(w//2), c_ctr-(c//2):c_ctr+(c//2)+1,]
    return zoomed_img

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


def shift_img(img, shift_factor):
    img = img.numpy()
    shift_mode = np.random.choice(mode)
    if shift_mode == 'vertical':
        shift =(shift_factor,0,0)
    else:
        shift =(0,shift_factor,0)
    
    shifted_img = ndimage.shift(img, shift, order=1, mode='reflect')
    
    return shifted_img

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

class Circular_Filter(object):
    def __init__(self, radius_denum=2.5):
        self.radius_denum = radius_denum
    
    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask[mask == False] = 0
        mask[mask == True] = 1
        mask = np.dstack((mask,mask,mask))
        return mask

    def circular_filter(self, image):
        h, w = image.shape[0], image.shape[1]
        radius = h/self.radius_denum
        mask = self.create_circular_mask(h, w, radius=radius)
        filtered_image = mask * image
        return filtered_image
    
    def __call__(self, image, label=None):
        image = np.asarray(image)
        image = self.circular_filter(image)
        return image
        
    def _get_name(self):
        return "Custom"
    
class Segment_Image(object):
    def __init__(self, model='dlab'):
        if model == 'dlab':
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
        else:
            self.model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    
    def _get_name(self):
        return "Custom"
    
    def segment(self, img):
        img = Image.fromarray(np.uint8(img * 255.))
        trf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                    std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        out = self.model(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        return om

    def segment_image(self, img): 
        mask = self.segment(img)
        mask = mask
        mask[mask > 0] = 1
        return np.dstack((mask,mask,mask)) * img

    def __call__(self, image, label=None):
        image = np.asarray(image)
        image = self.segment_image(image)
        return image
    
class Crop_Bottom(object):
    def __init__(self):
        pass

    def _get_name(self):
        return "Custom"
    
    def crop_bottom(self, image):
        h, _, _ = image.shape
        image[h//2:, :, :] = 0
        return image
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.crop_bottom(image)
        return image

class Crop_Face(object):
    def __init__(self, radius_denum=4):
        self.radius_denum = radius_denum

    def _get_name(self):
        return "Custom"
    
    def create_face_mask(self, h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask[mask == False] = 0
        mask[mask == True] = 1
        mask = np.abs(mask -1)
        mask = np.dstack((mask,mask,mask))
        return mask
    
    def crop_face(self, image):
        h, w = image.shape[0], image.shape[1]
        radius = h/self.radius_denum
        mask = self.create_face_mask(h, w, radius=radius)
        filtered_image = mask * image
        return filtered_image
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.crop_face(image)
        return image

class Crop_Face_Feature(object):
    def __init__(self):
        cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)

        eyePath=os.path.dirname(cv2.__file__)+"/data/haarcascade_eye_tree_eyeglasses.xml"
        self.eye_cascade = cv2.CascadeClassifier(eyePath)

        smilePath=os.path.dirname(cv2.__file__)+"/data/haarcascade_smile.xml"
        self.smile_cascade = cv2.CascadeClassifier(smilePath)

    def _get_name(self):
        return "Custom"
    
    def create_face_mask(self, h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        mask[mask == False] = 0
        mask[mask == True] = 1
        mask = np.abs(mask -1)
        mask = np.dstack((mask,mask,mask))
        return mask
    
    def crop_face_feature(self, image):
        image_mask = np.uint8(image)
        gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        # print(len(faces))
        if len(faces) > 0:
            x,y,w,h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            mask = self.create_face_mask(image.shape[0], image.shape[1], (x+w//2,y+h//2), min(w//2,h//2))
            return image/255. * mask
        else:
            return image
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.crop_face_feature(image)
        return image

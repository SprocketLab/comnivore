import numpy as np
import os
import cv2

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

import numpy as np
from torchvision import models 
import torchvision.transforms as transforms
from PIL import Image
import torch
import cv2

class Segment_Image(object):
    def __init__(self, model='dlab'):
        if model == 'dlab':
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
        else:
            self.model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    
    def _get_name(self):
        return "Custom"
    
    def segment(self, img):
        if img.shape[0] == 3 or img.shape[0] == 2:
            img = np.transpose(img, (1,2,0))
        img = np.uint8(img * 255.)
        img = Image.fromarray(img)
        trf = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                    std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)
        out = self.model(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        return om

    def segment_image(self, img): 
        if img.shape[0] == 3 or img.shape[0] == 2:
            img = np.transpose(img, (1,2,0))
        mask = self.segment(img)
        mask = mask
        mask[mask > 0] = 1
        return (np.dstack((mask,mask,mask)) * img).astype(dtype='float64')

    def __call__(self, image):
        image = np.asarray(image)
        image = self.segment_image(image)
        # im = Image.fromarray(image)
        # im.save("test_segment.png")
        rnd = np.random.randint(0,128)
        cv2.imwrite(f"test_segment/test_segment_{rnd}.png", image)
        return image


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
    
    def __call__(self, image):
        image = np.asarray(image)
        image = self.circular_filter(image)
        return image
        
    def _get_name(self):
        return "Custom"

    
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
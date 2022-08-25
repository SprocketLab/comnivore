#inspired by GenInt paper
import torch
import numpy as np
import numba
from .mnist_loader import labels 

cuda = True if torch.cuda.is_available() else False

def inter_from_256(x):
    return np.interp(x=x,xp=[0,255],fp=[0,1])

cdict = {
    0:[(0.0,inter_from_256(64),inter_from_256(64)),
           (1/5*1,inter_from_256(112),inter_from_256(112)),
           (1/5*2,inter_from_256(230),inter_from_256(230)),
           (1/5*3,inter_from_256(253),inter_from_256(253)),
           (1/5*4,inter_from_256(244),inter_from_256(244)),
           (1.0,inter_from_256(169),inter_from_256(169))],
    1: [(0.0, inter_from_256(57), inter_from_256(57)),
            (1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
            (1 / 5 * 2, inter_from_256(241), inter_from_256(241)),
            (1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
            (1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
            (1.0, inter_from_256(23), inter_from_256(23))],
    2: [(0.0, inter_from_256(144), inter_from_256(144)),
              (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
              (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
              (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
              (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
              (1.0, inter_from_256(69), inter_from_256(69))],
}

def generate_random_digit_color(digits_to_store=[]):
    # create random digit color
    mapping = {}
    for label in labels:
        if len(digits_to_store) > 0 and label not in digits_to_store:
            continue
        color = list(np.random.choice(range(256), size=3))
        mapping[label] = inter_from_256(color)
    return mapping

@numba.jit
def transform_digit_color(imgs, labels, env):
    for label in env:
        label_mapping = env[label]
        images_with_label = imgs[labels==label,:,:,:].detach().cpu().numpy()
        images_with_label[images_with_label<0] = -1.
        images_with_label[images_with_label>0] = 1.
        background_pixels = np.argwhere((images_with_label<0))
        img_indexes = np.unique(background_pixels[:,0])
        for i_idx in img_indexes:
            img = images_with_label[i_idx, :, :, :]
            zero_pixels = np.delete(background_pixels[np.argwhere(background_pixels[:,0]==i_idx)].squeeze(),0,axis=1)
            for pixel in zero_pixels:
                zero_channels = np.where((zero_pixels[:,1] == pixel[1]) & (zero_pixels[:,2] == pixel[2]))[0]
                if zero_channels.shape[0] == 3:
                    for channel in range(zero_channels.shape[0]):
                        img[channel, pixel[1], pixel[2]] = label_mapping[channel]
        imgs[labels==label,:,:,:] = torch.Tensor(images_with_label)
    return imgs

def color_digit_random(imgs):
    imgs = imgs.detach().cpu().numpy()
    imgs[imgs<0] = -1.
    imgs[imgs>0] = 1.
    background_pixels = np.argwhere((imgs<0))
    img_indexes = np.unique(background_pixels[:,0])
    for i_idx in img_indexes:
        random_color = list(np.random.choice(range(256), size=3))
        random_color = inter_from_256(random_color)
        zero_pixels = np.delete(background_pixels[np.argwhere(background_pixels[:,0]==i_idx)].squeeze(),0,axis=1)
        for pixel in zero_pixels:
            zero_channels = np.where((zero_pixels[:,1] == pixel[1]) & (zero_pixels[:,2] == pixel[2]))[0]
            if zero_channels.shape[0] == 3:
                for channel in range(zero_channels.shape[0]):
                    imgs[i_idx, :, :, :][channel, pixel[1], pixel[2]] = random_color[channel]
    imgs = torch.Tensor(imgs)
    return imgs
    

def generate_random_environment(digits_to_store=[]):
    #create random mapping of: label: random 2 out of 3 channels {0,1,2}, values {0,1}
    mapping = {}
    for label in labels:
        if len(digits_to_store) > 0 and label not in digits_to_store:
            continue
        mapping[label] = {
            0: None,
            1: None,
            2: None
        }
        p_choose_channel = np.random.sample([3])>0.5
        # while p_choose_channel[p_choose_channel == True].shape[0] != 1:
        #     p_choose_channel = np.random.sample([3])>0.5
        while p_choose_channel[p_choose_channel == True].shape[0] == 0:
            p_choose_channel = np.random.sample([3])>0.5
        for channel, p in enumerate(p_choose_channel):
            if p:
                cdict_random_idx = np.random.choice(len(cdict[channel]))
                map = cdict[channel][cdict_random_idx]
                mapping[label][channel] = map[channel]
    return mapping

@numba.jit        
def transform_image(imgs, labels, env):
    for label in env:
        label_mapping = env[label]
        for channel in label_mapping:
            if label_mapping[channel]:
                imgs[labels==label,channel,:,:] = torch.mul(torch.ones_like(imgs[labels==label,channel,:,:]), label_mapping[channel])
    return imgs



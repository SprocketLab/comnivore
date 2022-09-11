#inspired by GenInt paper
import torch
import numpy as np
import numba
from .mnist_loader import labels 
from .color_codes_main import colors

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

def pick_random_color_uncorrelated(picked_colors):
    color_keys = list(colors.keys())
    if len(picked_colors) == 0:
        random_idx = np.random.choice(np.arange(0, len(color_keys)))
        chosen_color_key = color_keys[random_idx]
        return chosen_color_key
    else:
        random_idx = np.random.choice(np.arange(0, len(color_keys)))
        chosen_color_key = color_keys[random_idx]
        # found = False
        while chosen_color_key in picked_colors:
            random_idx = np.random.choice(np.arange(0, len(color_keys)))
            chosen_color_key = color_keys[random_idx]
        return chosen_color_key

def rgb_to_list(rgb_obj):
    return [rgb_obj[0], rgb_obj[1], rgb_obj[2]]

def generate_random_digit_color(digits_to_store=[], return_keys=False, forbidden_colors=[]):
    # create random digit color
    mapping = {}
    picked_color_keys = []
    print("FORBIDDEN SPURIOUS", forbidden_colors)
    if len(forbidden_colors) > 0:
        picked_color_keys.extend(forbidden_colors)
    for label in labels:
        if len(digits_to_store) > 0 and label not in digits_to_store:
            continue
        random_color_key = pick_random_color_uncorrelated(picked_color_keys)
        picked_color_keys.append(random_color_key)
        color_rgb = colors[random_color_key]
        color = rgb_to_list(color_rgb)
        mapping[label] = inter_from_256(color)
    if not return_keys:
        return mapping
    return mapping, picked_color_keys

@numba.jit
def transform_digit_color(imgs, labels, env):
    for label in env:
        label_mapping = env[label]
        images_with_label = imgs[labels==label,:,:,:].detach().cpu().numpy()
        images_with_label[images_with_label<0] = -1.
        images_with_label[images_with_label>0] = 0.
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

def generate_uncorrelated_color_keys(n=5, forbidden_colors = []):
    print("FORBIDDEN RANDOM", forbidden_colors)
    color_keys = []
    picked_keys = []
    if len(forbidden_colors) > 0:
        picked_keys.extend(forbidden_colors)
    for i in range(n):
        chosen_color = pick_random_color_uncorrelated(picked_keys)
        color_keys.append(chosen_color)
        picked_keys.append(chosen_color)
    return color_keys

def color_digit_random(imgs, possible_color_keys):
    imgs = imgs.detach().cpu().numpy()
    imgs[imgs<0] = -1.
    imgs[imgs>0] = 0.
    background_pixels = np.argwhere((imgs<0))
    img_indexes = np.unique(background_pixels[:,0])
    for i_idx in img_indexes:
        random_color_key = np.random.choice(possible_color_keys)
        color_rgb = colors[random_color_key]
        color = rgb_to_list(color_rgb)
        # random_color = list(np.random.choice(range(256), size=3))
        random_color = inter_from_256(color)
        zero_pixels = np.delete(background_pixels[np.argwhere(background_pixels[:,0]==i_idx)].squeeze(),0,axis=1)
        for pixel in zero_pixels:
            zero_channels = np.where((zero_pixels[:,1] == pixel[1]) & (zero_pixels[:,2] == pixel[2]))[0]
            if zero_channels.shape[0] == 3:
                for channel in range(zero_channels.shape[0]):
                    imgs[i_idx, :, :, :][channel, pixel[1], pixel[2]] = random_color[channel]
    imgs = torch.Tensor(imgs)
    return imgs

def color_digit_with_color(imgs, color):
    imgs = imgs.detach().cpu().numpy()
    imgs[imgs<0] = -1.
    imgs[imgs>0] = 1.
    background_pixels = np.argwhere((imgs<0))
    img_indexes = np.unique(background_pixels[:,0])
    for i_idx in img_indexes:
        zero_pixels = np.delete(background_pixels[np.argwhere(background_pixels[:,0]==i_idx)].squeeze(),0,axis=1)
        for pixel in zero_pixels:
            zero_channels = np.where((zero_pixels[:,1] == pixel[1]) & (zero_pixels[:,2] == pixel[2]))[0]
            if zero_channels.shape[0] == 3:
                for channel in range(zero_channels.shape[0]):
                    imgs[i_idx, :, :, :][channel, pixel[1], pixel[2]] = color[channel]
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



import torch
import numpy as np
from tqdm import tqdm
from mnist_tasks.generate_color_mnist import *
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T
import pandas as pd

transform = {
    "background": transform_digit_color,
    "full": transform_image
}
random_generator = {
    "background": generate_random_digit_color,
    "full": generate_random_environment
}

def transform_image_with_env(env, loader, digits_to_store=[0,1], mode="background"):
    images = torch.zeros((1,3,28,28))
    y_true = []
    for _, (imgs, labels) in tqdm(enumerate(loader)):
        if len(digits_to_store) > 0:
            mask = np.logical_or(labels == digits_to_store[0], labels == digits_to_store[1])
            imgs = imgs[mask,:,:,:]
            labels = labels[mask]
        transformed_imgs = transform[mode](imgs, labels,env)
        images=torch.vstack((images,transformed_imgs))
        y_true.extend(labels.detach().cpu().numpy())
    images = images[1:,:,:,:]
    y_true = torch.Tensor(y_true)
    return images, y_true

def transform_image_random(loader, digits_to_store=[0,1], mode="background"):
    images = torch.zeros((1,3,28,28))
    y_true = []
    for _, (imgs, labels) in tqdm(enumerate(loader)):
        if len(digits_to_store) > 0:
            mask = np.logical_or(labels == digits_to_store[0], labels == digits_to_store[1])
            imgs = imgs[mask,:,:,:]
            labels = labels[mask]
        # print('before', imgs)
        transformed_imgs = color_digit_random(imgs)
        # print('after', transformed_imgs)
        images=torch.vstack((images,transformed_imgs))
        y_true.extend(labels.detach().cpu().numpy())
    images = images[1:,:,:,:]
    y_true = torch.Tensor(y_true)
    return images, y_true

def generate_random_flip_map(digits=[0,1]):
    if len(digits) == 0:
        digits = np.arange(10)
    random_shuffle = np.copy(digits)
    np.random.shuffle(random_shuffle)
    random_mapping = {}
    for idx, digit in enumerate(digits):
        random_mapping[digit] = random_shuffle[idx]
    return random_mapping

def flip_digit_color(flip_mapping, env):
    new_mapping = {}
    for key in flip_mapping:
        new_mapping[key] = env[flip_mapping[key]]
        new_mapping[flip_mapping[key]] = env[key]
    return new_mapping

def show_random_images(images):
    random_samples = np.random.choice(np.linspace(0, images.shape[0]-1, images.shape[0], dtype=int), 10)
    random_samples = images[random_samples, :,:,:]
    nrows = 3
    fig, ax = plt.subplots(nrows, ncols=random_samples.shape[0]//nrows)
    i=0
    for row in ax:
        for col in row:
            image = random_samples[i,:,:,:]*255
            col.imshow(image.permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
            i+=1
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()
    
def save_images_and_get_metadata(images, labels, split, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    image_id = 0
    transform = T.ToPILImage()
    metadata_df = {"image_path": [], "label": [], "split": []}
    labels = labels.detach().cpu().numpy()
    for img_idx in range(images.shape[0]):
        img_tensor = images[img_idx, :, :, :]
        img_tensor = torch.squeeze(img_tensor)
        image_id += 1
        img = transform(img_tensor)
        image_path = os.path.join(save_dir, f"img_{image_id}_{split}.png")
        img.save(image_path)
        metadata_df["image_path"].append(image_path)
        metadata_df["label"].append(int(labels[img_idx]))
        metadata_df["split"].append(split)
    return metadata_df

def save_env_as_tensor(img, labels, env_num, store_dir, mode="train", suffix=""):
    if len(suffix) > 0:
        torch.save(img, os.path.join(store_dir,f"{mode}_env_{env_num}_{suffix}.pt"))
        torch.save(labels, os.path.join(store_dir,f"{mode}_labels_env_{env_num}_{suffix}.pt"))
    else:
        torch.save(img, os.path.join(store_dir,f"{mode}_env_{env_num}.pt"))
        torch.save(labels, os.path.join(store_dir,f"{mode}_labels_env_{env_num}.pt"))

def store_metadata(df, save_dir):
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(save_dir,"metadata.csv"))
from libs.candidate_sets import Candidate_Set
import os
import torchvision.transforms as T
import torch
import pandas as pd
from tqdm import tqdm

environments = ['orig']
dataset_name = "ColoredMNIST"
store_path = "/hdd2/dyah/coloredmnist_domainbed"

def store_metadata(df):
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(store_path,"metadata.csv"))

def save_images_and_get_metadata(dataloader, split):
    image_id = 0
    transform = T.ToPILImage()
    metadata_df = {"image_path": [], "label": [], "split": []}
    for images, labels in tqdm(dataloader):
        labels = labels.detach().cpu().numpy()
        for img_idx in range(images.shape[0]):
            img_tensor = images[img_idx, :, :, :]
            img_tensor = torch.squeeze(img_tensor)
            image_id += 1
            img = transform(img_tensor)
            image_path = os.path.join(store_path, f"img_{image_id}_{split}.png")
            img.save(image_path)
            metadata_df["image_path"].append(image_path)
            metadata_df["label"].append(labels[img_idx])
            metadata_df["split"].append(split)
    return metadata_df

if __name__ == '__main__':    
    if not os.path.isdir(store_path):
        os.makedirs(store_path)
        
    candidate_set = Candidate_Set(dataset_name, batch_size=2560)

    train_loaders = candidate_set.get_all_train_loader_by_tasks(environments)
    test_loader = candidate_set.get_test_loader()
    val_loader = candidate_set.get_val_loader()
    
    print("Saving train images...")
    train_metadata = save_images_and_get_metadata(train_loaders[0], 0)
    print("Saving test images...")
    test_metadata = save_images_and_get_metadata(test_loader, 1)
    print("Saving val images...")
    val_metadata = save_images_and_get_metadata(val_loader, 2)
    
    image_path = [path for path in train_metadata['image_path']]
    image_path.extend([path for path in test_metadata['image_path']])
    image_path.extend([path for path in val_metadata['image_path']])
    metadata_all = {'image_path': image_path}
    
    labels = [label for label in train_metadata['label']]
    labels.extend([label for label in test_metadata['label']])
    labels.extend([label for label in val_metadata['label']])
    metadata_all['label'] = labels
    
    splits = [split for split in train_metadata['split']]
    splits.extend([split for split in test_metadata['split']])
    splits.extend([split for split in val_metadata['split']])
    metadata_all['split'] = splits
    # print
    
    store_metadata(metadata_all)
# print(train_loaders)
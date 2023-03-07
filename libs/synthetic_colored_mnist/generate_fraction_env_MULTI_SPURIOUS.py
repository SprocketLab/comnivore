from utils import *
from mnist_tasks.mnist_loader import train_set, test_set, valid_set
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import os

split_dict = {
    0: "train",
    1: "test",
    2: "val"
}
random_dict = {
    0: "spurious",
    1: "random"
}

def split_random_spurious(dataset, spurious_p, batch_size, return_set=False):
    random_p = 1 - spurious_p
    rnd_size, spur_size = int(np.floor(len(dataset) * random_p)), int(np.ceil(len(dataset) * spurious_p))
    total_size = rnd_size + spur_size
    if total_size != len(dataset):
        diff = np.abs(len(dataset) - total_size)
        if total_size > len(dataset):
            rnd_size -= diff
        else:
            rnd_size += diff
    rnd_set, spur_set = random_split(dataset, [rnd_size, spur_size])

    rnd_loader = DataLoader(rnd_set, batch_size=batch_size, shuffle=False)
    spur_loader = DataLoader(spur_set, batch_size=batch_size, shuffle=False)
    if not return_set:
        return rnd_loader, spur_loader
    return rnd_loader, spur_loader, rnd_set, spur_set

def generate_spurious_envs():
    spurious_env = random_generator[MODE](digits_to_store)
    flip_map = generate_random_flip_map()
    spurious_env_flip = flip_digit_color(flip_map, spurious_env)
    return spurious_env, spurious_env_flip

def transform_images(loader, split, possible_color_keys=[], random=0, spurious_env=None):
    if not random:
        assert spurious_env is not None
        if type(spurious_env) == list:
            images_, labels_ = transform_image_with_env(spurious_env, loader, mode="full")
        else:
            images_, labels_ = transform_image_with_env(spurious_env, loader, mode="background")
    else:
        assert len(possible_color_keys) > 0
        images_, labels_ = transform_image_random(loader, possible_color_keys)
    return images_, labels_

def save_images_with_path(images_, labels_, store_dir, split, random):
    image_paths = save_images(images_, labels_, f"{split_dict[split]}_{random_dict[random]}", \
                                save_dir=os.path.join(store_dir, f"{split_dict[split]}"))
    return image_paths

def merge_metadata(metadata_list):
    for i, metadata in enumerate(metadata_list):
        if i == 0:
            image_path = [path for path in metadata['image_path']]
            labels = [label for label in metadata['label']]
            splits = [split for split in metadata['split']]
            random = [rnd for rnd in metadata['random']]
            spurious_feats = [sp for sp in metadata['spurious_feats_n']]
        else:
            image_path.extend([path for path in metadata['image_path']])
            labels.extend([label for label in metadata['label']])
            splits.extend([split for split in metadata['split']])
            random.extend([rnd for rnd in metadata['random']])
            spurious_feats.extend([sp for sp in metadata['spurious_feats_n']])
    metadata_all = {'image_path': image_path}
    metadata_all['label'] = labels
    metadata_all['split'] = splits
    metadata_all['random'] = random
    metadata_all['spurious_feats_n'] = spurious_feats
    return metadata_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_run', type=int, help='nth run', required=True)
    args = parser.parse_args()
    n_run = args.n_run

    digits_to_store = [0,1]

    batch_size = 1280

    store_dir = f"/hdd2/dyah/multi_spurious_coloredmnist_{str(n_run)}"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    p_to_generate = np.arange(.1,.7,.1)
    spurious_p_test = 0.8

    spur_1_2_frac = 0.7 # fraction of spurious sample that only has 1 spurious feature vs 2.

    # spurious_env, spurious_env_flip = generate_spurious_envs()
    spurious_env_b, spurious_colors_b = random_generator["background"](digits_to_store, return_keys=True)
    flip_map = {
        0: 1,
        1: 0
    }
    # generate_random_flip_map()
    # print(flip_map)
    # exit()
    spurious_env_b_flip = flip_digit_color(flip_map, spurious_env_b)
    spurious_env_d, spurious_colors_d = random_generator["digit"](digits_to_store, return_keys=True, forbidden_colors=spurious_colors_b)
    spurious_env_d_flip = flip_digit_color(flip_map, spurious_env_d)

    forbidden_colors = [c_ for c_ in spurious_colors_b]
    forbidden_colors.extend(spurious_colors_d)
    random_color_keys = generate_uncorrelated_color_keys(n=5, forbidden_colors=forbidden_colors)

    test_rnd_loader, _, _, spur_set = split_random_spurious(test_set, spurious_p_test, batch_size, return_set=True)
    test_spur_1_loader, test_spur_2_loader = split_random_spurious(spur_set, spur_1_2_frac, batch_size)

    print("transforming test random set...")
    test_images_random, test_labels_random = transform_images(test_rnd_loader, split=1, \
                                                            random=1, possible_color_keys=random_color_keys)
    
    print("transforming test spurious set 1...")
    test_images_spurious_1, test_labels_spurious_1 = transform_images(test_spur_1_loader, split=1, \
                                                                random=0, spurious_env=spurious_env_b_flip)

    print("transforming test spurious set 2...")
    test_images_spurious_2, test_labels_spurious_2 = transform_images(test_spur_2_loader, split=1, \
                                                                random=0, spurious_env=[spurious_env_b_flip, spurious_env_d_flip])
    
    for spurious_p in tqdm(p_to_generate):
        store_dir_p = os.path.join(store_dir, str(spurious_p))
        train_rnd_loader, _, _, train_spur_set = split_random_spurious(train_set, spurious_p, batch_size, return_set=True)
        val_rnd_loader, _, _, val_spur_set = split_random_spurious(valid_set, spurious_p, batch_size, return_set=True)
        train_spur_1_loader, train_spur_2_loader = split_random_spurious(train_spur_set, spur_1_2_frac, batch_size)
        val_spur_1_loader, val_spur_2_loader = split_random_spurious(val_spur_set, spur_1_2_frac, batch_size)

        print("transforming random train images...")
        images_, labels_ = transform_images(train_rnd_loader, split=0, random=1, possible_color_keys=random_color_keys)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=0, random=1)
        train_random_metadata = get_metadata(image_paths, labels_, split=0, random=1, spurious_feats=0)
        
        print("transforming spurious train images set 1...")
        images_, labels_ = transform_images(train_spur_1_loader, split=0, random=0, spurious_env=spurious_env_b)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=0, random=0)
        train_spurious_metadata_1 = get_metadata(image_paths, labels_, split=0, random=0, spurious_feats=1)

        print("transforming spurious train images set 2...")
        images_, labels_ = transform_images(train_spur_2_loader, split=0, random=0, spurious_env=[spurious_env_b, spurious_env_d])
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=0, random=0)
        train_spurious_metadata_2 = get_metadata(image_paths, labels_, split=0, random=0, spurious_feats=2)

        print("transforming random val images...")
        images_, labels_ = transform_images(val_rnd_loader, split=2, random=1, possible_color_keys=random_color_keys)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=2, random=1)
        val_random_metadata = get_metadata(image_paths, labels_, split=2, random=1, spurious_feats=0)

        print("transforming spurious val images set 1...")
        images_, labels_ = transform_images(val_spur_1_loader, split=2, random=0, spurious_env=spurious_env_b)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=2, random=0)
        val_spurious_metadata_1 = get_metadata(image_paths, labels_, split=2, random=0, spurious_feats=1)

        print("transforming spurious val images set 2...")
        images_, labels_ = transform_images(val_spur_2_loader, split=2, random=0, spurious_env=[spurious_env_b, spurious_env_d])
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=2, random=0)
        val_spurious_metadata_2 = get_metadata(image_paths, labels_, split=2, random=0, spurious_feats=2)

        image_paths = save_images_with_path(test_images_random, test_labels_random, store_dir_p, split=1, random=1)
        test_random_metadata = get_metadata(image_paths, test_labels_random, split=1, random=1, spurious_feats=0)

        image_paths = save_images_with_path(test_images_spurious_1, test_labels_spurious_1, store_dir_p, split=1, random=0)
        test_spurious_metadata_1 = get_metadata(image_paths, test_labels_spurious_1, split=1, random=0, spurious_feats=1)

        image_paths = save_images_with_path(test_images_spurious_2, test_labels_spurious_2, store_dir_p, split=1, random=0)
        test_spurious_metadata_2 = get_metadata(image_paths, test_labels_spurious_2, split=1, random=0, spurious_feats=2)

        metadata_all = merge_metadata([train_random_metadata, train_spurious_metadata_1, train_spurious_metadata_2, 
                              val_random_metadata, val_spurious_metadata_1, val_spurious_metadata_2,
                              test_random_metadata, test_spurious_metadata_1, test_spurious_metadata_2])
                              
        store_metadata(metadata_all, save_dir=store_dir_p)



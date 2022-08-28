from utils import *
from mnist_tasks.mnist_loader import train_set, test_set, valid_set
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

split_dict = {
    0: "train",
    1: "test",
    2: "val"
}
random_dict = {
    0: "spurious",
    1: "random"
}

def split_random_spurious(dataset, spurious_p, batch_size):
    random_p = 1 - spurious_p
    rnd_size, spur_size = int(len(dataset) * random_p), int(len(dataset) * spurious_p)
    rnd_set, spur_set = random_split(dataset, [rnd_size, spur_size])

    rnd_loader = DataLoader(rnd_set, batch_size=batch_size, shuffle=False)
    spur_loader = DataLoader(spur_set, batch_size=batch_size, shuffle=False)
    return rnd_loader, spur_loader

def generate_spurious_envs():
    spurious_env = random_generator[MODE](digits_to_store)
    flip_map = generate_random_flip_map()
    spurious_env_flip = flip_digit_color(flip_map, spurious_env)
    return spurious_env, spurious_env_flip

def transform_images(loader, split, random=0, spurious_env=None):
    if not random:
        assert spurious_env is not None
        images_, labels_ = transform_image_with_env(spurious_env, loader)
    else:
        images_, labels_ = transform_image_random(loader)
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
        else:
            image_path.extend([path for path in metadata['image_path']])
            labels.extend([label for label in metadata['label']])
            splits.extend([split for split in metadata['split']])
            random.extend([rnd for rnd in metadata['random']])
    metadata_all = {'image_path': image_path}
    metadata_all['label'] = labels
    metadata_all['split'] = splits
    metadata_all['random'] = random
    return metadata_all

if __name__ == '__main__':
    digits_to_store = [0,1]
    MODE = "background"

    batch_size = 1280

    store_dir = "/hdd2/dyah/coloredmnist_synthetic_spurious"
    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    p_to_generate = np.arange(.2,.3,.01)
    print(p_to_generate)
    spurious_p_test = 0.7

    spurious_env, spurious_env_flip = generate_spurious_envs()
    test_rnd_loader, test_spur_loader = split_random_spurious(test_set, spurious_p_test, batch_size)

    print("transforming test random set...")
    test_images_random, test_labels_random = transform_images(test_rnd_loader, split=1, random=1)
    
    print("transforming test spurious set...")
    test_images_spurious, test_labels_spurious = transform_images(test_spur_loader, split=1, random=0, spurious_env=spurious_env_flip)
    
    for spurious_p in tqdm(p_to_generate):
        store_dir_p = os.path.join(store_dir, str(spurious_p))
        train_rnd_loader, train_spur_loader = split_random_spurious(train_set, spurious_p, batch_size)
        val_rnd_loader, val_spur_loader = split_random_spurious(valid_set, spurious_p, batch_size)

        print("transforming random train images...")
        images_, labels_ = transform_images(train_rnd_loader, split=0, random=1)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=0, random=1)
        train_random_metadata = get_metadata(image_paths, labels_, split=0, random=1)
        
        print("transforming spurious train images...")
        images_, labels_ = transform_images(train_spur_loader, split=0, random=0, spurious_env=spurious_env)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=0, random=0)
        train_spurious_metadata = get_metadata(image_paths, labels_, split=0, random=0)

        print("transforming random val images...")
        images_, labels_ = transform_images(val_rnd_loader, split=2, random=1)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=2, random=1)
        val_random_metadata = get_metadata(image_paths, labels_, split=2, random=1)

        print("transforming spurious val images...")
        images_, labels_ = transform_images(val_spur_loader, split=2, random=0, spurious_env=spurious_env)
        image_paths = save_images_with_path(images_, labels_, store_dir_p, split=2, random=0)
        val_spurious_metadata = get_metadata(image_paths, labels_, split=2, random=0)


        image_paths = save_images_with_path(test_images_random, test_labels_random, store_dir_p, split=1, random=1)
        test_random_metadata = get_metadata(image_paths, test_labels_random, split=1, random=1)

        image_paths = save_images_with_path(test_images_spurious, test_labels_spurious, store_dir_p, split=1, random=0)
        test_spurious_metadata = get_metadata(image_paths, test_labels_spurious, split=1, random=0)

        metadata_all = merge_metadata([train_random_metadata, train_spurious_metadata, 
                              val_random_metadata, val_spurious_metadata,
                              test_random_metadata, test_spurious_metadata])
                              
        store_metadata(metadata_all, save_dir=store_dir_p)



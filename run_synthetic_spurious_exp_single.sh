CUDA_VISIBLE_DEVICES=$1
root_images_dir=/hdd2/dyah/uncorrelated_coloredmnist_synthetic_0/0.6

feat_path=$(basename $root_images_dir)
image_path=$root_images_dir
load_path=./artifacts/extracted_features/Synthetic_ColoredMNIST/$(basename $root_images_dir)
python get_spurious_samples.py -c configs/spur_exp/Synthetic_ColoredMNIST.yaml --log_path $feat_path  -img_path $image_path -feat_path $load_path
# done
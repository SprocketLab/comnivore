CUDA_VISIBLE_DEVICES=$1

python get_spurious_samples.py -c configs/spur_exp/Waterbirds.yaml
# python get_spurious_samples.py -c configs/spur_exp/CelebA.yaml
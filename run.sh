CUDA_VISIBLE_DEVICES=$0
python generate_candidate_sets.py -c configs/Synthetic_ColoredMNIST.yaml 
# python fuse_causal_estimates.py -c configs/irm_waterbirds.yaml
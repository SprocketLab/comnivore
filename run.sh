CUDA_VISIBLE_DEVICES=$1 
python generate_candidate_sets.py -c configs/Waterbirds.yaml 
python fuse_causal_estimates.py -c configs/Waterbirds.yaml 
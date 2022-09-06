CUDA_VISIBLE_DEVICES=$0
# python generate_candidate_sets.py -c configs/fuse_causal/FMoW.yaml 
python get_spurious_samples.py -c configs/spur_exp/FMoW.yaml 
# python fuse_causal_estimates.py -c configs/irm_waterbirds.yaml
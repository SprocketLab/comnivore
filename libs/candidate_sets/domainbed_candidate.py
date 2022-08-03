from libs.candidate_sets.utils.domainbed_const import HOLDOUT_FRACTION
from .utils import domainbed_const
from libs.domainbed.lib import misc
from libs.domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from torchvision import transforms

split_seed = domainbed_const.SPLIT_SEED
holdout_fraction = domainbed_const.HOLDOUT_FRACTION
test_envs = domainbed_const.TEST_ENVS

class DomainBed_Candidate_Set:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def get_splits(self, test_envs):
        in_splits = []
        out_splits = []
        for env_i, env in enumerate(self.dataset):
            out, in_ = misc.split_dataset(env,
                                            int(len(env) * holdout_fraction),
                                            misc.seed_hash(split_seed, env_i))

            if env_i in test_envs:
                uda, in_ = misc.split_dataset(in_,
                                                int(len(in_) * 0),
                                                misc.seed_hash(split_seed, env_i))
            
            in_weights, out_weights = None, None
            in_splits.append((in_, in_weights))
            out_splits.append((out, out_weights))
        return in_splits, out_splits
        
    def get_train_loader(self, batch_size, extra_transform=None):
        in_splits, _ = self.get_splits(test_envs)
        for i, (env, _) in enumerate(in_splits):
            if i == 0:
                transform_list = vars(env.underlying_dataset.transforms.transform)['transforms']
                if extra_transform is not None:
                    extra_transform_list = vars(extra_transform)['transforms']
                    for item in extra_transform_list:
                        transform_list.insert(1, item)
                env.underlying_dataset.transforms.transform = transforms.Compose(transform_list)
        train_loaders = [InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=0)
            for i, (env, env_weights) in enumerate(in_splits)
            if i not in test_envs]
        return zip(*train_loaders)
    
    def get_val_loader(self, batch_size):
        in_splits, out_splits = self.get_splits(test_envs)
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=batch_size,
            num_workers=0)
            for env, _ in (in_splits + out_splits)]
        eval_weights = [None for _, weights in (in_splits + out_splits)]
        eval_loader_names = ['env{}_in'.format(i)
                            for i in range(len(in_splits))]
        eval_loader_names += ['env{}_out'.format(i)
                            for i in range(len(out_splits))]
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        return evals
from .wilds_candidate import WILDS_Candidate
from torchvision import transforms

dataset_name = "fmow"

class FMoW_Candidate_Set(WILDS_Candidate):
    def __init__(self, reshape_size, batch_size):
        super(FMoW_Candidate_Set, self).__init__(dataset_name, reshape_size, batch_size)
        self.batch_size = batch_size
        self.reshape_size = reshape_size
    
    def get_loader_dict(self):
        return {
            'orig': super().get_train_loader_orig(),
            'augment': self.get_train_loader_augment(),
        }
    
    def get_train_loader_augment(self):
        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(self.reshape_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _, trainloader_augment = super().get_train_dataloader(
            self.batch_size,
            augment_transform,
        )
        return trainloader_augment
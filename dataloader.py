from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.functional as TF



class pet_dataset(Dataset):
    def __init__(self,data, image_transforms, mask_transforms):
        super().__init__()
        self.data = data
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image, mask = self.data[index]
        image_resized = TF.resize(image, size=[224,224],interpolation= TF.InterpolationMode.BILINEAR)
        mask_resized = TF.resize(mask, size=[224,224],interpolation= TF.InterpolationMode.NEAREST)
        image_transformed= self.image_transforms(image_resized)
        mask_transformed = self.mask_transforms(mask_resized)
        mask_transformed -=1  # as our classes as 1,2,3 but pytorch needs as 0,1,2
        return image_transformed, mask_transformed.squeeze(0).long()


def get_dataloader():

    image_transforms = transforms.Compose([ transforms.ToTensor()])
    mask_transforms = transforms.Compose([transforms.PILToTensor()])
    
    train_data = datasets.OxfordIIITPet("./data", download=True, split="trainval",target_types="segmentation")
    test_data = datasets.OxfordIIITPet("./data", download=True, split="test",target_types="segmentation")


    train_data = pet_dataset(train_data, image_transforms, mask_transforms)
    test_data = pet_dataset(test_data, image_transforms, mask_transforms)

    train_loader = DataLoader(train_data, batch_size=12, shuffle=True, pin_memory=False, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=12, shuffle=True, pin_memory=False, num_workers=2) 
    return train_loader, test_loader
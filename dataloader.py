from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class data_class(Dataset):
    def __init__(self,data, image_transforms, name):
        super().__init__()
        self.data = data
        self.image_transforms = image_transforms
        self.name = name
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image, mask = self.data[index]
        transformed_img_mask= self.image_transforms(image= np.array(image), mask=np.array(mask))
        image_transformed = transformed_img_mask["image"]
        mask_transformed = transformed_img_mask["mask"]
        if self.name=="oxford":
            mask_transformed -=1  # as our classes as 1,2,3 but pytorch needs as 0,1,2
        elif self.name=="voc":
            mask_transformed[mask_transformed == 255] = 21
        return image_transformed, mask_transformed.squeeze(0).long()


def get_dataloader(data="oxford"):

    image_transforms = A.Compose([A.Resize(height=224,width=224),
                                  A.HorizontalFlip(p=0.5),
                                  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                  A.ShiftScaleRotate(),
                                  ToTensorV2()])
    
    if data=="oxford":
        train_data = datasets.OxfordIIITPet("./oxford", download=True, split="trainval",target_types="segmentation")
        test_data = datasets.OxfordIIITPet("./oxford", download=True, split="test",target_types="segmentation")
    elif data=="voc":
        train_data= datasets.VOCSegmentation("./voc",image_set="train",year="2012",download=True)
        test_data= datasets.VOCSegmentation("./voc",image_set="val",year="2012",download=True)


    train_data = data_class(train_data, image_transforms, name=data)
    test_data = data_class(test_data, image_transforms, name=data)

    train_loader = DataLoader(train_data, batch_size=12, shuffle=True, pin_memory=False, num_workers=5)
    test_loader = DataLoader(test_data, batch_size=12, shuffle=True, pin_memory=False, num_workers=2) 
    return train_loader, test_loader
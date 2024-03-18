from einops import rearrange
import torch

def segment_image_oxford(image, mask):
    mask = mask.unsqueeze(0).repeat(3,1,1)
    image[mask==1]=0
    image[mask==2]= 255
    return image

def segment_image_voc(image, mask):
    mask = mask.unsqueeze(0).repeat(3,1,1)
    image[mask==0]=255
    return image

def denormalise(image, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    for img, mean, std in zip(image, means, stds):
        img.mul_(std).add_(mean)
    return image

import matplotlib.pyplot as plt 
def plot_prediction(image, label, pred, data_name):
    image = denormalise(image)

    if data_name=="oxford":
        label_seg = segment_image_oxford(image.clone(), label)
        pred_seg = segment_image_oxford(image.clone(), pred)
    else:
        label_seg = segment_image_voc(image.clone(), label)
        pred_seg = segment_image_voc(image.clone(), pred)
    plt.figure(figsize=(16,8))
    plt.subplot(1,3,1)
    plt.imshow(rearrange(image, "c h w -> h w c"))
    plt.axis('off')
    plt.title("Image")
    plt.subplot(1,3,2)
    plt.imshow(rearrange(label_seg, "c h w -> h w c"), cmap="gray")
    plt.axis('off')
    plt.title("True_Image_segmented")
    plt.subplot(1,3,3)
    plt.imshow(rearrange(pred_seg, "c h w -> h w c"), cmap="gray")
    plt.title("Predicted_Image_segmented")
    plt.axis("off")

import torch.nn.functional as F
def dice_loss(output, target, num_classes):
    pred = F.softmax(output, dim=1)
    dice_loss = 0
    target = rearrange(torch.nn.functional.one_hot(target,num_classes),"b h w c -> b c h w")
    classes_present = torch.sum(target,dim = (2,3))>0
    intersection = torch.sum(pred * target, dim= (2,3)) # [N,C]
    union = torch.sum(pred, dim=(2,3)) + torch.sum(target, dim=(2,3))
    dice_cofficient = 2 * intersection / (union + 1e-5)
    dice_loss = (1 - dice_cofficient) * classes_present
    return dice_loss.sum()/ classes_present.sum()
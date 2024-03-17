from einops import rearrange
def segment_image(image, mask):
    mask = mask.unsqueeze(0).repeat(3,1,1)
    image[mask==1]=0
    image[mask==2]= 255
    return image

import matplotlib.pyplot as plt 
def plot_prediction(image, label, pred):
    label_seg = segment_image(image.clone(), label)
    pred_seg = segment_image(image.clone(), pred)
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
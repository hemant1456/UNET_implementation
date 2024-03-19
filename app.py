import gradio as gr
import torch
from torchvision import transforms

from UNET_model import UNET
from utils import segment_image_oxford, segment_image_voc

model = UNET(3, 64, num_classes=3, loss_type="dice", pool_type="strided", size_increase="transpose") #class 3 for oxfored and 22 for voc
model.load_state_dict(torch.load("./unet_model_oxford.pth"))


def image_segment(image):
    image_transforms = transforms.Compose([ transforms.Resize((224,224)),transforms.ToTensor()])
    transformed_image = image_transforms(image)
    output = model(transformed_image.unsqueeze(0))
    pred = output.argmax(dim=1)
    image_out = segment_image_oxford(transformed_image, pred[0])
    to_pil_image = transforms.ToPILImage()
    return to_pil_image(image_out)


app = gr.Interface(image_segment, inputs=[gr.Image(type="pil", height=224, width=224)], outputs=[gr.Image(type="pil", height=224, width=224)], examples=['./examples/Abyssinian_2.jpg','./examples/american_bulldog_101.jpg',
                                                                                              './examples/american_pit_bull_terrier_184.jpg'])
app.launch()
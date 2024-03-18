from encoder import EncoderBlock
from decoder import DecoderBlock
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy

class UNET(L.LightningModule):
    def __init__(self, in_channels, initial_filters, num_classes=3):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels, initial_filters, to_pool=False)   # 224x224x64
        self.encoder_block2 = EncoderBlock(initial_filters, initial_filters*2)   # 112x112x128
        self.encoder_block3 = EncoderBlock(initial_filters*2, initial_filters*4)  # 56x56x1256
        self.encoder_block4 = EncoderBlock(initial_filters*4, initial_filters*8) # 28x28x512
        self.encoder_block5 = EncoderBlock(initial_filters*8, initial_filters*16) # 14x14x512

        self.decoder_block1 = DecoderBlock(16* initial_filters, 8* initial_filters) # 28x28x256
        self.decoder_block2 = DecoderBlock(8* initial_filters, 4* initial_filters) # 56x56x128
        self.decoder_block3 = DecoderBlock(4* initial_filters,  2* initial_filters) # 112x112x64
        self.decoder_block4 = DecoderBlock(2 * initial_filters,  initial_filters) # 224x224x64

        self.classifer = nn.Conv2d(initial_filters, num_classes, 1, 1, 0)
        self.accuracy= Accuracy(num_classes=num_classes, task="multiclass")
    def forward(self,x):
        x_1 = self.encoder_block1(x)
        x_2 = self.encoder_block2(x_1)
        x_3 = self.encoder_block3(x_2)
        x_4 = self.encoder_block4(x_3)
        x_5 = self.encoder_block5(x_4)

        x_dec_1 = self.decoder_block1(x_5, x_4)
        x_dec_2 = self.decoder_block2(x_dec_1, x_3)
        x_dec_3 = self.decoder_block3(x_dec_2, x_2)
        x_dec_4 = self.decoder_block4(x_dec_3, x_1)

        out = self.classifer(x_dec_4)
        return out
    def training_step(self, batch, batch_idx):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out,label)
        accuracy = self.accuracy(out, label)
        self.log("training_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True, on_step=True)
        return loss
    def validation_step(self, batch, batch_idx):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out,label)
        accuracy = self.accuracy(out, label)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("validation_accuracy", accuracy, prog_bar=True, on_epoch=True, on_step=True)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer        
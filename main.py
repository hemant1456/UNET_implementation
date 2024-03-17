from dataloader import get_dataloader
from UNET_model import UNET
import lightning as L
import torch

if __name__=="__main__":
    
    train_loader, test_loader = get_dataloader()
    model = UNET(3, 64, 3)
    trainer = L.Trainer(max_epochs=4, limit_val_batches=10)
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_loader, test_loader)
    torch.save(model.state_dict(), "unet_model.pth")

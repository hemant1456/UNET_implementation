from dataloader import get_dataloader
from UNET_model import UNET
import lightning as L
import torch
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="voc",help="dataset to train on",choices=["voc","oxford"])
    parser.add_argument("--batch_size", type=int, default=32 ,help="batch size")
    parser.add_argument("--loss", type=str, default="ce",help="loss function to use",choices=["ce","dice"])
    parser.add_argument("--size_increase", type=str, default="transpose", choices=["transpose", "upsample"],help="method to use in decoder to increase size" )
    parser.add_argument("--pool_type", type=str, choices=["strided","max_pool"],default="strided", help="how to reduce feature map in encoder block")
    args = parser.parse_args()

    num_classes = 3 if args.dataset=="oxford" else 22
    
    train_loader, test_loader = get_dataloader(args.dataset)
    model = UNET(3, 64, num_classes=num_classes, loss_type=args.loss, size_increase=args.size_increase, pool_type=args.pool_type)
    trainer = L.Trainer(max_epochs=4, limit_val_batches=10)
    torch.set_float32_matmul_precision("medium")
    trainer.fit(model, train_loader, test_loader)
    torch.save(model.state_dict(), "unet_model_voc.pth")

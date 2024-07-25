import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import pandas as pd
import wandb

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize DataModule
    data_module = DInterface(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             val_split=args.val_split,
                             num_workers=args.num_workers,
                             augment=True)
    
    # Initialize Model
    model = MInterface(input_dim=args.input_dim,
                       num_heads=args.num_heads, 
                        dropout_rate=args.dropout_rate,
                       lr=args.lr)
    model.to(device)  # Move model to the device

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=True)

    # Initialize wandb
    # wandb.init(project='Insurance-Cross-Selling', config=args)
    # Create WandbLogger
    # wandb_logger = WandbLogger()
    csv_logger = CSVLogger("logs", name="zzh")
    
    trainer = Trainer(max_epochs=args.max_epochs,
                      callbacks=[checkpoint_callback],
                      logger=csv_logger,
                      accelerator="auto",  # Use auto to let PyTorch Lightning handle GPU
                      devices=1 if device.type == 'cuda' else None)  # Specify number of GPUs if CUDA

    trainer.fit(model, data_module)

def predict(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize DataModule and Model
    data_module = DInterface(data_dir=args.data_dir,
                             num_workers=args.num_workers,
                             batch_size=args.batch_size)
    data_module.setup(stage='test')
    
    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            input_dim=args.input_dim,
                                            num_heads=args.num_heads, 
                                            dropout_rate=args.dropout_rate,
                                            lr=args.lr)
    model.to(device)  # Move model to the device

    # Prepare Test Data and DataLoader
    test_data = pd.read_csv(f'{args.data_dir}/test.csv')
    test_loader = data_module.test_dataloader()

    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            print(batch)  # Print the batch to inspect its structure
            # Process the batch based on its actual structure
            # Example: if batch is a Tensor, directly move to device
            if isinstance(batch, torch.Tensor):
                x = batch
                x = x.to(device)
                preds = model(x)
                test_pred.extend(preds.cpu().numpy())
            elif isinstance(batch, tuple):
                x, _ = batch  # Adjust based on actual batch structure
                x = x.to(device)
                preds = model(x)
                test_pred.extend(preds.cpu().numpy())
            # Add other conditions if necessary

    submission = pd.DataFrame({'Id': test_data['id'], 'Response': np.array(test_pred).ravel()})
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--input_dim', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--model_checkpoint', type=str, 
                        default='logs/zzh/version_2/checkpoints/epoch=0-step=17977.ckpt')
    torch.set_float32_matmul_precision('medium')
    args = parser.parse_args()
    if args.is_train:
        train(args)
    else:
        predict(args)

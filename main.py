import argparse
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import DInterface
from model import MInterface
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
import wandb

def train(args):
    data_module = DInterface(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             val_split=args.val_split,
                             augment=True)

    model = MInterface(input_dim=args.input_dim,
                       hidden_dim=args.hidden_dim,
                       lr=args.lr)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=True)

    # 初始化 wandb
    wandb.init(project='Insurance-Cross-Selling', config=args)
    # 创建 WandbLogger
    wandb_logger = WandbLogger()

    trainer = Trainer(max_epochs=args.max_epochs, callbacks=[checkpoint_callback], logger=wandb_logger)
    trainer.fit(model, data_module)
    wandb.finish()


def predict(args):
    # Initialize DataModule and Model
    data_module = DInterface(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.setup(stage='test')
    model = MInterface.load_from_checkpoint(args.model_checkpoint,
                                            input_dim=args.input_dim,
                                            hidden_dim=args.hidden_dim,
                                            lr=args.lr)

    # Prepare Test Data and DataLoader
    test_data = pd.read_csv(f'{args.data_dir}/test.csv')
    test_loader = data_module.test_dataloader()

    model.eval()
    test_pred = []
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch)
            test_pred.extend(preds.cpu().numpy())

    submission = pd.DataFrame({'Id': test_data['id'], 'Response': np.array(test_pred).ravel()})
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--model_checkpoint', type=str, default='')

    args = parser.parse_args()
    if args.is_train:
        train(args)
    else:
        predict(args)

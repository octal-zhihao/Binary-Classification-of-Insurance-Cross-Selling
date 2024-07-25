import pandas as pd
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import numpy as np
from .dataset import InsuranceDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=16, val_split=0.2, augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment = augment

    def setup(self, stage=None):
        # Load data
        train_data = pd.read_csv(f'{self.data_dir}/train.csv')
        test_data = pd.read_csv(f'{self.data_dir}/test.csv')

        # Identify categorical and numerical columns
        categorical_features = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        numerical_features = ['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

        # One-Hot Encoding for categorical features
        train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
        test_data = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)

        # Ensure train and test have the same columns after one-hot encoding
        train_data, test_data = train_data.align(test_data, join='left', axis=1, fill_value=0)

        if 'Response' in test_data.columns:
            test_data = test_data.drop(columns=['Response'])

        # Standard Scaling for numerical features
        scaler = StandardScaler()
        train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
        test_data[numerical_features] = scaler.transform(test_data[numerical_features])

        # Drop any remaining non-numeric columns
        X_train = train_data.drop(columns=['id', 'Response']).astype(np.float32)
        X_test = test_data.drop(columns=['id']).astype(np.float32)

        # Ensure 'Response' column is float32
        y = train_data['Response'].astype(np.float32)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=self.val_split, random_state=42)

        # Create datasets
        self.train_dataset = InsuranceDataset(X_train, y_train)
        self.val_dataset = InsuranceDataset(X_val, y_val)
        self.test_dataset = InsuranceDataset(X_test, None)

        # Print column names
        # print("Train columns:", X_train.columns.tolist())
        # print("Test columns:", X_test.columns.tolist())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# In your main function or script
# if __name__ == "__main__":
#     data_module = DInterface(data_dir='../data')
#     data_module.setup()

#     # Check DataLoader
#     train_loader = data_module.train_dataloader()
#     for batch in train_loader:
#         x, y = batch
#         print("Batch input shape:", x.shape)
#         print("Batch labels shape:", y.shape)
#         break  # Just check the first batch
#     test_loader = data_module.test_dataloader()
#     for batch in test_loader:
#         x = batch
#         print("Batch input shape:", x.shape)
#         break  # Just check the first batch

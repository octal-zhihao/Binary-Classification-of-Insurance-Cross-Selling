import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import InsuranceDataset

class DInterface(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=16, val_split=0.2, augment=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augment = augment

        # Initialize preprocessing objects
        self.le_gender = LabelEncoder()
        self.le_vehicle_age = LabelEncoder()
        self.le_vehicle_damage = LabelEncoder()
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            df = pd.read_csv(f'{self.data_dir}/train.csv')

            # Fit label encoders and scaler
            df['Gender'] = self.le_gender.fit_transform(df['Gender'])
            df['Vehicle_Age'] = self.le_vehicle_age.fit_transform(df['Vehicle_Age'])
            df['Vehicle_Damage'] = self.le_vehicle_damage.fit_transform(df['Vehicle_Damage'])
            df[['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']] = self.scaler.fit_transform(
                df[['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']])

            X = df.drop(columns=['id', 'Response'])
            y = df['Response']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=42)

            self.train_dataset = InsuranceDataset(X_train, y_train)
            self.val_dataset = InsuranceDataset(X_val, y_val)

        if stage == 'test' or stage is None:
            df = pd.read_csv(f'{self.data_dir}/test.csv')

            # Use the same label encoders and scaler
            df['Gender'] = self.le_gender.transform(df['Gender'])
            df['Vehicle_Age'] = self.le_vehicle_age.transform(df['Vehicle_Age'])
            df['Vehicle_Damage'] = self.le_vehicle_damage.transform(df['Vehicle_Damage'])
            df[['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']] = self.scaler.transform(
                df[['Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']])

            X_test = df.drop(columns=['id'])
            self.test_dataset = InsuranceDataset(X_test, None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

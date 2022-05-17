
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from copy import deepcopy


# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
NUM_WORKERS = 8                 
HIDDEN_DIM = 4096               
LEARNING_RATE = 0.0002754       
PATIENCE = EPOCHS/4             
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Preprocessing
train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# remove id column
train_df.drop(["Id"], axis=1, inplace=True)
test_df.drop(["Id"], axis=1, inplace=True)

# dropping target column from train data before preprocessing
# will re-add after
target_col_name = train_df.columns[-1]
target_col = train_df[target_col_name]
train_df.drop(target_col_name, axis=1, inplace=True)

# EDA has shown all NaNs can be replaced with 0
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)


# create new features in {1,0} for each option of each categorical features
# note: I'm using unique(), which assumes the training dara has examples
# of each option.

for idx, c in enumerate(train_df.columns):
    if train_df[c].dtype == "object":
        options = train_df[c].unique()
        for opt in options:
            # add column for each option
            train_df[f"{c}_{opt}"] = (train_df[c] == opt).astype(int)
            test_df[f"{c}_{opt}"] = (test_df[c] == opt).astype(int)
        # drop the original cateogrical column
        train_df.drop(c, axis=1, inplace=True)
        test_df.drop(c, axis=1, inplace=True)

IN_FEATURES = train_df.shape[1]  # number of columns


# add the target column back
train_df[target_col_name] = target_col

# split the training data into train and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.15, shuffle=False)
test_data = test_df

# Custom Dataset class
class HousingPricesDataset(Dataset):
    def __init__(self, df, test=False): 
        self.test = test
        if test:
            # test set,  no target column    
            self.data = torch.tensor(df.values).float()
        else:
            # train and val sets
            self.data = torch.tensor(df.iloc[:, :-1].values).float()
            self.target = torch.tensor(df.iloc[:, -1].values).float()
        self.data = F.normalize(self.data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, i):
        if self.test:
            return self.data[i]
        else:
            return self.data[i], self.target[i]


train_dataset = HousingPricesDataset(train_data) 
val_dataset = HousingPricesDataset(val_data)
test_dataset = HousingPricesDataset(test_data, test=True)


train_dl = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)
val_dl = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True,
)
test_dl = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True
)


class KiteNet(nn.Module):
    """
           *  
        *    *  
   ~~*         *
        *    *
           *
    
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.kite = nn.Sequential(
            nn.Linear(in_dim, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim//2, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc1 = nn.Linear(in_dim, (in_dim+out_dim)//2)
        self.fc2 = nn.Linear((in_dim+out_dim)//2, out_dim)


    def forward(self, x):
        x = self.kite(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = KiteNet(in_dim=IN_FEATURES, hidden_dim=HIDDEN_DIM, out_dim=1)
#model.load_state_dict(torch.load('trained_model.m'))
model.to(DEVICE)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2.0)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE, verbose=True)


# training loop
if __name__ == "__main__":
    running_losses = []
    running_val_losses = []
    min_val_loss = float('inf')

    for epoch in tqdm(range(EPOCHS)):
        running_loss = 0
        running_val_loss = 0

        model.train()
        for i, batch in enumerate(train_dl):
            data, target = batch
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            output = output.squeeze(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for val_batch in val_dl:
                val_data, val_target = val_batch
                val_data, val_target = val_data.to(DEVICE), val_target.to(DEVICE)
                val_output = model(val_data)
                val_output = val_output.squeeze(-1)
                val_loss = criterion(val_output, val_target)
                running_val_loss += val_loss.item()

        scheduler.step(running_val_loss)
        running_val_losses.append(running_val_loss / len(val_dl))
        running_losses.append(running_loss / len(train_dl))
        if running_val_loss < min_val_loss:
            min_val_loss = running_val_loss
            torch.save(deepcopy(model.state_dict()), "best_model.m")

    

    torch.save(deepcopy(model.state_dict()), "trained_model.m")

    
    # Evaluation the best model on the val set
    model = KiteNet(in_dim=IN_FEATURES, hidden_dim=HIDDEN_DIM, out_dim=1)
    model.load_state_dict(torch.load('best_model.m'))
    model.to(DEVICE)
    
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True)
    model.eval()
    val_preds = []
    for batch in val_dl:
        data, target = batch
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        output = output.squeeze(-1)
        loss = criterion(output, target)
        val_preds.append(
            (output.cpu().detach().numpy()[0], target.cpu().detach().numpy()[0])
        )

    mean_error = np.mean([abs(p - t) for p, t in val_preds])

    with open("log.txt", "a") as f:
        f.write(f"{BATCH_SIZE=}, {LEARNING_RATE=}{HIDDEN_DIM=}, {EPOCHS=}, {mean_error=}\n")


    print(mean_error)
    
    # Run the model on the test set
    model.eval()
    with torch.no_grad():
        test_preds = []
        for example in test_dl:
            example = example.to(DEVICE)
            output = model(example)
            test_preds.append(output.item())

    
    # Save the predictions to a file
    with open("submission.csv", "w") as f:
        f.write("Id,SalePrice\n")
        for i, pred in enumerate(test_preds):
            f.write(f"{i+1461},{pred}\n")


    # plot training loss
    plt.plot(running_losses)
    plt.show()

    
    # plot val loss
    plt.plot(running_val_losses)
    plt.show()
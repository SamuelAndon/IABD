import pandas as pd
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from sklearn.preprocessing import LabelEncoder
import lightning.pytorch as pl


class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def __repr__(self):
        return f"mean: {self.mean}, std:{self.std}, epsilon:{self.epsilon}"
    

class AutomobilDataset(Dataset):
  def __init__(self, src_file, root_dir, transform=None):

    automobilDataset = pd.read_csv('imports-85.data', header=None)

    automobilDataset.replace('?', pd.NA, inplace=True)
    automobilDataset.dropna(subset=automobilDataset.columns, inplace=True)

    columnas_categoricas = [2, 3, 4, 5, 6, 7, 8, 14, 15, 17]

    class_label_encoder = LabelEncoder()

    for i in columnas_categoricas:
        automobilDataset[i] = class_label_encoder.fit_transform(automobilDataset[i])

    print(automobilDataset.info())

    automobilDataset[[1, 18, 19, 21, 22, 25]] = automobilDataset[[1, 18, 19, 21, 22, 25]].astype(float)

    X = automobilDataset.iloc[:,:25]
    Y = automobilDataset.iloc[:, 25]

    s1=X.iloc[:,0:25].values
    x_tensor = torch.tensor(s1)
    
    y_tensor = torch.tensor(Y.values).type(torch.float32)
    scaler = StandardScaler()
    XScalada = scaler.fit_transform(x_tensor).type(torch.float32)

    self.data = torch.cat((XScalada,y_tensor.unsqueeze(1)),1)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    preds = self.data[idx, 0:25]
    spcs = self.data[idx, 25:]
    sample = (preds, spcs)
    if self.transform:
      sample = self.transform(sample)
    return sample
  

class Model(pl.LightningModule):
   
    def __init__(self, entradas):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(entradas, 100)
        self.layer2 = nn.Linear(100, 50)
        self.layer3 = nn.Linear(50, 1)
    
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        self.log('"Train/Loss"', loss, on_step= False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        vinputs, vlabels = val_batch
        voutputs = model(vinputs)

        mean_squared_error = MeanSquaredError()
        mean_absolute_error = MeanAbsoluteError()

        mean_squared_error(voutputs,vlabels)
        mean_absolute_error(voutputs,vlabels)

        errorMedio = mean_squared_error.compute()
        errorAbsolute =mean_absolute_error.compute()

        loss = loss_fn(voutputs, vlabels)
        self.log('Test/Loss', loss)
        self.log('Tes/Squared Error', errorMedio)
        self.log('Train/Absolute Error', errorAbsolute)

automobilDataset = AutomobilDataset("imports-85.data",".")
print(automobilDataset[0])

lonxitudeDataset = len(automobilDataset)
tamTrain =int(lonxitudeDataset*0.8)
tamVal = lonxitudeDataset - tamTrain
print(f"Tam dataset: {lonxitudeDataset} train: {tamTrain} tamVal: {tamVal}")
train_set, val_set = random_split(automobilDataset,[tamTrain,tamVal])
train_ldr = torch.utils.data.DataLoader(train_set, batch_size=2,
    shuffle=True, drop_last=False)
validation_loader =torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, drop_last=True)

model     = Model(25)
loss_fn   = nn.MSELoss(reduction='sum')
trainer = pl.Trainer(accelerator="cpu",num_nodes=2, devices=2, precision=32,
                    limit_train_batches=0.5, max_epochs=100)
trainer.fit(model, train_ldr, validation_loader)

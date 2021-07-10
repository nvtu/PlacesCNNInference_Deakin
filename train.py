import h5py
import torch
from torch.autograd import Variable as V
import configparser
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import configparser
import os
import os.path as osp
from tqdm import tqdm
import numpy as np


class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H_1, H_2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H_1)
        self.linear2 = torch.nn.Linear(H_1, H_2)
        self.linear3 = torch.nn.Linear(H_2, D_out)
        self.softmax = torch.nn.Softmax(dim=1)
        self.bn0 = torch.nn.BatchNorm1d(D_in)
        self.bn1 = torch.nn.BatchNorm1d(H_1)
        self.bn2 = torch.nn.BatchNorm1d(H_2)
        self.bn3 = torch.nn.BatchNorm1d(D_out)
        self.elu = torch.nn.ELU()
        
    
    def forward(self, x):
        x_bn = self.bn0(x)
        h1_elu = self.elu(self.linear1(x_bn))
        h1_elu_bn = self.bn1(h1_elu)
        h2_elu = self.elu(self.linear2(h1_elu_bn))
        h2_elu_bn = self.bn2(h2_elu)
        out = self.elu(self.linear3(h2_elu_bn))
        out_bn = self.bn3(out)
        pred = self.softmax(out_bn)
        return pred


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        total_items, _ = self.X.shape
        return total_items
    
    def __getitem__(self, index):
        _X = self.X[index]
        _y = self.y[index]
        return _X, _y


# ---------------------------------------------------------------- PREPARE MODEL OUTPUT FILE PATH -----------------------------------------------------------
config = configparser.ConfigParser()
config.read('config.ini')
processed_data_path = config['PATH']['PROCESSED_DATA_DIR']
model_dir = config['PATH']['MODEL_DIR']
if not osp.exists(model_dir):
    os.makedirs(model_dir)

# ---------------------------------------------------------------- PREPARE DATA FOR TRAINING ----------------------------------------------------------------
h5_file_path = osp.join(processed_data_path, 'train_data.h5')
with h5py.File(h5_file_path, 'r') as f:
    X = f['X_train'][()]
    y = f['y_train'][()]

total_training_items, feat_size = X.shape
num_classes = len(list(set(y)))

# Convert into torch variable tensor
X = V(torch.from_numpy(X))
y = V(torch.from_numpy(y))

# Train/Val split
seed = 2020
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

# Train dataloader with batch_size
params = {
    'batch_size': 1000,
    'shuffle': True,
}
train_set = Dataset(X_train, y_train)
train_generator = torch.utils.data.DataLoader(train_set, **params)

print(f"Number of training items: {X_train.shape[0]}")
print(f"Number of testing items: {X_test.shape[0]}")

# ---------------------------------------------------------------- START PARAMS FOR TRAINING ----------------------------------------------------------------
D_in, H_1, H_2, D_out = feat_size, 256, 128, num_classes
learning_rate = 1e-4
max_epochs = 100
best_score = 0.0
best_loss = 1e18

model = ThreeLayerNet(D_in, H_1, H_2, D_out)

optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

model_index = 1
for epoch in tqdm(range(max_epochs)):
    losses = []
    # Forward pass
    for X_batch, y_batch in train_generator:
        y_pred_prob = model(X_batch)
        # Compute loss
        loss = loss_fn(y_pred_prob, y_batch)

        y_pred = torch.argmax(y_pred_prob, dim=1)
        train_acc = accuracy_score(y_batch.detach().numpy(), y_pred.detach().numpy())
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    # Compute validation lost and acc

    if (epoch + 1) % 10 == 0:
        mean_loss = np.mean(losses)
        print(f"Mean loss train iter {epoch + 1}: {mean_loss}")
        with torch.set_grad_enabled(False):
            y_pred_prob = model(X_test)
            loss = loss_fn(y_pred_prob, y_test)
            y_pred = torch.argmax(y_pred_prob, dim=1)
            test_acc = accuracy_score(y_test.detach().numpy(), y_pred.detach().numpy())
            print(f"Loss valid iter {epoch + 1}: {loss.item()}")
            print(f"Accuracy valid iter {epoch + 1}: {test_acc}")

            if best_loss >= mean_loss and best_score <= test_acc:
                print(f"Saving model_{model_index}...")
                model_file_path = osp.join(model_dir, f'model_{model_index}.pt')
                torch.save(model.state_dict(), model_file_path)
                model_index += 1
                best_loss = mean_loss
                best_score = test_acc
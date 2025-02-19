import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Evaluation:
    @staticmethod
    def evaluate(loss, test_loader, model, device):
        with torch.no_grad():
            total_loss = 0
            results = []
            for test_loader_data, labels in test_loader:
                test_loader_data = test_loader_data.to(device)
                labels = labels.to(device)

                outputs = model(test_loader_data).squeeze(1)
                total_loss += loss(outputs, labels)

                results.extend(zip(labels, outputs))

            return total_loss / len(test_loader), results


class RegressionModel(nn.Module):
    def __init__(self, input_channels_number):
        super(RegressionModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_channels_number, out_features=input_channels_number * 4),
            nn.ReLU(),
            nn.Linear(in_features=input_channels_number * 4, out_features=1)
        )

    def forward(self, x):
        return self.classifier(x)


class Training:
    @staticmethod
    def train(epochs, device, optimizer, model, loss, train_loader, validation_loader, threshold):
        old_validation_value = Validation.validate(validation_loader, device, model, loss)
        counter = 0
        best_weights = cp.deepcopy(model.state_dict())
        losses = []

        for i in range(epochs):
            epoch_loss = 0

            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                prediction = model(data).squeeze(1)

                current_loss = loss(prediction, labels)

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                epoch_loss += current_loss.item()

            print(f'epoch {i + 1}, loss per item: {epoch_loss / len(train_loader)}')

            current_validation_value = Validation.validate(validation_loader, device, model, loss)

            losses.append(current_validation_value)

            if current_validation_value <= old_validation_value:
                old_validation_value = current_validation_value
                counter = 0
                best_weights = cp.deepcopy(model.state_dict())
            else:
                if counter < threshold:
                    counter += 1
                else:
                    model.load_state_dict(best_weights)
                    print(f"Risk of over fitting parameters, ending learning curve at iteration {i + 1}")
                    return losses[: -counter]
        return losses


class Validation:
    @staticmethod
    def validate(validation_loader, device, model, loss):
        with torch.no_grad():
            current_validation_value = 0

            for validation_loader_data, validation_labels in validation_loader:
                validation_loader_data = validation_loader_data.to(device)
                validation_labels = validation_labels.to(device)

                val_prediction = model(validation_loader_data).squeeze(1)

                validation_loss = loss(val_prediction, validation_labels)
                current_validation_value += validation_loss.item()

            return current_validation_value / len(validation_loader)


class CsvLoader(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == '__main__':
    epochs = 10000
    BATCH_SIZE = 10

    encoder = LabelEncoder()

    dFrame = pd.read_csv('Concrete_Data.csv', header=0)

    train_data, test_data = train_test_split(dFrame, test_size=0.2)
    train_data, validation_data = train_test_split(train_data, test_size=0.125)

    train_data = CsvLoader(train_data)
    test_data = CsvLoader(test_data)
    validation_data = CsvLoader(validation_data)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RegressionModel(len(dFrame.iloc[:, :-1].values[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = nn.L1Loss()

    losses = Training.train(epochs, device, optimizer, model, loss, train_loader, validation_loader, 20)

    print(f"training of model complete")

    total_loss, results = Evaluation.evaluate(loss, test_loader, model, device)

    print(f'results of prediction: {np.asarray(results)}')
    print(f'Loss per item in test dataset: {total_loss}')

    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss per Item')

    plt.plot(losses)

    plt.show()

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from regression_model import RegressionModel
from csvloader import CsvLoader
from training import Training
from evaluation import Evaluation


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

    print(f'results: {np.asarray(results)}')
    print(f'Loss per item in test: {total_loss}')

    plt.title('Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Loss per item')

    plt.plot(losses)

    plt.show()

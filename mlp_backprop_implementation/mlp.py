import numpy as np
import pandas as pd
import copy as cp


# np.random.seed(42)


class Layer:
    limit = 0.000001
    """
    This is just a dummy class that is supposed to represent the general
    functionality of a neural network layer. Each layer can do two things:
     - forward pass - prediction
     - backward pass - training
    """

    def __init__(self):
        pass

    def forward(self, inp):
        # a dummy layer returns the input
        return inp

    def backward(self, inp, grad_outp):
        placeholder = []

        for idx, val in enumerate(inp):
            wrapper = []

            for index, value in enumerate(val):
                copy = cp.deepcopy(val).astype('float64')
                copy[index] = value + self.limit

                wrapper.append((self.forward(copy) - self.forward(val)) / self.limit)

            placeholder.append(wrapper)

        grad_outp.insert(0, placeholder)
        return grad_outp


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return np.where(inp > 0, inp, 0)

    def backward(self, inp, grad_outp):
        return Layer.backward(self, inp, grad_outp)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return 1 / (1 + np.e ** (-inp))

    def backward(self, inp, grad_outp):
        return Layer.backward(self, inp, grad_outp)


class TanH(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return ((np.e ** inp) - (np.e ** (-inp))) / ((np.e ** inp) + (np.e ** (-inp)))

    def backward(self, inp, grad_outp):
        return Layer.backward(self, inp, grad_outp)


class Softmax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return np.e ** inp / np.sum(np.e ** inp, axis=0)

    def backward(self, inp, grad_outp):
        return Layer.backward(self, inp, grad_outp)


class Dense(Layer):
    def __init__(self, inp_units, outp_units, learning_rate=0.01):
        super().__init__()
        self.weights = np.random.random((inp_units, outp_units)) * 2 - 1
        self.bias = np.zeros(outp_units)
        self.lr = learning_rate

    def forward(self, inp):
        return np.matmul(inp, self.weights) + self.bias

    def backward(self, inp, grad_outp):
        return Layer.backward(self, inp, grad_outp)

    def backward_by_weight(self, inp):
        weights = cp.deepcopy(self.weights)
        weights = np.transpose(weights)
        inp = np.transpose(inp)

        self.weights = inp
        original_bias = cp.deepcopy(self.bias)
        self.bias = np.array([[x] for x in self.bias])

        vals = Layer.backward(self, weights, [])

        self.weights = np.transpose(weights)
        self.bias = original_bias

        helper = []

        for j, weight_num in enumerate(self.weights):
            row = np.array([])
            for k, weight in enumerate(weight_num):
                row = np.append(row, np.sum(vals[0][k][j][k]) / len(vals[0][k][j][k]))
            helper.append(row)

        return helper


class MLP:
    def __init__(self):
        self.layers = []
        self.classes_count = 0

    def set_classes_count(self, classes_count):
        self.classes_count = classes_count

    def add_layer(self, neuron_count, inp_shape=None, activation='sigmoid'):
        if inp_shape:
            self.layers.append(Dense(inp_shape, neuron_count))
        elif self.layers:
            self.layers.append(Dense(len(self.layers[-2].bias), neuron_count))
        else:
            print('No input shape')

        self.set_classes_count(neuron_count)

        if activation == 'sigmoid':
            self.layers.append(Sigmoid())
        elif activation == 'relu':
            self.layers.append(ReLU())
        elif activation == 'tanh':
            self.layers.append(TanH())
        elif activation == 'softmax':
            self.layers.append(Softmax())
        else:
            print('Wrong activation function')

    def forward(self, inputs):
        activations = []

        for layer in self.layers:
            inputs = layer.forward(inputs)
            activations.append(inputs)
            # print(X)

        return activations

    def predict(self, inputs):
        return np.argmax(self.forward(inputs)[-1], axis=-1)

    def get_weights_and_activ_functions(self, iteration, model_loss, test_acc, train_acc):
        model_data = dict()
        model_data.update({'iter': iteration})
        model_data.update({'loss': model_loss})
        model_data.update({'test_accuracy': test_acc})
        model_data.update({'train_accuracy': train_acc})

        for index, layer in enumerate(self.layers):
            key = str(index)
            model_data.update({key: dict()})
            if isinstance(layer, Dense):
                model_data[key].update({'bias': layer.bias})
                model_data[key].update({'weights': layer.weights})
            else:
                model_data[key].update({'activ': type(layer)})

        return model_data

    def fit(self, batch_x, batch_y):
        values = self.forward(batch_x)
        values.insert(0, batch_x)

        gradients = []
        one_hot_encoded = np.zeros((batch_y.size, self.classes_count))
        one_hot_encoded[np.arange(batch_y.size), batch_y] = 1

        total = np.sum(values[-1] - one_hot_encoded, axis=0) / len(one_hot_encoded)
        gradients.append([[x] for x in total])

        for idx, layer in enumerate(self.layers[::-1]):
            usable_index = len(self.layers) - idx - 1

            gradients = layer.backward(values[usable_index], gradients)

            if isinstance(layer, Dense):
                gradients[0] = (np.sum(gradients[0], axis=0) / len(gradients[0])) * gradients[1]
                weight_diffs = layer.backward_by_weight(values[usable_index])

                layer.weights = layer.weights - (layer.lr * gradients[1] * weight_diffs)
                layer.bias = layer.bias - (layer.lr * gradients[1])
            else:
                gradients[0] = np.matmul(np.sum(gradients[0], axis=0)/len(gradients[0]), np.sum(gradients[1], axis=-1))

        return gradients


def train_test_split(total_data, y_total):
    zipped = list(zip(total_data, y_total))

    percentage = 0.7
    train_zip = zipped[:int(len(zipped) * percentage)]
    test_zip = zipped[int(len(zipped) * percentage):]

    train_x, train_y = zip(*train_zip)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x, test_y = zip(*test_zip)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    dFrame = pd.read_csv("../iris.data", header=None)
    # dFrame = pd.read_csv("../assignment3/heart.csv", header=0)
    dFrame = dFrame.sample(frac=1)
    y_values = dFrame.iloc[:, len(dFrame.columns) - 1].values
    possible_classes = list(set(y_values))
    y = np.array([possible_classes.index(x) for x in y_values])

    X = dFrame.iloc[:, 0:len(dFrame.columns) - 1]
    X = (X - X.min()) / (X.max() - X.min())
    data = np.array(X.values)

    train_data, train_answers, test_data, test_answers = train_test_split(data, y)

    network = MLP()

    BATCH_SIZE = 10
    network.add_layer(4, len(X.columns), 'relu')
    network.add_layer(6, activation='relu')
    network.add_layer(len(possible_classes), activation='softmax')

    old_loss = 0
    loss_counter = 0
    iters = 10000
    loss_check = 0
    model = dict()

    for i in range(iters):
        split_data = np.array_split(train_data, BATCH_SIZE)
        split_correct = np.array_split(train_answers, BATCH_SIZE)
        # grads = network.fit(train_data, train_answers)
        for data_batch, answers_batch in zip(split_data, split_correct):
            grads = network.fit(data_batch, answers_batch)

        test_predictions = network.predict(test_data)
        train_predictions = network.predict(train_data)

        forward = network.forward(test_data)

        one_hot = np.zeros((test_answers.size, len(possible_classes)))
        one_hot[np.arange(test_answers.size), test_answers] = 1

        loss = np.sum(1 / 2 * ((forward[-1] - one_hot) ** 2)) / len(test_answers)

        test_accuracy = (test_answers == test_predictions).sum().item() / len(test_answers)
        train_accuracy = (train_answers == train_predictions).sum().item() / len(train_answers)

        print(f'Iteration: {i}, Test loss: {loss}, Test accuracy: {test_accuracy}, Train accuracy: {train_accuracy}')

        if loss_counter > 0:
            loss_counter += 1
        elif loss > old_loss:
            loss_check = old_loss
            loss_counter += 1

        if loss < loss_check or loss_check == 0:
            loss_check = 0
            loss_counter = 0

        if loss <= old_loss and loss_counter == 0:
            model = network.get_weights_and_activ_functions(i, loss, test_accuracy, train_accuracy)

        old_loss = loss

        if loss_counter > 20 or i + 1 == iters:
            print('Ending learning curve')
            print(f'model: {model}')
            break

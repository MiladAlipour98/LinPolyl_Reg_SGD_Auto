from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def append_bias(X): return np.append(X, np.ones((X.shape[0], 1)), axis=1)


def hypothesis(w, X): return X @ w


def load_auto(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["name"])
    ys = df.iloc[:, 0]
    xs = df.iloc[:, 1:]
    return xs.to_numpy(), ys.to_numpy().reshape(-1, 1)


def split_ds(xs, ys, batch_size):
    x_batches = [xs[i: i + batch_size] for i in range(0, len(xs), batch_size)]
    y_batches = [ys[i: i + batch_size] for i in range(0, len(ys), batch_size)]
    return x_batches, y_batches


def initialize_with_zeros(n_features): return np.zeros((n_features + 1, 1))


def shuffle_ds(xs, ys):
    ds = list(zip(xs, ys))
    shuffle(ds)
    xs, ys = list(zip(*ds))
    return np.array(xs), np.array(ys)


def propagate(w, X, Y):
    X = append_bias(X)
    n_features = X.shape[1]

    A = hypothesis(w, X)
    cost = (1 / (2 * n_features)) * np.sum((A - Y) ** 2)
    dw = (1 / n_features) * (X.T @ (A - Y))

    return dw, cost


def mse(preds, Y): return 1 / (len(Y)) * np.sum((preds - Y) ** 2)


def main():
    xs, ys = load_auto("Auto.csv")
    n_features = xs.shape[1]
    xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2)
    batch_size = 32
    epochs = 10000
    lr = 5e-10
    costs = []
    w = initialize_with_zeros(n_features)
    for i in range(epochs):
        xs_train, ys_train = shuffle_ds(xs_train, ys_train)
        xs_train_batch, ys_train_batch = split_ds(xs_train, ys_train, batch_size)
        for x, y in zip(xs_train_batch, ys_train_batch):
            dw, cost = propagate(w, x, y)
            w -= lr * dw
            costs.append(cost)

    print("train mse:", mse(hypothesis(w, append_bias(xs_train)), ys_train))
    print("test mse:", mse(hypothesis(w, append_bias(xs_test)), ys_test))


if __name__ == '__main__':
    main()

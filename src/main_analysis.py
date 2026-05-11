import pandas as pd
import numpy as np

data = pd.read_csv("../data/digits.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 1. Controlla quante immagini sono presenti e quanti pixel (64)
n_campioni, n_pixel = X.shape
print(f"Il dataset contiene {n_campioni} immagini, ognuna composta da {n_pixel} pixel.")

# 2. Controlla quali numeri sono presenti (0, 1 ... 9)
classi = np.unique(y)
print(f"Le classi target sono: {classi}")

# 3. Controlla se il dataset è bilanciato
print("Distribuzione delle classi:")
print(pd.Series(y).value_counts().sort_index())

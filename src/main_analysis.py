import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../data/digits.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Controlla quante immagini sono presenti e quanti pixel (64)
n_campioni, n_pixel = X.shape
print(f"Il dataset contiene {n_campioni} immagini, ognuna composta da {n_pixel} pixel.")

# Controlla quali numeri sono presenti (0, 1 ... 9)
classi = np.unique(y)
print(f"Le classi target sono: {classi}")

# Controlla se il dataset è bilanciato
print("Distribuzione delle classi:")
print(pd.Series(y).value_counts().sort_index())


index = 121

# Ripristina la struttura bidimensionale 8x8
sample_image = X[index].reshape(8,8)

# Crea il grafico
plt.figure(figsize=(4,4))
plt.imshow(sample_image, cmap='gray_r')
plt.title(f"Etichetta reale: {y[index]}")
plt.axis('off')

# Salva l'immagine
plt.savefig('../output/digit_sample.png', bbox_inches='tight')
#plt.show()

print(f"Visualizzazione completata. L'immagine rappresenta un: {y[index]}")

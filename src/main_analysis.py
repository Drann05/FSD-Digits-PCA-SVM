import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# 1. Creiamo il modello PCA dicendogli di ridurre i dati a 2 dimensioni
pca = PCA(n_components=2)

# 2. Addestriamo la PCA sui nostri pixel (X) e trasformiamo i dati
X_pca = pca.fit_transform(X)

print("\n--- RISULTATI PCA ---")
print(f"Forma dei dati originali: {X.shape} (64 dimensioni)")
print(f"Forma dei dati dopo la PCA: {X_pca.shape} (2 dimensioni)")

# 3. Calcoliamo la Varianza Spiegata
variance = pca.explained_variance_ratio_
print(f"Varianza spiegata dalla Componente 1: {variance[0] * 100:.2f}%")
print(f"Varianza spiegata dalla Componente 2: {variance[1] * 100:.2f}%")
print(f"Varianza totale mantenuta: {sum(variance) * 100:.2f}%")

# Creazione di una figura per vedere i punti
plt.figure(figsize=(10, 8))

# Disegniamo i punti (Scatter Plot)
# X_pca[:, 0] prende tutte le righe della prima colonna (Asse X = PC1)
# X_pca[:, 1] prende tutte le righe della seconda colonna (Asse Y = PC2)
# c=y colora i punti in base alla loro etichetta vera (da 0 a 9)
# cmap='tab10' usa una tavolozza di 10 colori ben distinti
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7, edgecolors='none', s=20)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Proiezione PCA del Dataset Digits')

# Aggiungiamo la legenda laterale (Colorbar)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label('Classi (Numeri da 0 a 9)')

# Salviamo il grafico
plt.savefig('../output/pca_scatterplot.png', dpi=300, bbox_inches='tight')

# Mostriamo il grafico a schermo
# plt.show()

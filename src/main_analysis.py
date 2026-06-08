import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("../data/digits.csv", header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Controlla quante immagini sono presenti e quanti pixel (64)
n_campioni, n_pixel = X.shape
print(f"Il dataset contiene {n_campioni} immagini, ognuna composta da {n_pixel} pixel.")

# Controlla quali numeri sono presenti (0, 1 ... 9)
classi = np.unique(y)
print(f"\nLe classi target sono: {classi}")

# Controlla se il dataset è bilanciato
print(f"\nDistribuzione delle classi:")
print(pd.Series(y).value_counts().sort_index())

index = 121

# Ripristina la struttura bidimensionale 8x8
sample_image = X[index].reshape(8, 8)

# Crea il grafico
plt.figure(figsize=(4, 4))
plt.imshow(sample_image, cmap="gray_r")
plt.title(f"Etichetta reale: {y[index]}")
plt.axis("off")

# Salva l'immagine
plt.savefig("../output/digit_sample.png", bbox_inches="tight")

print(f"\nVisualizzazione completata. L'immagine rappresenta un: {y[index]}")

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

# 4. Dividiamo i dati proiettati in training e test set (80% train, 20% test)

# Parametri per lo split
random_state = 42
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=test_size, random_state=random_state, stratify=y
)

print("\n--- SPLIT TRAIN/TEST ---")
print(
    f"Campioni per Training: {X_train.shape[0]} ({X_train.shape[0] / len(y) * 100:.1f}%)"
)
print(f"Campioni per Test: {X_test.shape[0]} ({X_test.shape[0] / len(y) * 100:.1f}%)")

# Holdout estimation - Verifica distribuzione classi (per controllare la stratificazione)
train_counts = pd.Series(y_train).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()
print(f"\nDistribuzione classi nel training set:")
print(train_counts.to_string())
print(f"\nDistribuzione classi nel test set:")
print(test_counts.to_string())

# Creazione di una figura per vedere i punti
plt.figure(figsize=(10, 8))

# Disegniamo i punti (Scatter Plot)
# X_pca[:, 0] prende tutte le righe della prima colonna (Asse X = PC1)
# X_pca[:, 1] prende tutte le righe della seconda colonna (Asse Y = PC2)
# c=y colora i punti in base alla loro etichetta vera (da 0 a 9)
# cmap='tab10' usa una tavolozza di 10 colori ben distinti
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.7, edgecolors="none", s=20
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Proiezione PCA del Dataset Digits")

# Aggiungiamo la legenda laterale (Colorbar)
cbar = plt.colorbar(scatter, ticks=range(10))
cbar.set_label("Classi (Numeri da 0 a 9)")

# Salviamo il grafico
plt.savefig("../output/pca_scatterplot.png", dpi=300, bbox_inches="tight")

# 5. Inizializzazione e Addestramento del modello SVM
print("\n--- ADDESTRAMENTO SVM ---")

# Inizializziamo il classificatore SVC con kernel RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state)
print(f"Configurazione modello:")
print(f" - Kernel utilizzato: {svm_model.kernel}")
print(f" - Parametro C: {svm_model.C}")
print(f" - Training set: {X_train.shape[0]} campioni, {X_train.shape[1]} feature (componenti PCA)")

# Addestriamo il modello sulle coordinate ottenute dalla PCA
svm_model.fit(X_train, y_train)
print(f"\nModello SVM addestrato con successo.")

# Predizione sul set di test
y_pred = svm_model.predict(X_test)

# Valutazione delle performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRisultati SVM:")
print(f"Accuratezza del modello: {accuracy * 100:.2f}%")
print("\nReport di classificazione dettagliato:")
print(classification_report(y_test, y_pred))

# 6. Risultati Finali (matrice di confusione e metriche di accuratezza)

# Matrice di confusione
print("\nGenerazione della Matrice di Confusione...")

# Calcolo della matrice
cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)

# Plot della matrice
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
# Usiamo una mappa di colori blu per una visualizzazione pulita
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())

plt.show()
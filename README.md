# RNN Language Model From Scratch

Implémentation pédagogique d'un **modèle de langage basé sur un RNN (Recurrent Neural Network)** en NumPy pur, sans utiliser de frameworks de deep learning.

Ce projet démontre les concepts fondamentaux des réseaux de neurones récurrents : propagation temporelle, backpropagation through time (BPTT), et prédiction de séquences.

## Caractéristiques

- **Implémentation from scratch** : Toute la logique du RNN codée en NumPy
- **Backpropagation Through Time (BPTT)** : Calcul complet des gradients à travers le temps
- **Prédiction de mots** : Génération de suggestions basées sur le contexte
- **Architecture personnalisable** : Hidden state size configurable
- **Code commenté** : Explications détaillées de chaque étape mathématique

## Architecture du RNN

Le modèle utilise une architecture RNN classique :

```
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b_h)
y_t = softmax(W_y @ h_t + b_y)
```

**Composants** :
- `W_x` : Matrice de poids input → hidden state
- `W_h` : Matrice de poids hidden state → hidden state (mémoire temporelle)
- `W_y` : Matrice de poids hidden state → output (prédictions)
- `b_h`, `b_y` : Biais

## Installation

```bash
# Cloner le repo
git clone https://github.com/VOTRE_USERNAME/rnn-language-model-from-scratch.git
cd rnn-language-model-from-scratch

# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installer NumPy
pip install numpy jupyter
```

## Utilisation

### Lancer le notebook

```bash
jupyter notebook RNNLanguageModel.ipynb
```

### Exemple rapide

```python
import numpy as np
from RNNLanguageModel import RNNLanguageModel

# Créer le modèle
model = RNNLanguageModel(hidden_size=64)

# Corpus d'entraînement
texts = [
    "le chat mange du poisson frais",
    "le chien court rapidement",
    "je suis très content de te voir"
]

# Construire le vocabulaire
model.build_vocabulary(texts)

# Entraîner le modèle
learning_rate = 0.05
epochs = 300

for epoch in range(epochs):
    total_loss = 0
    for sentence in texts:
        loss = model.train_on_sentence(sentence, learning_rate)
        total_loss += loss

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss/len(texts):.4f}")

# Prédire le mot suivant
model.predict_next_word("le chat mange du poisson", top_k=5)
```

**Résultat** :
```
Après 'le chat mange du poisson', top 5 prédictions :
  1. frais           -> 0.9969
  2. est             -> 0.0010
  3. canapé          -> 0.0005
  ...
```

## Contenu du projet

| Fichier | Description |
|---------|-------------|
| [`RNNLanguageModel.ipynb`](RNNLanguageModel.ipynb) | Notebook principal avec implémentation et démonstrations |
| `LSTMLanguageModel.ipynb` | Extension avec architecture LSTM (optionnel) |
| `README.md` | Ce fichier |

## Concepts implémentés

### Forward Pass
Calcul séquentiel des hidden states pour chaque mot d'une phrase.

### Backpropagation Through Time (BPTT)
- Calcul des gradients en remontant dans le temps
- Gradient clipping pour éviter l'explosion des gradients
- Mise à jour des poids par descente de gradient

### Prédiction de mots
- Conversion des mots en vecteurs one-hot
- Calcul des probabilités avec softmax
- Affichage des top-k prédictions

## Résultats

Le modèle atteint de bonnes performances sur un petit corpus :

| Métrique | Valeur |
|----------|--------|
| Loss finale | ~2.55 |
| Hidden size | 64 |
| Epochs | 300 |
| Learning rate | 0.05 |
| Précision top-1 | >99% sur phrases du corpus |

**Exemples de prédictions** :
- `"le chat"` → `"et"` (39.4%), `"joue"` (24.2%), `"dort"` (19.6%)
- `"le joueur court très"` → `"vite"` (99.7%)
- `"je suis très content de"` → `"te"` (99.7%)

## Limitations

- **Vocabulaire limité** : Fonctionne uniquement avec les mots du corpus
- **Pas de généralisation** : Modèle basique sans regularization
- **Gradient vanishing** : Sur de très longues séquences, les gradients peuvent s'effacer
- **NumPy uniquement** : Performance limitée comparé à des frameworks optimisés

Pour dépasser ces limitations, voir l'implémentation LSTM dans [`LSTMLanguageModel.ipynb`](LSTMLanguageModel.ipynb).

## Concepts clés pour apprendre

Ce projet est idéal pour comprendre :

1. **Traitement de séquences** : Comment les RNN maintiennent une mémoire temporelle
2. **BPTT** : Backpropagation dans les réseaux récurrents
3. **Gradient clipping** : Technique pour stabiliser l'entraînement
4. **One-hot encoding** : Représentation des mots comme vecteurs
5. **Softmax & Cross-Entropy** : Fonctions de perte pour la classification

## Ressources

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy
- [Deep Learning Book - Chapter 10: Sequence Modeling](https://www.deeplearningbook.org/contents/rnn.html)

## Auteur

**Guillaume** - Formation Introduction au Deep Learning

## Licence

Ce projet est à usage éducatif. Libre d'utilisation pour l'apprentissage.

---

**Note** : Ce projet est conçu dans un but pédagogique. Pour des applications en production, utilisez des frameworks comme PyTorch ou TensorFlow avec des architectures modernes (Transformers, BERT, GPT).

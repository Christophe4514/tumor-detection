# Segmentation floue d'IRM cérébrales par Fuzzy C-Means (FCM)

Projet académique de **Master 2 - Logique Floue**:

> **Segmentation floue d'images médicales par l'algorithme Fuzzy C-Means : Application à la détection de tumeurs cérébrales**

Ce projet implémente **manuellement** l'algorithme **Fuzzy C-Means (FCM)** en respectant les contraintes suivantes:
- Pas de bibliothèque FCM prête à l'emploi.
- Calculs matriciels avec **NumPy**.
- Traitement d'image avec **OpenCV**.
- Interface graphique avec **Tkinter**.

---

## 1) Arborescence du projet

```text
brain_tumor_fcm/
├── main.py
├── README.md
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── fuzzy_c_means.py
│   ├── feature_extraction.py
│   ├── fuzzy_classifier.py
│   ├── dataset_calibration.py
│   ├── segmentation.py
│   ├── tumor_detection.py
│   └── tumor_classification.py
├── ui/
│   ├── __init__.py
│   └── interface.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── metrics.py
│   └── evaluation.py
├── datasets/
└── results/
```

---

## 2) Rappel théorique FCM

### Fonction objectif

\[
J_m(U, V) = \sum_{i=1}^{N}\sum_{j=1}^{C}u_{ij}^{m}\,\|x_i - v_j\|^2
\]

- \(N\): nombre de pixels,
- \(C\): nombre de clusters,
- \(u_{ij}\): degré d'appartenance du pixel \(x_i\) au cluster \(j\),
- \(v_j\): centre du cluster \(j\),
- \(m>1\): coefficient de flou (ici \(m=2\) par défaut).

### Mise à jour des centres

\[
v_j = \frac{\sum_{i=1}^{N} u_{ij}^{m} x_i}{\sum_{i=1}^{N} u_{ij}^{m}}
\]

### Mise à jour des appartenances

\[
u_{ij} = \frac{1}{\sum_{k=1}^{C}\left(\frac{\|x_i-v_j\|}{\|x_i-v_k\|}\right)^{\frac{2}{m-1}}}
\]

### Convergence

On arrête quand:
\[
\|U^{(t+1)} - U^{(t)}\| < \epsilon
\]
ou quand le nombre maximal d'itérations est atteint.

---

## 3) Fonctionnalités implémentées

1. Chargement d'image IRM (`png`, `jpg`, `jpeg`).
2. Prétraitement:
   - conversion en niveaux de gris,
   - filtre médian,
   - normalisation dans \([0,1]\).
3. Segmentation FCM complète:
   - initialisation aléatoire de \(U\),
   - mise à jour itérative de \(V\) et \(U\),
   - calcul de \(J_m\) à chaque itération,
   - affichage de l'évolution du coût.
4. Segmentation finale par `argmax` des appartenances.
5. Détection automatique de la tumeur:
   - cluster de plus forte intensité moyenne,
   - masque binaire tumoral.
6. Classification affichée après détection:
   - `glioma`
   - `meningioma`
   - `pituitary tumor`
   - `no tumor`
   - (classification basée sur des règles floues multi-critères: aire, position, circularité, intensité)
   - (résultat pédagogique non diagnostique: ne pas utiliser en contexte clinique réel)
7. Calibration supervisée du classifieur flou:
   - extraction de features sur dataset annoté,
   - apprentissage des prototypes flous (mu/sigma) par classe,
   - évaluation train/test avec accuracy et matrice de confusion,
   - entraînement direct depuis `datasets/segmentation`,
   - chargement automatique du modèle calibré dans l'interface.
8. Post-traitement morphologique:
   - érosion,
   - dilatation,
   - ouverture,
   - fermeture.
9. Visualisation:
   - originale,
   - gris,
   - prétraitée,
   - segmentée,
   - masque tumoral,
   - tumeur extraite.
10. Interface Tkinter:
   - boutons demandés,
   - chargement multi-images (lot),
   - navigation image précédente/suivante,
   - zones d'affichage,
   - barre de progression,
   - métriques d'évaluation.
11. Évaluation:
   - nombre d'itérations,
   - temps d'exécution,
   - valeur finale de \(J_m\),
   - état de convergence.

---

## 4) Installation

### Prérequis
- Python 3.10+ recommandé
- pip

### Étapes

```bash
python -m venv .venv
```

Sous Windows:
```bash
.venv\Scripts\activate
```

Sous Linux/Mac:
```bash
source .venv/bin/activate
```

Installer les dépendances:

```bash
pip install -r requirements.txt
```

---

## 5) Exécution de l'application

Depuis la racine du projet:

```bash
python main.py
```

Workflow recommandé dans l'interface:
1. **Charger image(s)** (sélection une ou plusieurs IRM)
2. Si lot: utiliser **Image précédente / Image suivante**
3. Calibration automatique au démarrage depuis `datasets/segmentation` (si disponible)
4. Optionnel: **Calibrer depuis datasets/segmentation** pour réentraîner manuellement
5. **Prétraiter**
6. **Segmenter (FCM)**
7. **Détecter tumeur** (utilise le modèle calibré s'il existe)
8. **Sauvegarder résultat**

---

## 6) Utilisation du dataset Kaggle

Le dataset peut être récupéré avec `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("indk214/brain-tumor-dataset-segmentation-and-classification")
print("Path to dataset files:", path)
```

Copier ensuite les images d'intérêt dans `datasets/` ou charger directement depuis leur emplacement via l'interface.

---

## 7) Notes pédagogiques pour soutenance

- Le FCM est une **segmentation floue**: un pixel peut appartenir partiellement à plusieurs classes.
- L'affectation finale (segmentation "dure") est faite seulement après convergence via `argmax(u_ij)`.
- Le choix du cluster tumoral par intensité maximale est une hypothèse simple et interprétable.
- Les opérations morphologiques améliorent la qualité spatiale du masque (réduction du bruit et comblement de trous).

---

## 8) Sorties générées

Lors de la sauvegarde, l'application crée un dossier dans `results/` contenant:
- images intermédiaires (`original`, `gray`, `preprocessed`, `segmented`),
- masques de clusters,
- masque tumoral initial et raffiné,
- tumeur extraite,
- fichier `metrics.txt`.

---

## 9) Auteurs / contexte

Projet développé pour une présentation académique en **Logique Floue (M2)**, avec une implémentation explicite et commentée des équations FCM pour faciliter la défense orale devant un enseignant.

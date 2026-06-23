# 🏦 Bank Customer Churn Prediction

Projet de **Data Science** visant à prédire le départ potentiel de clients bancaires à partir de données clients.

L’objectif est d’identifier les clients qui risquent de quitter la banque afin de permettre la mise en place d’actions de fidélisation ciblées.

Ce projet met en avant un workflow complet de Machine Learning : analyse exploratoire, préparation des données, entraînement de modèles, évaluation, interprétation et recommandations métier.

---

## 🎯 Objectif du projet

Le churn client est un enjeu important pour les banques. Perdre un client peut coûter cher, surtout lorsqu’il s’agit d’un client actif, solvable ou détenteur de plusieurs produits bancaires.

Ce projet cherche à répondre à la question suivante :

> Peut-on prédire quels clients sont susceptibles de quitter la banque à partir de leurs caractéristiques personnelles, financières et comportementales ?

L’objectif final est de construire un modèle capable d’aider une banque à détecter les clients à risque et à agir avant leur départ.

---

## 🧠 Problématique Data Science

Il s’agit d’un problème de **classification binaire**.

La variable cible indique si un client a quitté la banque ou non :

| Valeur | Signification |
|---|---|
| `0` | Client conservé |
| `1` | Client ayant quitté la banque |

Le modèle doit apprendre à distinguer les clients stables des clients à risque à partir de plusieurs variables comme l’âge, le solde bancaire, le nombre de produits, le pays, le score de crédit ou encore l’activité du client.

---

## 📊 Données utilisées

Le dataset contient des informations clients bancaires.

Exemples de variables possibles :

| Variable | Description |
|---|---|
| `CreditScore` | Score de crédit du client |
| `Geography` | Pays du client |
| `Gender` | Genre du client |
| `Age` | Âge du client |
| `Tenure` | Ancienneté du client |
| `Balance` | Solde bancaire |
| `NumOfProducts` | Nombre de produits bancaires détenus |
| `HasCrCard` | Possession d’une carte bancaire |
| `IsActiveMember` | Client actif ou non |
| `EstimatedSalary` | Salaire estimé |
| `Exited` | Variable cible : client parti ou non |

---

## 🛠️ Technologies utilisées

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- Jupyter Notebook
- Git / GitHub

---

## 📁 Structure du projet

```txt
bank-customer-churn-analysis/
│
├── data/
│   ├── raw/
│   │   └── churn_data.csv
│   │
│   └── processed/
│       └── churn_clean.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretability.ipynb
│
├── reports/
│   ├── figures/
│   └── churn_analysis_report.pdf
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🔎 Méthodologie

### 1. Compréhension du problème métier

La première étape consiste à comprendre pourquoi le churn est important pour une banque.

Les objectifs métier sont :

- identifier les clients à risque ;
- comprendre les facteurs qui influencent le départ ;
- aider les équipes commerciales à prioriser leurs actions ;
- réduire la perte de clients ;
- améliorer la fidélisation.

---

### 2. Analyse exploratoire des données

L’analyse exploratoire permet de mieux comprendre le dataset.

Analyses réalisées :

- taille du dataset ;
- types de variables ;
- valeurs manquantes ;
- doublons ;
- distribution de la variable cible ;
- analyse de l’âge ;
- analyse du solde bancaire ;
- analyse par pays ;
- analyse par genre ;
- analyse du nombre de produits ;
- comparaison entre clients partis et clients conservés.

Exemples de questions explorées :

- Les clients âgés quittent-ils davantage la banque ?
- Le churn est-il plus élevé dans certains pays ?
- Les clients avec un solde élevé partent-ils plus souvent ?
- Le nombre de produits influence-t-il le départ ?
- Les clients inactifs ont-ils plus de chances de partir ?

---

### 3. Préparation des données

Avant l’entraînement des modèles, les données doivent être préparées.

Étapes réalisées :

- suppression des colonnes inutiles ;
- gestion des valeurs manquantes ;
- encodage des variables catégorielles ;
- séparation des variables explicatives et de la cible ;
- séparation train/test ;
- standardisation des variables numériques si nécessaire.

Exemples de transformations :

| Type de variable | Transformation |
|---|---|
| Variables numériques | Standardisation |
| Variables catégorielles | Encodage |
| Variable cible | Conservation en binaire |
| Colonnes inutiles | Suppression |

---

### 4. Modélisation

Plusieurs modèles peuvent être testés afin de comparer leurs performances.

Modèles utilisés ou prévus :

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost ou LightGBM, amélioration future

L’objectif n’est pas seulement d’avoir un bon score global, mais aussi de bien identifier les clients réellement à risque.

---

### 5. Évaluation des modèles

Les modèles sont évalués avec plusieurs métriques.

| Métrique | Rôle |
|---|---|
| Accuracy | Mesure globale de bonnes prédictions |
| Precision | Fiabilité des clients prédits comme churn |
| Recall | Capacité à détecter les vrais clients qui quittent |
| F1-score | Équilibre entre precision et recall |
| ROC-AUC | Capacité du modèle à séparer les deux classes |
| Matrice de confusion | Visualisation des erreurs du modèle |

Dans ce projet, le **recall** est important car il est préférable d’identifier un maximum de clients à risque, même si cela génère quelques faux positifs.

---

## 📊 Résultats des modèles

À compléter avec tes vrais résultats après entraînement.

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | À compléter | À compléter | À compléter | À compléter | À compléter |
| Random Forest | À compléter | À compléter | À compléter | À compléter | À compléter |
| Gradient Boosting | À compléter | À compléter | À compléter | À compléter | À compléter |

---

## 📌 Matrice de confusion

À compléter avec une image dans le dossier `reports/figures/`.

Exemple :

```txt
reports/figures/confusion_matrix.png
```

Ajouter ensuite dans le README :

```md
![Matrice de confusion](reports/figures/confusion_matrix.png)
```

---

## 🔍 Interprétation du modèle

L’interprétation du modèle permet de comprendre pourquoi certains clients sont considérés comme à risque.

Méthodes possibles :

- importance des variables ;
- SHAP values ;
- analyse des erreurs ;
- comparaison des profils clients.

Variables qui peuvent influencer le churn :

- âge ;
- activité du client ;
- nombre de produits ;
- solde bancaire ;
- pays ;
- score de crédit ;
- ancienneté.

---

## 💼 Recommandations métier

À partir des résultats, plusieurs actions peuvent être proposées à une banque.

### Clients à fort risque

Pour les clients ayant une probabilité de churn élevée :

- contact par un conseiller ;
- offre personnalisée ;
- réduction de frais ;
- proposition d’un produit adapté ;
- suivi prioritaire.

### Clients inactifs

Pour les clients peu actifs :

- campagne de réactivation ;
- relance commerciale ;
- proposition de services digitaux ;
- amélioration de l’accompagnement client.

### Clients avec peu de produits

Pour les clients avec un seul produit :

- proposition d’un produit complémentaire ;
- offre groupée ;
- accompagnement personnalisé.

---

## ✅ Compétences démontrées

Ce projet montre des compétences en :

- Data Science ;
- analyse exploratoire ;
- classification supervisée ;
- préparation de données ;
- Machine Learning ;
- évaluation de modèles ;
- interprétation de modèles ;
- recommandations business ;
- structuration d’un projet data.

---

## ⚙️ Installation

### 1. Cloner le repository

```bash
git clone https://github.com/Ethan941/bank-customer-churn-analysis.git
cd bank-customer-churn-analysis
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sur macOS / Linux :

```bash
source .venv/bin/activate
```

Sur Windows :

```bash
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Puis ouvrir les notebooks dans le dossier :

```txt
notebooks/
```

---

## ▶️ Utilisation

Exemple de workflow :

```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
```

Si le projet est principalement en notebook :

```txt
1. Ouvrir 01_eda.ipynb
2. Lancer l’analyse exploratoire
3. Ouvrir 02_modeling.ipynb
4. Entraîner les modèles
5. Ouvrir 03_interpretability.ipynb
6. Analyser les résultats
```

---

## 🚀 Améliorations possibles

- Ajouter un pipeline Scikit-learn complet ;
- tester XGBoost ou LightGBM ;
- ajouter une optimisation des hyperparamètres ;
- ajouter SHAP de manière plus détaillée ;
- sauvegarder le meilleur modèle avec Joblib ;
- créer une API FastAPI de prédiction ;
- créer une interface Streamlit ;
- ajouter un dashboard de suivi du churn ;
- ajouter des tests unitaires ;
- déployer le projet.

---

## 📌 Statut du projet

Projet Data Science en cours d’amélioration.

L’objectif est de construire un projet complet et professionnel autour de la prédiction du churn client, avec une approche métier et technique.

---

## 👤 Auteur

**Ethan Pandor**  
Étudiant en Bachelor Data & IA à HETIC  
Recherche stage ou alternance en Data Science / Data Engineering

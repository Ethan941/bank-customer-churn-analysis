# Bank Customer Churn Analysis

## 1. Contexte
L’objectif de ce projet est de prédire le churn des clients bancaires et
d’identifier les principaux facteurs explicatifs afin d’aider à la mise en
place de stratégies de rétention ciblées.

Le dataset contient 10 000 clients avec des informations démographiques,
financières et comportementales.

---

## 2. Approche Data
- Analyse exploratoire des données (EDA)
- Préparation des données (nettoyage, encodage, standardisation)
- Entraînement de modèles de classification :
  - Logistic Regression (baseline)
  - Random Forest
- Optimisation du seuil de décision selon les objectifs business
- Analyse d’interprétabilité du modèle

---

## 3. Performance des modèles

### Logistic Regression
- ROC-AUC : ~0.78
- Recall churn : ~70 %
- Modèle adapté à des campagnes de rétention à faible coût

### Random Forest
- ROC-AUC : ~0.85
- Précision élevée (~77 % à seuil 0.5)
- Après tuning du seuil :
  - Recall ≈ 65 %
  - Précision ≈ 57 %

👉 Le Random Forest permet de cibler plus efficacement les clients à fort
risque de churn lorsque les actions de rétention sont coûteuses.

---

## 4. Facteurs clés du churn (Random Forest)

Top variables explicatives :
1. Age (~25 %)
2. Balance (~14 %)
3. EstimatedSalary (~14 %)
4. CreditScore (~13 %)
5. NumOfProducts (~13 %)
6. Tenure
7. IsActiveMember
8. Geography (notamment Germany)

Le churn est fortement influencé par le profil client et son niveau
d’engagement avec la banque.

---

## 5. Recommandations business

### Segments à risque
- Clients âgés avec une balance élevée
- Clients peu actifs
- Clients avec peu de produits bancaires
- Clients récents
- Clients à hauts revenus

### Actions de rétention
- Offres premium pour les clients à forte valeur
- Campagnes d’activation pour les clients peu engagés
- Stratégies de cross-selling pour augmenter le multi-équipement
- Onboarding renforcé pour les nouveaux clients
- Approche spécifique par zone géographique

---

## 6. Conclusion
Ce projet montre comment un modèle de machine learning peut être utilisé
pour prédire le churn client et transformer les résultats en actions
opérationnelles concrètes pour une banque ou une assurance.

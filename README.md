# 🏥 Health-InsurTech — Application Streamlit

Application de simulation de frais de santé éthique et transparente.

## � Structure du Projet

```
health-insurtech/
├── data/
│   └── raw/
│       └── insurance_data.csv
├── src/
│   └── app.py
├── notebooks/
│   └── health_insurance_modeling.ipynb
├── models/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'application
streamlit run src/app.py
```

L'application s'ouvre sur http://localhost:8501

## 🚀 Déploiement Streamlit Cloud

1. Pousser le projet sur GitHub
2. Se connecter à [share.streamlit.io](https://healthinsure-yerp3brrojfdxdbjq9ycnu.streamlit.app/)
3. Sélectionner le repo et définir `src/app.py` comme fichier principal
4. Déployer !

---

## 🔑 Comptes de démonstration

| Identifiant | Mot de passe | Rôle     | Accès |
|-------------|-------------|----------|-------|
| `admin`     | `admin123`  | Admin    | Tout  |
| `actuary`   | `actuary123`| Actuaire | Dashboard, Simulateur, Audit biais |
| `client`    | `client123` | Client   | Simulateur uniquement |

---

## 📋 Fonctionnalités

### 🔒 Consentement RGPD
- Bannière de consentement obligatoire à l'entrée
- Données de santé non conservées (traitement éphémère)
- Conformité Art. 9 RGPD (données de santé sensibles)

### 🔐 Authentification
- 3 niveaux de rôles : Admin / Actuaire / Client
- Mots de passe hashés (SHA-256)
- Contrôle d'accès par page (RBAC)

### 🧮 Simulateur Tarifaire
- Formulaire interactif (âge, IMC, enfants, tabac, région, sexe)
- Estimation en temps réel (modèle Ridge + calibration)
- Jauge de risque animée
- Fourchette de confiance ±12%
- Indicateurs des facteurs principaux

### 📊 Dashboard Analytique
- 4 onglets : IMC vs Frais / Âge vs Frais / Par région / Distributions
- Graphiques Plotly interactifs
- Matrice de corrélation
- KPIs clés

### ⚖️ Audit des Biais
- Biais par fumeur, région, sexe
- Visualisation des distributions d'erreurs
- Zone d'acceptabilité ±15%

### 📋 Journaux d'Accès
- Logs horodatés de chaque action
- Filtrage par niveau (INFO / WARNING / ERROR)
- Export CSV

### 🛡️ Administration
- Liste des utilisateurs et rôles
- Matrice des permissions
- État du système

---

## 🏗️ Architecture

```
app.py               ← Application principale
requirements.txt     ← Dépendances Python
insurance_data.csv   ← Dataset (à placer ici)
app_logs.log         ← Généré automatiquement
```

## 🧠 Modèle

- **Ridge Regression** (alpha=10, L2) — résistant au surapprentissage
- **Calibration par groupe** — corrige le biais fumeur/non-fumeur
- **Features** : age, bmi, children, smoker, region, sex
- **PII exclues** : nom, email, numéro de sécu, adresse IP...

---

# Language Detection NLP - Application Streamlit

## A propos du projet

Ce projet propose un détecteur automatique de langue capable d'identifier 17 langues différentes à partir d'un texte saisi par l'utilisateur. Il repose sur une architecture BiLSTM (LSTM bidirectionnel) combinée à une tokenisation sous-mots (SentencePiece), entraînée sur le dataset Kaggle "Language Detection".

L'application offre une interface simple et intuitive permettant de tester le modèle en temps réel.

---

## Lancer l'application en local

### 1. Prérequis

Assurez-vous d'avoir Python 3.10 installé sur votre machine.

```bash
python --version
```

### 2. Cloner le dépôt

```bash
git clone https://github.com/kabechedihia20-ux/language-detection-nlp.git
cd language-detection-nlp
```

### 3. Créer un environnement virtuel (recommandé)

**Sous Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Sous macOS/Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Lancer l'application

```bash
streamlit run app/app.py
```

Après quelques secondes, l'application s'ouvrira automatiquement dans votre navigateur à l'adresse :
http://localhost:8501

---

## Structure des fichiers importants

```
language-detection-nlp/
├── app/
│   └── app.py                 # Application Streamlit
├── models/
│   ├── bilstm_subword_final.pt   # Poids du modèle entraîné
│   └── spm_lang_detector.model   # Tokenizer SentencePiece
├── requirements.txt           # Dépendances Python
└── README.md                  # Documentation
```

---

## Utilisation

1. Entrez un texte dans la zone de saisie
2. Cliquez sur "Prédire la langue"
3. Le modèle affiche la langue détectée avec un score de confiance

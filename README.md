# 💡 Insight Ray: Une Suite d'IA Médicale Avancée

Bienvenue au projet Insight Ray ! Il s'agit d'une suite de solutions d'intelligence artificielle conçues pour transformer l'analyse d'images médicales et l'assistance diagnostique. En tirant parti de modèles d'apprentissage profond et de techniques NLP, Insight Ray vise à améliorer la précision, la rapidité et l'efficacité des diagnostics médicaux, en offrant un soutien inestimable aux professionnels de la santé.

## 🚀 Projets Inclus

Insight Ray est composé de trois projets distincts mais complémentaires :

### 1. Détection des Anomalies sur les Radiographies Pulmonaires

Ce projet se concentre sur l'identification des diverses pathologies présentes dans les images de radiographies pulmonaires.

**Objectif :** Détecter des anomalies telles que la pneumonie, la cardiomégalie, etc., à l'aide de techniques de vision par ordinateur.

**Modèle clé :** YOLOv8 (You Only Look Once, version 8) pour une détection d'objets en temps réel et précise.

**Caractéristiques :**
- Prétraitement avancé des images (NMS, suppression des petites boîtes, sous-échantillonnage de la classe majoritaire)
- Entraînement de modèles avec transfert d'apprentissage pour des performances robustes
- Visualisation des résultats d'inférence avec des boîtes englobantes

**Documentation détaillée :** Consulter `docs/chest_xray_detection.rst` pour plus d'informations.

### 2. Détection des Fractures Osseuses

Ce projet vise à automatiser la détection et la classification des fractures osseuses à partir d'images radiographiques.

**Objectif :** Identifier les fractures dans différentes parties du corps (coude, main, épaule).

**Modèles clés :** Réseaux de Neurones Convolutifs (CNN), notamment l'architecture ResNet50.

**Approche :** Un pipeline de classification en deux étapes :
1. **Classification des parties osseuses :** Détermine si l'image contient un coude, une main ou une épaule
2. **Détection des fractures :** Pour la partie osseuse identifiée, un modèle spécifique détecte la présence d'une fracture

**Caractéristiques :**
- Utilisation du vaste ensemble de données MURA
- Processus d'entraînement distinct pour chaque phase de détection
- Sortie indiquant si une fracture est présente ou si l'os est normal

**Documentation détaillée :** Consulter `docs/bone_fracture.rst` pour plus d'informations.

### 3. Chatbot Médical

Un chatbot interactif conçu pour aider les utilisateurs à identifier des maladies potentielles en fonction de leurs symptômes, en fournissant des informations et des précautions.

**Objectif :** Fournir une assistance initiale pour le diagnostic basée sur les symptômes et offrir des conseils pertinents.

**Modèle clé :** Classifieur K-Nearest Neighbors (KNN) pour la prédiction des maladies.

**Techniques NLP :** Traitement du Langage Naturel pour comprendre les entrées utilisateur (tokenisation, lemmatisation, TF-IDF, similarité cosinus).

**Caractéristiques :**
- Interface web interactive basée sur Flask
- Gestion progressive des symptômes et affinement du diagnostic
- Évaluation de la gravité des symptômes et suggestions de précautions
- Gestion de la session pour des conversations continues

**Documentation détaillée :** Consulter `docs/chatbot.rst` pour plus d'informations.

## 🚀 Démarrage Rapide

Pour commencer avec les projets Insight Ray, suivez les étapes générales ci-dessous. Pour des instructions spécifiques à chaque projet, veuillez consulter les fichiers RST correspondants dans le répertoire `docs/`.

### Prérequis

- Python 3.7.x ou supérieur
- Git (pour cloner le dépôt)

### Installation Générale

1. **Clonez le dépôt :**
   ```bash
   git clone https://github.com/votre_utilisateur/InsightRay.git
   cd InsightRay
   ```

2. **Créez un environnement virtuel (recommandé) :**
   ```bash
   python -m venv env
   # Sur Linux/macOS
   source env/bin/activate
   # Sur Windows
   .\env\Scripts\activate
   ```

3. **Installez les dépendances :**
   
   Chaque projet aura un fichier `requirements.txt` dans son répertoire respectif. Naviguez vers le répertoire du projet qui vous intéresse et installez ses dépendances.
   
   ```bash
   # Exemple pour le Chatbot
   cd medical_chatbot/
   pip install -r requirements.txt
   ```

4. **Préparez les données et les modèles :**
   
   - Assurez-vous que les ensembles de données (MURA, radiographies pulmonaires, fichiers médicaux pour le chatbot) sont placés dans les répertoires spécifiés par chaque projet (par exemple, `medical_chatbot/Medical_dataset/`, `bone_fracture_project/Dataset/`, `chest_xray_detection_project/train/`)
   - Les modèles pré-entraînés ou les scripts d'entraînement devront être exécutés pour générer les poids des modèles (`.h5`, `.pt`, `.pkl`)

5. **Exécutez l'application ou le script :**
   
   Suivez les instructions spécifiques du fichier RST de chaque projet pour lancer l'application (par exemple, `python app.py` pour le chatbot ou la détection des fractures, ou le script d'entraînement/inférence pour la détection des radiographies pulmonaires).

## 🤝 Contribution

Nous accueillons les contributions ! Si vous souhaitez contribuer au projet Insight Ray, veuillez consulter nos `GUIDELINES.md` (à créer) pour plus de détails sur le processus de contribution.

## 📄 Licence

Ce projet est sous licence [Nom de la Licence, par exemple MIT License]. Voir le fichier `LICENSE` (à créer) pour plus de détails.

## ✉️ Contact

Pour toute question ou demande de renseignements, veuillez ouvrir une "Issue" sur ce dépôt GitHub ou contacter [Votre Nom/Email/Organisation].

---

**Clause de non-responsabilité :** Insight Ray est un outil d'aide au diagnostic et ne remplace pas l'expertise clinique professionnelle. Les résultats générés par l'IA doivent toujours être interprétés et confirmés par des professionnels de la santé qualifiés.

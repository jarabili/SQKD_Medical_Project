# SQKD Quantum Security - Physique Médicale

Ce projet est une implémentation complète du protocole de distribution quantique de clé semi-quantique (SQKD), Il intègre un modèle de QML pour sécuriser le transfert de données médicales critiques (dosimétrie radiothérapie, IRM) vers des infrastructures cloud..,

## 🎯 Objectif Scientifique
Les algorithmes cryptographiques actuels (RSA) seront vulnérables face aux futurs calculateurs quantiques. La cryptographie quantique résout ce problème mais le protocole classique (BB84) exige des équipements onéreux. Le protocole **SQKD** propose une asymétrie : un centre hospitalier n'a besoin que d'un matériel optique classique (miroir), réduisant drastiquement les coûts de déploiement..,

Ce projet démontre que l'utilisation d'un Classifieur Variationnel Quantique (VQC) permet de distinguer efficacement le bruit thermique naturel d'une fibre optique d'une cyberattaque furtive..,

## 🛠️ Architecture Technologique
1. **Physique & Simulation (`qiskit`) :** Modélisation d'un canal optique bruité (dépolarisation) et de l'attaque "Intercept-Resend".
2. **Intelligence Artificielle (`pennylane`) :** Réseau de neurones quantiques analysant les anomalies du QBER et du décalage de gain de la clé.
3. **Infrastructure Web (`fastapi` / `uvicorn`) :** Encapsulation de la logique physique dans une API REST.
4. **Automatisation SaaS (`n8n` / `Telegram`) :** Orchestration décisionnelle simulant une alerte de sécurité en temps réel.

## 🚀 Installation & Déploiement
```bash
# 1. Cloner le dépôt et installer les dépendances
pip install -r requirements.txt

# 2. Générer le jeu de données optique
python generate_sqkd_data.py

# 3. Entraîner le modèle QML
python train.py

# 4. Lancer le serveur d'inférence
uvicorn app:app --host 0.0.0.0 --port 8000
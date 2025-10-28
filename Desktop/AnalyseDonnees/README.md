# Application de recommandation de vêtements (K-Means) avec interface PyQt5

Cette application permet de recommander des vêtements en utilisant l’algorithme de clustering K-Means et une interface graphique interactive développée avec PyQt5.
Elle illustre le pré-traitement des données, la normalisation, la réduction dimensionnelle avec PCA, et la génération de recommandations personnalisées selon les préférences utilisateur.

## Objectifs

- Illustrer l’utilisation de K-Means pour regrouper des vêtements selon leurs caractéristiques.

- Montrer comment PyQt5 peut servir d’interface utilisateur interactive.

- Démontrer la compréhension du pré-traitement, de la normalisation, de la réduction de dimensions (PCA) et du calcul des recommandations.

## Prérequis

Installer les packages Python nécessaires :

pip install pandas scikit-learn matplotlib pyqt5 openpyxl

## Usage

Cloner le dépôt ou télécharger le fichier app-recommandation-vetements.py

Lancer l’application :

python .\app-recommandation-vetements.py

# Structure de l’application

RecommenderApp : classe principale gérant l’interface et la logique.

Colonnes gauche : paramètres, préférences utilisateur, boutons.

Colonnes droite : tableau des recommandations et graphique PCA des clusters.

<img width="498" height="364" alt="image" src="https://github.com/user-attachments/assets/a09bd188-5df0-4d25-9964-48f5c38e71af" />


## Dans l’interface :

1- Charger un fichier Excel contenant vos données de vêtements.

2- Sélectionner les colonnes numériques pour le clustering.

3- Définir les paramètres K-Means (nombre de clusters et itérations max).

4- Saisir vos préférences utilisateur pour chaque caractéristique.

5- Cliquer sur Recommander pour afficher les vêtements les plus proches de vos préférences.

6- Visualiser les clusters et les recommandations sur le graphique PCA 2D.


## Fonctionnalités principales :

load_excel() : charger un fichier Excel et détecter les colonnes numériques.

<img width="752" height="406" alt="image" src="https://github.com/user-attachments/assets/e6c2d9ab-1c14-4f58-8e28-7055ee36dff3" />


run_kmeans() : exécuter K-Means sur les données normalisées.

<img width="594" height="410" alt="image" src="https://github.com/user-attachments/assets/c8bf648d-0ec0-4fd2-a611-e88137ae7373" />

plot_clusters() : visualiser les clusters en 2D.

<img width="494" height="226" alt="image" src="https://github.com/user-attachments/assets/d425d403-f03c-4a21-808a-e3f959240733" />

recommend() : générer les recommandations basées sur les préférences utilisateur.

<img width="552" height="353" alt="image" src="https://github.com/user-attachments/assets/186623ce-95a1-4ee3-bc70-53b6f51d3f3b" />

<img width="482" height="329" alt="image" src="https://github.com/user-attachments/assets/1855d0f3-7cdc-419d-bf3f-1f394eb6055f" />

## Exemple d’utilisation
1. Charger un fichier Excel
Cliquez sur “Charger Excel” et sélectionnez votre fichier.
Les colonnes numériques détectées apparaissent automatiquement dans la liste déroulante.

2. Paramétrer K-Means
Sélectionnez le nombre de clusters (k) et le max d’itérations. (k=4,maxitérations=300)
Cliquez sur “Exécuter K-Means” pour générer les clusters.

<img width="959" height="500" alt="image" src="https://github.com/user-attachments/assets/7e7e9c13-3e7d-462c-a218-f26a1a013209" />

3. Ajouter vos préférences utilisateur
Saisissez vos valeurs pour les caractéristiques (ex. Price, Size, MaterialQuality, Comfort, StyleScore, Rating).

<img width="317" height="197" alt="image" src="https://github.com/user-attachments/assets/2afd89c1-1f87-4bee-9a66-9b02dc96e091" />

4. Générer les recommandations
Cliquez sur “Recommander”.
Les meilleurs vêtements correspondant à vos préférences apparaissent dans le tableau.
Les points recommandés sont également mis en évidence dans le graphique PCA des clusters.

<img width="959" height="490" alt="image" src="https://github.com/user-attachments/assets/8e69ef51-2d7d-42d7-9017-5c288449847f" />


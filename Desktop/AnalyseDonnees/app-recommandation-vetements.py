"""
Système de recommandation de vêtements (K-Means) avec interface PyQt5

Usage:
  python recommender_kmeans_pyqt.py

Pré-requis (installer via pip):
  pip install pandas scikit-learn matplotlib pyqt5 openpyxl

Objectifs: 
- Illustrer l'utilisation de K-Means pour regrouper des vêtements selon leurs caractéristiques.
- Montrer comment PyQt5 peut servir d'interface utilisateur interactive.
- Démontrer la compréhension du pré-traitement, normalisation, PCA et calcul des recommandations.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout,
    QComboBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# -------------------------------
# Classe principale de l'application
# -------------------------------
class RecommenderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Définition du titre et de la taille de la fenêtre
        self.setWindowTitle("Recommender K-Means - Vêtements")
        self.resize(1000, 700)

        # Initialisation des variables utilisées tout au long de l'application
        self.df = None          # DataFrame contenant les données du fichier Excel
        self.features = None    # Colonnes numériques sélectionnées pour le clustering
        self.scaler = None      # StandardScaler pour normaliser les données
        self.kmeans = None      # Modèle K-Means
        self.pca = None         # PCA pour réduction en 2D (visualisation graphique)
        self.X_scaled = None    # Données normalisées pour K-Means

        self._build_ui()        # Construction de l'interface graphique

    # -------------------------------
    # Construction de l'interface graphique
    # -------------------------------
    def _build_ui(self):
        main = QWidget()
        main_layout = QHBoxLayout()
        left_col = QVBoxLayout()   # Colonne gauche : boutons et paramètres
        right_col = QVBoxLayout()  # Colonne droite : tableau et graphique

        # -------------------------------
        # Bouton pour charger un fichier Excel
        # -------------------------------
        btn_load = QPushButton("Charger Excel")
        btn_load.clicked.connect(self.load_excel)  # Lier le bouton à la fonction load_excel
        left_col.addWidget(btn_load)

        # ComboBox pour afficher les colonnes numériques détectées
        self.cb_features = QComboBox()
        left_col.addWidget(QLabel("Colonnes numériques détectées :"))
        left_col.addWidget(self.cb_features)

        # -------------------------------
        # Paramètres K-Means : nombre de clusters et max itérations
        # -------------------------------
        k_box = QGroupBox("Paramètres K-Means")
        k_layout = QFormLayout()
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 20)  # Limiter k entre 1 et 20
        self.spin_k.setValue(4)       # Valeur par défaut
        k_layout.addRow("Nombre de clusters (k):", self.spin_k)

        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(10, 1000)
        self.spin_max_iter.setValue(300)  # Valeur par défaut du max d'itérations
        k_layout.addRow("Max itérations:", self.spin_max_iter)
        k_box.setLayout(k_layout)
        left_col.addWidget(k_box)

        # -------------------------------
        # Bouton pour exécuter K-Means
        # -------------------------------
        btn_cluster = QPushButton("Exécuter K-Means")
        btn_cluster.clicked.connect(self.run_kmeans)
        left_col.addWidget(btn_cluster)

        # -------------------------------
        # Préférences utilisateur (valeurs par défaut ou exemples)
        # -------------------------------
        pref_box = QGroupBox("Préférences utilisateur (valeurs d'exemple)")
        pref_layout = QFormLayout()
        self.pref_inputs = {}
        # Création automatique des champs pour chaque caractéristique
        for name, default, mn, mx in [
            ("Price", 100, 0, 5000),
            ("Size", 38, 30, 50),
            ("MaterialQuality", 7, 1, 10),
            ("Comfort", 8, 1, 10),
            ("StyleScore", 7, 1, 10),
            ("Rating", 4.0, 0.0, 5.0)
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setValue(default)
            spin.setSingleStep(1.0)
            pref_layout.addRow(name+":", spin)
            self.pref_inputs[name] = spin
        pref_box.setLayout(pref_layout)
        left_col.addWidget(pref_box)

        # -------------------------------
        # Bouton pour générer les recommandations
        # -------------------------------
        btn_reco = QPushButton("Recommander")
        btn_reco.clicked.connect(self.recommend)
        left_col.addWidget(btn_reco)

        # Nombre de recommandations à afficher
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 50)
        self.spin_n.setValue(5)
        left_col.addWidget(QLabel("Nombre de recommandations:"))
        left_col.addWidget(self.spin_n)

        left_col.addStretch()  # Permet de pousser les widgets vers le haut

        # -------------------------------
        # Tableau pour afficher les recommandations
        # -------------------------------
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Brand","Item","Price","Size","MaterialQuality","Distance"])
        right_col.addWidget(QLabel("Recommandations"))
        right_col.addWidget(self.table, 2)

        # -------------------------------
        # Graphique pour visualiser les clusters
        # -------------------------------
        self.figure = plt.Figure(figsize=(5,4))
        self.canvas = FigureCanvas(self.figure)
        right_col.addWidget(self.canvas, 3)

        # -------------------------------
        # Assemblage des colonnes gauche et droite
        # -------------------------------
        main_layout.addLayout(left_col, 1)
        main_layout.addLayout(right_col, 2)
        main.setLayout(main_layout)
        self.setCentralWidget(main)

    # -------------------------------
    # Charger un fichier Excel et détecter les colonnes numériques
    # -------------------------------
    def load_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choisir fichier Excel", "", "Excel Files (*.xlsx *.xls)")
        if not path:
            return
        try:
            self.df = pd.read_excel(path)  # Lecture du fichier Excel avec pandas
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier:\n{e}")
            return

        # Détection des colonnes numériques pour le clustering
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QtWidgets.QMessageBox.critical(self, "Erreur", "Aucune colonne numérique détectée dans le fichier.")
            return

        # Remplissage de la ComboBox avec les colonnes numériques détectées
        self.cb_features.clear()
        for c in numeric_cols:
            self.cb_features.addItem(c)

        # Mise à jour des valeurs par défaut des préférences à partir des données
        for key in list(self.pref_inputs.keys()):
            if key in self.df.columns:
                val = self.df[key].dropna().median()
                try:
                    self.pref_inputs[key].setValue(float(val))
                except Exception:
                    pass

        QtWidgets.QMessageBox.information(self, "Succès", f"Fichier chargé avec {len(self.df)} lignes et {len(numeric_cols)} colonnes numériques détectées.")

    # -------------------------------
    # Exécuter le K-Means sur les données
    # -------------------------------
    def run_kmeans(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "Avertissement", "Chargez d'abord un fichier Excel.")
            return

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QtWidgets.QMessageBox.warning(self, "Avertissement", "Aucune colonne numérique disponible pour le clustering.")
            return

        self.features = numeric_cols
        # Remplacement des valeurs manquantes par la médiane pour chaque colonne
        X = self.df[self.features].fillna(self.df[self.features].median())

        # Standardisation des données pour K-Means (moyenne=0, écart-type=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # Paramètres du K-Means
        k = self.spin_k.value()
        max_iter = self.spin_max_iter.value()
        # Création et apprentissage du modèle K-Means
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=max_iter)
        self.kmeans.fit(self.X_scaled)
        labels = self.kmeans.labels_  # Récupération du cluster assigné à chaque vêtement
        self.df['_cluster'] = labels   # Ajout des labels dans le DataFrame pour référence

        # Réduction dimensionnelle en 2D pour la visualisation (PCA)
        self.pca = PCA(n_components=2)
        coords = self.pca.fit_transform(self.X_scaled)
        self.df['_pc1'] = coords[:,0]
        self.df['_pc2'] = coords[:,1]

        QtWidgets.QMessageBox.information(self, "K-Means", f"K-Means exécuté (k={k}).")
        self.plot_clusters()

    # -------------------------------
    # Visualiser les clusters sur un graphique 2D
    # -------------------------------
    def plot_clusters(self):
        if self.df is None or '_cluster' not in self.df.columns:
            return
        self.figure.clf()  # Nettoyer le graphique précédent
        ax = self.figure.add_subplot(111)
        # Affichage des points par cluster
        groups = self.df.groupby('_cluster')
        for name, group in groups:
            ax.scatter(group['_pc1'], group['_pc2'], label=f'Cluster {name}', alpha=0.6)
        # Affichage des centroïdes
        centers = self.kmeans.cluster_centers_
        centers_pca = self.pca.transform(centers)
        ax.scatter(centers_pca[:,0], centers_pca[:,1], marker='X', s=150, c='k', label='Centroids')
        ax.set_title('Clusters de vêtements (PCA 2D)')
        ax.legend()
        self.canvas.draw()

    # -------------------------------
    # Recommander les vêtements selon les préférences de l'utilisateur
    # -------------------------------
    def recommend(self):
        if self.df is None or self.kmeans is None:
            QtWidgets.QMessageBox.warning(self, "Avertissement", "Chargez un fichier et exécutez K-Means d'abord.")
            return

        # Création du vecteur de préférences utilisateur
        pref = []
        for f in self.features:
            if f in self.pref_inputs:
                pref.append(self.pref_inputs[f].value())
            else:
                pref.append(float(self.df[f].median()))
        pref_df = pd.DataFrame([pref], columns=self.features)
        pref_scaled = self.scaler.transform(pref_df)  # Normalisation identique aux données

        # Détection du cluster le plus proche de la préférence
        centroid_dists = np.linalg.norm(self.kmeans.cluster_centers_ - pref_scaled, axis=1)
        nearest_cluster = np.argmin(centroid_dists)

        # Calcul de la distance entre chaque vêtement et la préférence
        dists = np.linalg.norm(self.X_scaled - pref_scaled, axis=1)
        self.df['_dist_pref'] = dists

        # Sélection des N meilleures recommandations (les plus proches)
        n = self.spin_n.value()
        top = self.df.sort_values('_dist_pref').head(n)

        # Remplissage du tableau PyQt
        self.table.setRowCount(len(top))
        for i, (_, row) in enumerate(top.iterrows()):
            brand = str(row.get('Brand', ''))
            item = str(row.get('Item', ''))
            price = row.get('Price', '') if 'Price' in row else row.get(self.features[0], '')
            size = row.get('Size', '') if 'Size' in row else ''
            material = row.get('MaterialQuality', '') if 'MaterialQuality' in row else ''
            dist = float(row['_dist_pref'])
            vals = [brand, item, f"{price}", f"{size}", f"{material}", f"{dist:.3f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))

        # Affichage graphique des recommandations sur le cluster correspondant
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        for c in sorted(self.df['_cluster'].unique()):
            g = self.df[self.df['_cluster'] == c]
            alpha = 0.6 if c != nearest_cluster else 1.0
            zorder = 1 if c != nearest_cluster else 3
            ax.scatter(g['_pc1'], g['_pc2'], label=f'Cluster {c}', alpha=alpha, zorder=zorder)
        centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
        ax.scatter(centers_pca[:,0], centers_pca[:,1], marker='X', s=150, c='k', label='Centroids')
        ax.scatter(top['_pc1'], top['_pc2'], marker='*', s=200, label='Recommandés', zorder=4)
        ax.set_title(f'Recommandations de vêtements (cluster: {nearest_cluster})')
        ax.legend()
        self.canvas.draw()


# -------------------------------
# Point d'entrée principal de l'application
# -------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RecommenderApp()
    win.show()
    sys.exit(app.exec_())

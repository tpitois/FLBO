# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.datasets.TOSCA import TOSCA
from src.datasets.transforms import FLBOTransform
from src.models.ascnn import ACSCNN
from src.utils.eval import evaluate_predictions, plot_pck_curve

print("Chargement et pré-traitement du dataset TOSCA (Cat)...")
# PyG va automatiquement télécharger TOSCA, appliquer le pre_transform, et cacher le résultat.
# Si tu changes l'alpha ou le tau, supprime le dossier 'processed' !
dataset = TOSCA(
    root='/content/drive/MyDrive/TOSCA/data',
    category='Cat',
    pre_transform=FLBOTransform(n_angles=8, alpha=10.0, tau=0.5)
)

# Séparation Train/Test (ex: les 9 premiers chats pour l'entraînement, les 2 derniers pour le test)
train_dataset = dataset[:9]
test_dataset = dataset[9:]

# DataLoader PyG (batch_size=1 car on traite une forme complète à la fois)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ==========================================
# 2. INITIALISATION DU MODÈLE
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_desc = dataset[0].x.size(1)  # 3 (si on utilise x, y, z)
n_class = dataset[0].num_nodes  # Nombre exact de sommets d'un Chat TOSCA (27894)

model = ACSCNN(n_desc, n_class).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ==========================================
num_epochs = 100

scaler = torch.amp.GradScaler('cuda')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_nodes = 0

    for data in train_loader:
        # Transfert sur le GPU
        x = data.x.to(device)
        L = data.L.to(device)
        y = data.y.to(device)

        optimizer.zero_grad()

        # 2. On encapsule le forward pass et la loss dans l'autocast
        with torch.amp.autocast('cuda'):
            outputs = model(x, L)
            loss = criterion(outputs, y)

        # 3. On utilise le scaler pour la backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == y).item()
        total_nodes += y.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / total_nodes
    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    print("\n--- Début de l'évaluation géodésique sur le premier chat de test ---")
    model.eval()

    # On prend juste le premier chat du set de test
    test_data = test_dataset[0]
    x = test_data.x.to(device)
    L = test_data.L.to(device)
    y = test_data.y.cpu().numpy()  # Labels en numpy

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            outputs = model(x, L)
            _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()

    # On récupère les sommets et les faces d'origine du chat
    # (Si ton transformateur a écrasé data.pos, tu devras peut-être le recharger,
    #  ou t'assurer que tu l'as gardé dans l'objet Data)
    V = test_data.pos.numpy()
    F = test_data.face.t().numpy()

    print("Calcul des chemins géodésiques (ça peut prendre une minute)...")
    # On calcule les erreurs sur 1000 points aléatoires
    errors = evaluate_predictions(preds, y, V, F, num_samples=1000)

    # On affiche les stats
    print(f"Erreur géodésique moyenne : {np.mean(errors):.4f}")
    print(f"Matches parfaits (erreur = 0) : {np.mean(errors == 0) * 100:.2f}%")

    # On trace la courbe ! (Jusqu'à 20% de la taille du chat)
    plot_pck_curve(errors, max_threshold=0.20, save_path="tosca_evaluation_curve.png")
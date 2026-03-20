# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
import torch
from torch_geometric.loader import DataLoader

from src.datasets.TOSCA import TOSCA
from src.datasets.transforms import FLBOTransform
from src.models.ascnn import ACSCNN

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

        # Forward pass (attention: si PyG rajoute une dimension de batch, il faut l'enlever)
        # ACSConv s'attend à du [N, in_size]
        outputs = model(x, L)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_corrects += torch.sum(preds == y).item()
        total_nodes += y.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / total_nodes
    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")

    # --- Mode Évaluation (Test) ---
    model.eval()
    test_corrects = 0
    test_nodes = 0
    with torch.no_grad():
        for data in test_loader:
            x = data.x.to(device)
            L = data.L.to(device)
            y = data.y.to(device)

            outputs = model(x, L)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == y).item()
            test_nodes += y.size(0)

    print(f"Test Acc: {test_corrects / test_nodes:.4f}")
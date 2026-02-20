import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- ARCHITECTURE DU CNN (Modèle Fouda Etundi) ---
class CIFAKE_CNN(nn.Module):
    def __init__(self):
        super(CIFAKE_CNN, self).__init__()
        # Couche 1 : Entrée 3 canaux (RVB), Sortie 32, Filtre 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Couche 2 : Sortie 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2) # Divise la taille par 2 (32->16->8)
        
        # Couches de classification
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2) # 2 classes : FAKE (0) et REAL (1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8) # Aplatissement
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocessing Entraînement
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CHARGEMENT DES DONNÉES : Assure-toi que le dossier 'data/train' existe
    # avec les sous-dossiers 'FAKE' et 'REAL'
    try:
        train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = CIFAKE_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"Entraînement lancé sur {device} pour 3 époques...")
        model.train()
        for epoch in range(3):
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f"Époque [{epoch+1}/3], Step [{i}/{len(train_loader)}], Perte: {loss.item():.4f}")

        # Sauvegarde des poids
        torch.save(model.state_dict(), 'cifake_model.pth')
        print("Fichier 'cifake_model.pth' créé avec succès !")
    except FileNotFoundError:
        print("Erreur : Le dossier 'data/train' n'a pas été trouvé.")

if __name__ == "__main__":
    run_training()
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 1. RÉ-DÉCLARATION DE TON ARCHITECTURE CNN
class CIFAKE_CNN(nn.Module):
    def __init__(self):
        super(CIFAKE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 2. CHARGEMENT DU MODÈLE
@st.cache_resource
def load_my_model():
    model = CIFAKE_CNN()
    if os.path.exists('cifake_model.pth'):
        # On charge les poids que tu viens de générer
        model.load_state_dict(torch.load('cifake_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

# 3. INTERFACE UTILISATEUR
st.set_page_config(page_title="Détecteur CNN - Fouda Etundi", page_icon="🧠")
st.title("🛡️ Détecteur d'Images IA (CNN Custom)")
st.subheader("Étudiant : Fouda Etundi - Master 2 IABD")

model = load_my_model()

if model is None:
    st.error("❌ Erreur : 'cifake_model.pth' introuvable dans le dossier.")
else:
    st.sidebar.success("✅ Modèle CNN chargé")
    
    file = st.file_uploader("Téléverser une image (JPG, PNG)...", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Image à analyser", width=300)
        
        # 4. PREPROCESSING (Exactement comme dans ton code d'entraînement)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Ajout de la dimension Batch [1, 3, 32, 32]
        img_tensor = transform(img).unsqueeze(0)
        
        if st.button("Lancer l'Analyse"):
            with st.spinner("Analyse par le réseau de neurones..."):
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = F.softmax(output, dim=1)
                    prediction = torch.argmax(prob, dim=1).item()
                    confiance = prob[0][prediction].item() * 100

                # Classes CIFAKE : 0 = FAKE, 1 = REAL
                labels = ["FAKE (IA)", "REAL (Humain)"]
                
                if prediction == 0:
                    st.error(f"**VERDICT : {labels[prediction]}**")
                else:
                    st.success(f"**VERDICT : {labels[prediction]}**")
                
                st.info(f"Confiance : {confiance:.2f}%")
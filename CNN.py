import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

train_csv = "/kaggle/input/deepfake-classification-unibuc/train.csv"
train_dir = "/kaggle/input/deepfake-classification-unibuc/train"
validation_csv = "/kaggle/input/deepfake-classification-unibuc/validation.csv"
validation_dir = "/kaggle/input/deepfake-classification-unibuc/validation"
test_csv = "/kaggle/input/deepfake-classification-unibuc/test.csv"
test_dir = "/kaggle/input/deepfake-classification-unibuc/test"

class DatasetProc(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, is_test=False):  # pentru preprocesarea setului de date
        self.data=pd.read_csv(csv_path)
        self.img_dir=img_dir
        self.transform=transform
        self.is_test=is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id=self.data.iloc[idx, 0]
        img_path=os.path.join(self.img_dir, f"{img_id}.png")
        image=Image.open(img_path).convert("RGB")

        if self.transform:
            image=self.transform(image)

        if self.is_test:
            return img_id, image
        return image, int(self.data.iloc[idx, 1])


train_transform=transforms.Compose([                  #augmentarea setului de antrenare
    transforms.Resize((128, 128)),                      #redimensionare
    transforms.RandomHorizontalFlip(0.5),               #intoarcere orizontala
    transforms.RandomAffine(0, translate=(0.1, 0.1)),   #transformare afina
    transforms.RandomRotation(10),                      #rotatie
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   #modificarea luminozitatii si a contrastului
    transforms.ToTensor(),
    transforms.Normalize([0.49, 0.458, 0.405], [0.225, 0.226, 0.224])   #normalizare
])

val_transform=transforms.Compose([
    transforms.Resize((128, 128)),                  #redimensionare
    transforms.ToTensor(),
    transforms.Normalize([0.49, 0.458, 0.405], [0.225, 0.226, 0.224])       #normalizare
])

model=nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),         #primul bloc convolutional
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(64, 128, 3, padding=1),       #al doilea bloc
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(128, 256, 3, padding=1),      #al treilea bloc
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(256, 512, 3, padding=1),      #al patrulea bloc
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(512, 512, 3, padding=1),      #al cincilea bloc
    nn.BatchNorm2d(512),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(),                           #stratul de flatten

    nn.Linear(8192, 1024),           #primul strat fully connected
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(1024, 512),                   #al doilea strat fully connected
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(512, 5)                       #ultimul strat fully connected
)

train_dataset=DatasetProc(train_csv, train_dir, transform=train_transform)          #incarcarea datelor
val_dataset=DatasetProc(validation_csv, validation_dir, transform=val_transform)
test_dataset=DatasetProc(test_csv, test_dir, transform=val_transform, is_test=True)

train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
criterion=nn.CrossEntropyLoss(label_smoothing=0.1)          #cross entropy
optimizer=optim.Adam(model.parameters(), lr=0.001)          #optimizatorul Adam cu o rata de invatare initiala de 0.001
scheduler=lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, min_lr=1e-7)        #pentru a preveni stagnarea modelului
                                                                                                            #rata de invatare scade odata la 5 epoci fara progres
best_ac=0
valori_acuratete=[]

for epoca in range(80):             #antrenare timp de 80 de epoci
    model.train()
    for imagini, etichete in train_loader:      #se ia fiecare batch
        imagini, etichete=imagini.to(device), etichete.to(device) #trecerea informatiilor pe gpu
        optimizer.zero_grad()       #stergerea gradientilor vechi
        iesiri=model(imagini)       #predictiile
        loss=criterion(iesiri, etichete)    #calculeaza loss-ul
        loss.backward()             #calculeaza gradientii pentru a vedea cu cat trebuie schimbate ponderile ca sa se reduca loss-ul
        optimizer.step()            #actualizarea parametrilor

    model.eval()                    #evaluarea setului de validare
    corect=0
    total=0
    with torch.no_grad():           #dezactivarea gradientilor
        for imagini, etichete in val_loader:
            imagini, etichete=imagini.to(device), etichete.to(device)
            iesiri=model(imagini)
            _, pred=torch.max(iesiri, 1)
            total +=etichete.size(0)                    #numarul total de etichete
            corect +=(pred==etichete).sum().item()      #numarul de etichete prezise egale cu cele reale

    val_ac=100*corect/total                             #stocarea acuratetilor
    valori_acuratete.append(val_ac)
    scheduler.step(val_ac)                  #in caz ca e nevoie de reducerea ratei de invatare

    if val_ac>best_ac:                      #doresc sa salvez cei mai buni parametri
        best_ac=val_ac
        torch.save(model.state_dict(), 'best_model.pth')

    print(f'Epoca {epoca+1}: Acuratetea pe setul de validare: {val_ac:.2f}%')


model.load_state_dict(torch.load('best_model.pth'))
print(f"Cea mai mare acuratete: {best_ac:.2f}%")

model.eval()
y_val, val_pred=[], []
with torch.no_grad():               #pentru matricea de confuzie
    for imagini, etichete in val_loader:
        imagini, etichete = imagini.to(device), etichete.to(device)
        iesiri = model(imagini)
        _, pred = torch.max(iesiri, 1)
        y_val.extend(etichete.cpu().numpy())       #etichetele reale
        val_pred.extend(pred.cpu().numpy())          #predictiile

cm=confusion_matrix(y_val, val_pred)                #matricea de confuzie pentru cel mai bun model
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matricea de confuzie: {best_ac:.2f}%)")
plt.show()

plt.figure(figsize=(10, 6))                         #evolutia pe fiecare epoca
plt.plot(valori_acuratete)
plt.xlabel('Epoca')
plt.ylabel('Acuratetea pe setul de validare: (%)')
plt.title("Performanta")
plt.grid(True)
plt.show()

print("Raport:")            #raport detaliat al altor metrici
print(classification_report(y_val, val_pred))

test_ids, test_pred=[], []
with torch.no_grad():       #predictiile pe setul de test
    for img_ids, imagini in test_loader:
        imagini=imagini.to(device)      #se trec datele pe gpu pentru a rula modelul
        iesiri=model(imagini)
        _, pred=torch.max(iesiri, 1)
        test_ids.extend(img_ids)        #se salveaza predictiile
        test_pred.extend(pred.cpu().numpy()) #si se revine pe cpu

pd.DataFrame({'image_id': test_ids, 'label': test_pred}).to_csv("submission.csv", index=False)

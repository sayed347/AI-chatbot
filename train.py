import json
import numpy as np
from nltk_utils import tokenize,stem,bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import neuralNet
with open('intents.json','r') as f:
    intents=json.load(f)

#application de la tokenisation , stemming et bag of words
all_words=[]
tags=[]
xy=[]
for intent in intents ['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))


# création d'un tableaux numpy a partir de notre Json intents
x_train=[]
y_train=[]
for (pattern_sentence,tag)in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)
    
    label=tags.index(tag)
    y_train.append(label) # on va utiliser crossEntropyloss aprés je vais utiliser le 1 hot encoding 

x_train=np.array(x_train)
y_train=np.array(y_train)

# création d'une dataset à partir de notre tableaux numpy

class chatDataset(Dataset):
    def __init__(self) :
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    def __getitem__(self, index) :
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size=8  
dataset=chatDataset()
train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

# création de notre Model avec le model définit dans le fichier model
# définition de paramétre
hidden_size=8
output_size=len(tags) #l'output doit étre le nombre des tags qu'ont a dans notre intents
input_size= len(x_train[0]) # c'est la longeur de notre bag_of_wors aussi c'est le méme que all_words on a choisit içi x_train[0] parceque tout les x_train on la méme longeur
learning_rate=0.001
num_epochs = 1000 # c'est le nombre de test et répitition lors de l'entrainement
# création de device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=neuralNet(input_size,hidden_size,output_size).to(device) #on ajoute notre modéle sur le device pytorch

# création du loss et du optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(dtype=torch.long).to(device)

        # içi on réaliser la partie forward
        outputs=model(words)
        loss=criterion(outputs,labels)

        #içi on réalise la partie backward et l'optimizer 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100==0:
         # içi on va afficher pour chaque 100 test sur notre 1000 test
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#içi on va crée un dictionnaire pour enregisté notre data

data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}
#ici on va crée et enregistré les données dans un fichier pythorch
File="data.pth"
torch.save(data,File)

print(f'entrainement terminée et fichier enregistré dnas {File}')

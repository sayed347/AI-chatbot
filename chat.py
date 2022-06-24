import random
import json
import torch
from model import neuralNet
from nltk_utils import bag_of_words,tokenize

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents= json.load(f)

File="data.pth"
data=torch.load(File)
input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["all_words"]
tags=data["tags"]
model_state=data["model_state"]

model=neuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name="Efrei"
#print("Bienvenue chez Efrei chat écrivez 'sortir' pour quitter de la discution")
def get_response(msg):
    #sentence=input('Toi: ')
    #if sentence=="sortir":
      #  break
    sentence= tokenize(msg)
    x=bag_of_words(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x).to(device)

    output=model(x)
    _, predicted = torch.max(output, dim=1)
    tag=tags[predicted.item()]

    #içi on va utiliser softmax pour tester qu'on une probabilité bien elevée pour la réponse
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]

    if prob.item()>0.75:
      for intent in intents["intents"]:
         if tag == intent["tag"]:
             return random.choice(intent['responses'])
            # print(f"{bot_name}: {random.choice(intent['responses'])}")
    
    return "j'ai pas compris pouvez vous reformulez"
    #else:
       # print(f"{bot_name}:désolé j'ai pas bien compris pouvez vous réformuler...")

    
if __name__=="__main__":
    print("Bienvenue aux chat je suis Efrei écrivez 'sortir' pour quitter ")
    while True:
        sentence= input("Toi:")
        if sentence=="sortir":
            break
        resp=get_response(sentence)
        print(resp)

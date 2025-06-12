import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from transformer_classes import DecoderOnlyTransformer

import lightning as L

# inspired by https://github.com/StatQuest/decoder_transformer_from_scratch

token_to_id = {'what' : 0,
               'is' : 1,
               'DL' : 2,
               'awesome': 3,
               '<EOS>' : 4,
              }

id_to_token = {v: k for k, v in token_to_id.items()}

sentence1 = [token_to_id["what"],
            token_to_id["is"], 
            token_to_id["DL"], 
            token_to_id["<EOS>"],
            token_to_id["awesome"]]

sentence2 = [token_to_id["DL"],
            token_to_id["is"], 
            token_to_id["what"], 
            token_to_id["<EOS>"], 
            token_to_id["awesome"]]

inputs = torch.tensor([sentence1, sentence2])

label1 = [token_to_id["is"], 
        token_to_id["DL"], 
        token_to_id["<EOS>"], 
        token_to_id["awesome"], 
        token_to_id["<EOS>"]]

label2 = [token_to_id["is"], 
        token_to_id["what"], 
        token_to_id["<EOS>"], 
        token_to_id["awesome"], 
        token_to_id["<EOS>"]]

labels = torch.tensor([label1,  
                       label2
                       ])

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)

model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6, seed=43)

model_input = torch.tensor([token_to_id["what"], 
                            token_to_id["is"], 
                            token_to_id["DL"], 
                            token_to_id["<EOS>"]])

def run_prompt(model_input: torch.tensor, model: L.LightningModule):
    print(">>> ", " ".join([id_to_token[i.item()] for i in model_input]))

    predictions = model(model_input) 
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
    predicted_ids = predicted_id
    input_length = model_input.size(dim=0)

    max_length = 6
    for i in range(input_length, max_length):
        if (predicted_id == token_to_id["<EOS>"]):
            break

        model_input = torch.cat((model_input, predicted_id))
        
        predictions = model(model_input) 
        predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    print("Predicted Tokens:\n") 
    print(predicted_ids)
    for id in predicted_ids: 
        print("\t", id_to_token[id.item()])

def train_model(model: L.LightningModule, dataloader: DataLoader):
    trainer = L.Trainer(max_epochs=30)
    trainer.fit(model, train_dataloaders=dataloader)

print("*** BEFORE TRAINING ***")
run_prompt(model_input, model)
print("*** RUNNING TRAINING ***")
train_model(model, dataloader)
print("*** AFTER TRAINING ***")
run_prompt(model_input, model)

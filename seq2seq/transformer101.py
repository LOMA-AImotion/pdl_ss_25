import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)


        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        self.register_buffer('pe', pe) 


    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :] 

class Attention(nn.Module): 

    def __init__(self, d_model=2):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = 0
        self.col_dim = 1


    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores

class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens=4, d_model=2, max_len=6, seed=42):
        super().__init__()
        L.seed_everything(seed=seed)

        self.we = nn.Embedding(num_embeddings=num_tokens, 
                               embedding_dim=d_model)     
        self.pe = PositionEncoding(d_model=d_model, 
                                   max_len=max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)

        self.loss = nn.CrossEntropyLoss()


    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)        
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded, 
                                                    position_encoded, 
                                                    position_encoded, 
                                                    mask=mask)

        residual_connection_values = position_encoded + self_attention_values        
        fc_layer_output = self.fc_layer(residual_connection_values)
        return fc_layer_output


    def configure_optimizers(self): 
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx): 
        input_tokens, labels = batch # collect input
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss
    

if __name__ == "__main__":
    ## first, we create a dictionary that maps vocabulary tokens to id numbers...
    token_to_id = {'what' : 0,
                'is' : 1,
                'statquest' : 2,
                'awesome': 3,
                '<EOS>' : 4, ## <EOS> = end of sequence
                }

    id_to_token = {v: k for k, v in token_to_id.items()}

    print(id_to_token)

    sentence1 = [token_to_id["what"], ## input #1: what is statquest <EOS> awesome
            token_to_id["is"], 
            token_to_id["statquest"], 
            token_to_id["<EOS>"],
            token_to_id["awesome"]]
    sentence2 = [token_to_id["statquest"], # input #2: statquest is what <EOS> awesome
                token_to_id["is"], 
                token_to_id["what"], 
                token_to_id["<EOS>"], 
                token_to_id["awesome"]]

    inputs = torch.tensor([sentence1, sentence2])

   
    label1 = [token_to_id["is"], 
            token_to_id["statquest"], 
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

    ## Now let's package everything up into a DataLoader...
    dataset = TensorDataset(inputs, labels) 
    dataloader = DataLoader(dataset)

    ## First, create a model from DecoderOnlyTransformer()
    model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6, seed=43)

    ## Now create the input for the transformer...
    model_input = torch.tensor([token_to_id["what"], 
                                token_to_id["is"], 
                                token_to_id["statquest"], 
                                token_to_id["<EOS>"]])
    input_length = model_input.size(dim=0)

    ## Now get get predictions from the model
    predictions = model(model_input) 
    print(predictions)

    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
    print(predicted_id)
    predicted_ids = predicted_id

    max_length = 6
    for i in range(input_length, max_length):
        print(id_to_token[predicted_id.item()])
        if (predicted_id == token_to_id["<EOS>"]): # if the prediction is <EOS>, then we are done
            break
        
        model_input = torch.cat((model_input, predicted_id))
        print("i = ",i)
        predictions = model(model_input) 
        predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))
            
        ## Now printout the predicted output phrase.
        print("Predicted Tokens:\n") 
        print(predicted_ids)
        for id in predicted_ids: 
            print("\t", id_to_token[id.item()])

        model_input = torch.tensor([token_to_id["what"], 
                                    token_to_id["is"], 
                                    token_to_id["statquest"], 
                                    token_to_id["<EOS>"]])
        input_length = model_input.size(dim=0)

        predictions = model(model_input) 
        predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
        predicted_ids = predicted_id

        for i in range(input_length, max_length):
            if (predicted_id == token_to_id["<EOS>"]): # if the prediction is <EOS>, then we are done
                break
            
            model_input = torch.cat((model_input, predicted_id))
            
            predictions = model(model_input) 
            predicted_id = torch.tensor([torch.argmax(predictions[-1,:])])
            predicted_ids = torch.cat((predicted_ids, predicted_id))
                
        print("Predicted Tokens:\n") 
        for id in predicted_ids: 
            print("\t", id_to_token[id.item()])

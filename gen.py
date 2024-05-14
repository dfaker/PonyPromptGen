

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import json

SEQUENCE_LENGTH = 64*2

word_to_int = json.load(open('word_to_int','r'))
int_to_word = {i: word for word, i in word_to_int.items()}

vocab_size = len(word_to_int)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(TextGen, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=SEQUENCE_LENGTH, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        emb = self.emb(x)
        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x)
        x = self.dropout(x)
        out = self.linear(x)
        return out

model = TextGen(
    vocab_size=vocab_size, 
    embed_dim=100,
    num_layers=2, 
    num_heads=2,
).to(device)

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int.get(word,1) for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, temperature=1.0):
    predictions = predictions / temperature
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.multinomial(probabilities, 1)
    return int(next_token.cpu().item())

def text_generator(sentence, generate_length,temperature=1.0):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions,temperature=temperature)
        sample += ' ' + int_to_word[next_token]
    return sample

try:
    model.load_state_dict(torch.load('generator.model'))
    model.eval()
    print('weights loaded')
except Exception as e:
    print(e)


prompt = input('seed prompt>')
if len(prompt)>0:
    prompt = ', '.join([x.strip().lower().replace('_',' ') for x in prompt.split(',')])
    prompt += ','
else:
    prompt = ''


while 1:
    try:
        prompt = '_SEP_ '+prompt

        temp=1.0

        resp = text_generator(prompt ,100,temperature=temp)
        resp = resp.replace(' , ',', ').replace('( ','(').replace(' )',')')

        outs = []
        temp = []
        for e in resp.split(' '):
            if '_SEP_' in e  and len(temp) > 1:
                outs.append(temp)
                temp = []
            elif '_SEP_' in e:
                temp = []
            else:
                temp.append(e)

        for tk in outs:
            o = ' '.join(tk).strip()
            o = 'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up, '+o


            if len(o)>40:
                print('')
                print(o)
            if prompt != '':
                break


    except Exception as ex:
        print(ex)
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
from torchtext.data import get_tokenizer

with open('prompts_all.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    text = text.split('\n')
    random.shuffle(text)
    text = '\n'.join(text)

words = text.split()
word_counts = Counter(words)
vocab = list(word_counts.keys())
vocab_size = len(vocab)

word_to_int = {word: i for i, word in enumerate(vocab)}

json.dump(word_to_int,open('word_to_int','w'))

int_to_word = {i: word for word, i in word_to_int.items()}
SEQUENCE_LENGTH = 64*2
samples = [words[i:i+SEQUENCE_LENGTH+1] for i in range(len(words)-SEQUENCE_LENGTH)]

print(vocab)
print(word_to_int)
print(int_to_word)

class TextDataset(Dataset):
    def __init__(self, samples, word_to_int):
        self.samples = samples
        self.word_to_int = word_to_int
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]])
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]])
        return input_seq, target_seq


BATCH_SIZE = 32
dataset = TextDataset(samples, word_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)
print(dataset[1])

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out

epochs = 100
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TextGen(
    vocab_size=vocab_size, 
    embed_dim=100,
    num_layers=2, 
    num_heads=2,
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.\n")


def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
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


if False:
    while 1:
        try:
            prompt = '_SEP_, '
            if not prompt.endswith(','):
                prompt = prompt+','
            for temp in [1.0,1.2,1.5,2]:
                print(f'\ntemp {temp}\n:')
                resp = text_generator(prompt ,100,temperature=temp)
                print(resp.replace(' , ',', '))
        except:
            pass

def train(model, epochs, dataloader, criterion):
    model.train()
    n=0
    for epoch in range(epochs):
        running_loss = 0
        
        for input_seq, target_seq in dataloader:
            n+=1
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, vocab_size)
            
            loss = criterion(outputs, target_seq.view(-1))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().numpy()


            
            print(running_loss / n,n,len(dataloader))

            if n % 500 == 0:
                torch.save(model.state_dict(), 'generator.model')

                for _ in range(4):
                    try:
                        prompt = '_SEP_ '

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

                    except Exception as ex:
                        print(ex)




                model.train()

        epoch_loss = running_loss / len(dataloader)
        torch.save(model.state_dict(), 'generator.model')
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")
train(model, epochs, dataloader, criterion)
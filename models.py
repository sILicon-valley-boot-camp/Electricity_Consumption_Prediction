import torch
from torch import nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros([self.num_layers, x.size(0), self.hidden_size], dtype = torch.float32).to(x.device) 

        out, _ = self.rnn(x, h0)  
        out = self.fc(out)

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros([self.num_layers, x.size(0), self.hidden_size], dtype = torch.float32).to(x.device) 
        c0 = torch.zeros([self.num_layers, x.size(0), self.hidden_size], dtype = torch.float32).to(x.device) 

        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros([self.num_layers, x.size(0), self.hidden_size], dtype = torch.float32).to(x.device) 

        out, _ = self.gru(x, h0)  
        out = self.fc(out)

        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead, num_encoder_layers=None, num_decoder_layers=None):
        super(TransformerModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        
        if not num_encoder_layers:
            num_encoder_layers = num_layers
        if not num_decoder_layers:
            num_decoder_layers = num_layers
        
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
        
        self.pos_encoder = nn.Embedding(5000, hidden_size)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = self.embedding(x) + self.pos_encoder(positions)
        
        out = self.transformer(x, x)
        
        out = self.fc(out)

        return out

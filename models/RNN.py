import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, args, feature_size):
        super().__init__()
        self.pooling = args.pooling
        self.rnn = getattr(nn , args.model)(input_size=feature_size, hidden_size=args.hidden, num_layers=args.num_layers)

    def forward(self, src):
        output, _ = self.rnn(src)
        output = torch.transpose(output, 0, 1).contiguous()

        if self.pooling == 'mean':
            output = torch.mean(output, 1).squeeze()
        elif self.pooling == 'max':
            output = torch.max(output, 1)[0].squeeze()
        elif self.pooling == 'last':
            output = output[:, -1, :]
        elif self.pooling == 'first':
            output = output[:, 0, :]
        elif self.pooling == 'all':
            pass
        else:
            raise NotImplementedError
        
        return output
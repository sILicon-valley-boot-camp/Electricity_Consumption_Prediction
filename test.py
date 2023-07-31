import torch
from tqdm import tqdm

class Tester():
    def __init__(self, test_loader, model):
        self.test_loader = test_loader
        self.model = model
        self.predictions = []

    def test(self):
        # Testing Phase
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            for i, batch in progress_bar:
                inputs = batch['input']
                outputs = self.model(inputs)

                preds = outputs.data.numpy()
                self.predictions.extend(preds)

        print('\nTesting completed.')

import torch
from tqdm import tqdm

class Tester():
    def __init__(self, test_loader, model, loss_fn):
        self.test_loader = test_loader
        self.model = model
        self.loss_fn = loss_fn

        self.predictions = []
        
    def test(self):
        # Testing Phase
        self.model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            for i, (inputs, labels) in progress_bar:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_test_loss += loss.item() * inputs.size(0)
                progress_bar.set_description(f'Test Loss: {loss.item():.4f}')

                preds = outputs.data.numpy()
                self.predictions.extend(preds)

        test_loss = running_test_loss / len(self.test_loader.dataset)
        print(f'\nTest Loss: {test_loss:.4f}')
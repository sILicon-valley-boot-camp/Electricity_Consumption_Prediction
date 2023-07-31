import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, epochs):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        train_loss_values = []
        valid_loss_values = []

        for epoch in range(self.epochs):
            running_loss = 0.0

            # Training Phase
            self.model.train()
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (inputs, labels) in progress_bar:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.4f}')

            epoch_loss = running_loss / len(self.train_loader.dataset)
            train_loss_values.append(epoch_loss)
            print(f'\nTrain Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}')

            # Validation Phase
            self.model.eval()
            with torch.no_grad():
                running_valid_loss = 0.0
                for inputs, labels in self.valid_loader:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    running_valid_loss += loss.item() * inputs.size(0)
                    
                valid_loss = running_valid_loss / len(self.valid_loader.dataset)
                valid_loss_values.append(valid_loss)
                print(f'Validation Loss: {valid_loss:.4f}\n')

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), train_loss_values, color='blue', label='Training Loss', marker='o', linestyle='dashed', linewidth=2, markersize=6)
        plt.plot(range(1, self.epochs + 1), valid_loss_values, color='red', label='Validation Loss', marker='o', linestyle='dashed', linewidth=2, markersize=6)
        plt.title('Training and Validation Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

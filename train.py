import os
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, epochs, device):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.result_dir = self.get_result_dir()

    def get_result_dir(self):
        existing_dirs = [dname for dname in os.listdir() if "result_kfold" in dname]
        dir_count = len(existing_dirs)
        result_dir = f'result_kfold{dir_count + 1}'
        os.mkdir(result_dir)
        return result_dir

    def train_one_epoch(self, epoch, logger):
        running_loss = 0.0
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, batch in progress_bar:
            inputs = batch[0].float().to(self.device)
            labels = batch[1].float().to(self.device)
            outputs = self.model(inputs)
            labels = labels.unsqueeze(2)
            loss = self.loss_fn(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.4f}')
            logger.info(f'Epoch {epoch+1} Batch {i+1} Loss: {loss.item():.4f}')

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        running_valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_loader:
                inputs = batch[0].float().to(self.device)
                labels = batch[1].float().to(self.device)
                outputs = self.model(inputs)
                labels = labels.unsqueeze(2)
                loss = self.loss_fn(outputs, labels)
                running_valid_loss += loss.item() * inputs.size(0)

        return running_valid_loss / len(self.valid_loader.dataset)

    def train(self, fold, logger):
        train_loss_values = []
        valid_loss_values = []
        best_valid_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch, logger)
            train_loss_values.append(train_loss)
            print(f'\nTrain Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}')
            valid_loss = self.validate()
            valid_loss_values.append(valid_loss)
            print(f'Validation Loss: {valid_loss:.4f}\n')
            
            torch.save(self.model.state_dict(), f'{self.result_dir}/fold_{fold}_model_weights_epoch_{epoch+1}.pth')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{self.result_dir}/fold_{fold}_best_model_weights.pth')

        return best_valid_loss

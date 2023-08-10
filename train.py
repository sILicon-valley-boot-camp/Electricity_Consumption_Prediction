import os
import torch
from torch.utils.checkpoint import checkpoint
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

    def train_one_epoch(self, epoch, fold, logger):
        running_loss = 0.0
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        for i, batch in progress_bar:
            inputs = batch['input'].float().to(self.device).requires_grad_(True)
            labels = batch['label'].float().to(self.device)
            outputs = checkpoint(self.model, inputs)
            labels = labels.unsqueeze(2)
            loss = self.loss_fn(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_description(f'Epoch {epoch+1}/{self.epochs} Loss: {loss.item():.4f}')

        logger.info(f'Fold {fold+1} | Epoch {epoch+1}/{self.epochs} | Average Training Loss: {running_loss / len(self.train_loader.dataset):.4f}')
        return running_loss / len(self.train_loader.dataset)

    def validate(self, fold, logger):
        running_valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in self.valid_loader:
                inputs = batch['input'].float().to(self.device)
                labels = batch['label'].float().to(self.device)
                outputs = self.model(inputs)
                labels = labels.unsqueeze(2)
                loss = self.loss_fn(outputs, labels)
                running_valid_loss += loss.item() * inputs.size(0)

        logger.info(f'Fold {fold+1} | Validation Average Loss: {running_valid_loss / len(self.valid_loader.dataset):.4f}')
        return running_valid_loss / len(self.valid_loader.dataset)

    def train(self, fold, logger):
        best_valid_loss = float('inf')

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(epoch, fold, logger)
            print(f'\nTrain Epoch {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}')
            valid_loss = self.validate(fold, logger)
            print(f'Validation Loss: {valid_loss:.4f}\n')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{self.result_dir}/fold_{fold}_best_model_weights.pth')

        return best_valid_loss

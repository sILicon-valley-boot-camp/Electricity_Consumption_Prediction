import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from utils import smape

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, target_scaler, device, patience, epochs, result_path, fold_logger, len_train, len_valid):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.target_scaler = target_scaler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.len_train = len_train
        self.len_valid = len_valid
    
    def train(self):
        best = np.inf
        for epoch in range(1,self.epochs+1):
            loss_train, smape_train = self.train_step()
            loss_val, smape_valid = self.valid_step()
            self.scheduler.step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_smape:{smape_train:.3f} v_loss:{loss_val:.3f} v_smape:{smape_valid:.3f}')

            if loss_val < best:
                best = loss_val
                torch.save(self.model.state_dict(), self.best_model_path)
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break

    def train_step(self):
        self.model.train()

        total_loss = 0
        total_smape = 0
        for batch in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            for key in batch.keys():
                batch[key] = batch[key].to(self.device).squeeze(0)
                
            y = batch.pop('y').reshape(-1, 1)
            self.optimizer.zero_grad()
            output = self.model(**batch)  
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_smape += smape(
                self.target_scaler.inverse_transform(y.detach().cpu().numpy()), 
                self.target_scaler.inverse_transform(output.detach().cpu().numpy())
            )
        
        return total_loss/self.len_train, total_smape/self.len_train
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_smape = 0
            for batch in self.valid_loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device).squeeze(0)
                
                y = batch.pop('y').reshape(-1, 1)
                output = self.model(**batch)            
                loss = self.loss_fn(output, y)

                total_loss += loss.item()
                total_smape += smape(
                    self.target_scaler.inverse_transform(y.detach().cpu().numpy()), 
                    self.target_scaler.inverse_transform(output.detach().cpu().numpy())
                )
                
        return total_loss/self.len_valid, total_smape/self.len_valid
    
    def test(self, test_loader): #for making predictions on validation set, generating input for stacking Ensemble
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        with torch.no_grad():
            result = []
            for batch in test_loader:
                del batch['y']
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device).squeeze(0)
                
                output = self.model(**batch).detach().cpu().unsqueeze(-1).numpy()
                result.append(output)

        result_array = np.stack(result,axis=0).T.reshape(-1, 1)
        if self.target_scaler is None:
            return result_array
        else:
            return self.target_scaler.inverse_transform(result_array)
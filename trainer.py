import os
import sys
import torch
import optuna
import numpy as np
from tqdm import tqdm
from ray.air import session

from utils import smape

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, scaling_fn, device, patience, epochs, result_path, fold_logger, len_train, len_valid, trial=None, use_ray=False):
        self.train_loader = tqdm(train_loader, file=sys.stdout) if trial is None else train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaling_fn = scaling_fn
        self.flat = train_loader.dataset.flat
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model.pt')
        self.len_train = len_train
        self.len_valid = len_valid
        
        self.trial = trial
        self.use_ray = use_ray
                  
    def train(self):
        best = np.inf
        for epoch in range(1,self.epochs+1):
            loss_train, smape_train = self.train_step(epoch)
            loss_val, smape_valid = self.valid_step(epoch)
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

            if self.use_ray:
                session.report({"loss": loss_val, "smape":smape_valid})

        return best

    def train_step(self, epoch):
        self.model.train()

        total_loss = 0
        total_smape = 0
        for batch in self.train_loader:
            del batch['batch']; del batch['ptr']
            batch = batch.to(self.device)
            flat = self.flat.to(self.device)

            y = batch.pop('y').reshape(-1, 1)
            self.optimizer.zero_grad()
            output = self.model(
                    node_feat=batch['x'], 
                    flat=flat, 
                    edge_index=batch['edge_index'] if 'edge_index' in batch.keys else None, 
                    edge_weight=batch['edge_attr'] if 'edge_attr' in batch.keys else None
            )
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * (batch['x'].shape[0]//100)
            total_smape += smape(
                self.scaling_fn(y.detach().cpu().numpy()), 
                self.scaling_fn(output.detach().cpu().numpy())
            ) * (batch['x'].shape[0]//100)
        
        return total_loss/self.len_train, total_smape/self.len_train
    
    def valid_step(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_smape = 0
            for batch in self.valid_loader:
                del batch['batch']; del batch['ptr']
                batch = batch.to(self.device)
                flat = self.valid_loader.dataset.flat.to(self.device)
                
                y = batch.pop('y').reshape(-1, 1)
                output = self.model(
                    node_feat=batch['x'], 
                    flat=flat, 
                    edge_index=batch['edge_index'] if 'edge_index' in batch.keys else None, 
                    edge_weight=batch['edge_attr'] if 'edge_attr' in batch.keys else None
                ) 
                loss = self.loss_fn(output, y)

                total_loss += loss.item() * (batch['x'].shape[0]//100)
                total_smape += smape(
                    self.scaling_fn(y.detach().cpu().numpy()), 
                    self.scaling_fn(output.detach().cpu().numpy())
                ) * (batch['x'].shape[0]//100)

                if self.trial is not None: #optuna
                    self.trial.report(total_loss/self.len_valid, epoch)
                    
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        return total_loss/self.len_valid, total_smape/self.len_valid
    
    def test(self, test_loader): #for making predictions on validation set, generating input for stacking Ensemble
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        with torch.no_grad():
            result = []
            for batch in test_loader:
                del batch['y']; del batch['batch']; del batch['ptr']
                batch = batch.to(self.device)
                flat = test_loader.dataset.flat.to(self.device)

                output = self.model(
                    node_feat=batch['x'], 
                    flat=flat, 
                    edge_index=batch['edge_index'] if 'edge_index' in batch.keys else None, 
                    edge_weight=batch['edge_attr'] if 'edge_attr' in batch.keys else None
                ).detach().cpu().unsqueeze(-1).reshape(-1, 100).numpy()
                
                result.append(output)

        result_array = np.concatenate(result,axis=0).T.reshape(-1, 1)
        return self.scaling_fn(result_array)
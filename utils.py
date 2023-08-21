import os
import sys
import json
import torch
import joblib
import optuna
import random
import warnings
warnings.simplefilter("once")
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def smape(true, pred):
    v = 2 * abs(pred - true) / ((abs(pred) + abs(true)) + 1e-9)
    output = np.mean(v) * 100
    return output

def handle_unhandled_exception(exc_type, exc_value, exc_traceback, logger=None):
    if issubclass(exc_type, KeyboardInterrupt):
                #Will call default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        #Create a critical level log message with info from the except hook.
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

def save_to_json(data, file_name):
    with open(file_name, 'w') as fp:
        json.dump(data, fp)

class SaveStudyCallback:
    def __init__(self, path):
        self.path = path

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        joblib.dump(study, os.path.join(self.path, f"study.pkl"))

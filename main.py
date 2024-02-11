
from nnModel import DiabeticsNN
import torch
import os
import pandas as pd


torch.manual_seed(41)
model = DiabeticsNN(8, 2, 200, 200, 200)

cwd = os.getcwd()
diabetic_data = pd.read_csv(os.path.join(cwd, "diabetes.csv"))


rom src.data import load_train
from src.train import train_model
import torch

train_dataset,train_loader=load_train()
model=train_model(train_dataset=train_dataset,train_dataloader=train_loader)
torch.save(model.state_dict(), "models/resnet18_weights.pth")
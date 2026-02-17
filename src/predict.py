import torch
import numpy as np

def get_predictions(model, loader, device):
  model.eval()
  all_preds = []
  all_labels = []
  
  with torch.no_grad():
    for batch in loader:
      input_ids = batch['input_ids'].to(device)
      labels = batch['label'].to(device)
      
      outputs = model(input_ids)
      _, predicted = torch.max(outputs, 1)
      
      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
  
  return np.array(all_preds), np.array(all_labels)
import torch

# Fonction d'évaluation
def evaluate(model, loader, criterion, device):
  model.eval()
  total_loss = 0
  correct = 0
  total = 0
  
  with torch.no_grad():
    for batch in loader:
      input_ids = batch['input_ids'].to(device)
      labels = batch['label'].to(device)
      
      outputs = model(input_ids)
      loss = criterion(outputs, labels)
      
      total_loss += loss.item()
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  
  return total_loss / len(loader), correct / total
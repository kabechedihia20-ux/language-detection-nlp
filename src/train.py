import torch

# Fonction d'entraînement
def train_epoch(model, loader, criterion, optimizer, device):
  model.train()
  total_loss = 0
  correct = 0
  total = 0
  
  for batch in loader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['label'].to(device)
    
    optimizer.zero_grad()
    outputs = model(input_ids)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  
  return total_loss / len(loader), correct / total
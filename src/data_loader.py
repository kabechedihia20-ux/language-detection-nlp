import torch
from torch.utils.data import Dataset, DataLoader

from src.preprocess import text_to_sequence, text_to_subword_sequence

class LanguageDataset(Dataset):
  def __init__(self, texts, labels, vocab, max_length):
    self.texts = texts
    self.labels = labels
    self.vocab = vocab
    self.max_length = max_length
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    
    sequence = text_to_sequence(text, self.vocab, self.max_length)
    
    return {
      'input_ids': torch.tensor(sequence, dtype=torch.long),
      'label': torch.tensor(label, dtype=torch.long)
    }
  

class SubwordLanguageDataset(Dataset):
  def __init__(self, texts, labels, sp, max_length):
    self.texts = texts
    self.labels = labels
    self.sp = sp
    self.max_length = max_length
  
  def __len__(self):
    return len(self.texts)
  
  def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    sequence = text_to_subword_sequence(text, self.sp, self.max_length)
    return {
      'input_ids': torch.tensor(sequence, dtype=torch.long),
      'label': torch.tensor(label, dtype=torch.long)
    }
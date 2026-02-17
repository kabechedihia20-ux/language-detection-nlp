import torch
import torch.nn as nn

class LanguageDetectorLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, n_layers=2, dropout=0.3):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(hidden_dim, num_classes)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x shape: (batch_size, seq_len)
    embedded = self.embedding(x) # (batch_size, seq_len, embedding_dim)
    lstm_out, (hidden, cell) = self.lstm(embedded)

    # Prendre la dernière hidden state
    last_hidden = hidden[-1] # (batch_size, hidden_dim)
    out = self.dropout(last_hidden)
    out = self.fc(out) # (bach_size, num_classes)
    return out



class BiLSTMSubword(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, n_layers=3, dropout=0.5):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(
      embedding_dim, 
      hidden_dim, 
      n_layers, 
      batch_first=True, 
      dropout=dropout, 
      bidirectional=True
    )
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(hidden_dim * 2, num_classes)
  
  def forward(self, x):
    # x: (batch_size, seq_len)
    embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
    lstm_out, (hidden, cell) = self.lstm(embedded)
    
    # Concaténer les dernières hidden states des deux directions
    # hidden[-2] = dernière couche, direction avant
    # hidden[-1] = dernière couche, direction arrière
    last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim * 2)
    
    out = self.dropout(last_hidden)
    out = self.fc(out)  # (batch_size, num_classes)
    return out
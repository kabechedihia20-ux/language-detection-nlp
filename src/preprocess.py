import re
from collections import Counter
import pandas as pd

max_length = 278 # Notre seuil de troncature (95e percentile)

# Fonction de prétraitement de texte
def preprocess_text(text, remove_digits=True, min_length=2):
    """
    Clean and preprocess text for language detection
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove digits (optional - sometimes useful for language detection)
    if remove_digits:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation but keep language-specific characters
    # We'll keep letters, spaces, and common punctuation that might be language-specific
    text = re.sub(r'[^\w\s\u0080-\uFFFF]', ' ', text)  # Keep Unicode characters
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short tokens (often noise)
    words = text.split()
    words = [w for w in words if len(w) >= min_length]
    text = ' '.join(words)
    
    return text



# Construisons le vocabulaire
def build_vocab(texts, max_vocab_size=1000):
    counter = Counter()
    for text in texts:
        counter.update(list(text))  # tokenisation caractères
    
    # Tokens spéciaux
    special_tokens = ['<PAD>', '<UNK>']
    
    # Prendre les caractères les plus fréquents
    most_common = counter.most_common(max_vocab_size - len(special_tokens))
    
    # Créer le vocabulaire
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in most_common:
        vocab[token] = len(vocab)
    
    return vocab



def text_to_sequence(text, vocab, max_length):
    """Convertit un texte en séquence d'identifiants avec padding"""
    # Tokeniser et convertir en identifiants
    tokens = list(text)
    sequence = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    
    # Padding ou troncature
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
    
    return sequence



# Définissons la fonction de conversion texte -> séquence
def text_to_subword_sequence(text, sp, max_length):
    """Convertit un texte en séquence d'identifiants sous-mots avec padding"""
    ids = sp.encode(text, out_type=int)
    
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + [0] * (max_length - len(ids))  # 0 = <pad>
    
    return ids
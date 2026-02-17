import torch
import torch
import torch.nn.functional as F
import numpy as np

import src
from src.preprocess import max_length, preprocess_text

import json
from pathlib import Path
import sentencepiece as spm
from src.model import BiLSTMSubword

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



# Fonctions d’encodage + prédiction (top-k)
def encode_text_sp(text: str, sp, max_length: int):
    ids = sp.encode(text, out_type=int)
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids = ids + [0] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def predict_language(text: str, model, sp, id2label: dict, device, top_k=3, already_clean=False):
    max_length = src.preprocess.max_length

    text_clean = text if already_clean else preprocess_text(text)

    x = encode_text_sp(text_clean, sp, max_length).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    top_probs, top_ids = torch.topk(probs, k=top_k, dim=1)

    results = []
    for p, idx in zip(top_probs[0].cpu().tolist(), top_ids[0].cpu().tolist()):
        results.append((id2label[idx], float(p)))
    return results


def load_pipeline(project_root: str, device=None):
    """
    Charge le modèle BiLSTM + SentencePiece + mapping id->label.
    project_root : chemin vers la racine du projet.
    """
    root = Path(project_root)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = root / "models" / "bilstm_subword_final.pt"
    spm_path = root / "notebooks" / "spm_lang_detector.model"
    labels_path = root / "models" / "labels.json"

    # labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    id2label = {i: labels[i] for i in range(len(labels))}

    # sentencepiece
    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))

    # checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # modèle
    model = BiLSTMSubword(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        n_layers=checkpoint["n_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, sp, id2label, device


def predict_language_app(
    text: str,
    model,
    sp,
    id2label: dict,
    device,
    top_k: int = 3,
    min_chars: int = 20,
    confidence_threshold: float = 0.50
):
    """
    Version Streamlit: retourne un dict (status + predictions).
    """
    text_clean = preprocess_text(text)

    if len(text_clean) < min_chars:
        return {
            "status": "too_short",
            "message": f"Le texte doit contenir au moins {min_chars} caractères.",
            "predictions": []
        }

    # preds = predict_language(text_clean, model, sp, id2label, device, top_k=top_k)
    preds = predict_language(text_clean, model, sp, id2label, device, top_k=top_k, already_clean=True)
    # preds = [(lang, prob), ...]
    predictions = [{"language": lang, "confidence": prob} for lang, prob in preds]

    status = "ok"
    if predictions and predictions[0]["confidence"] < confidence_threshold:
        status = "low_confidence"

    return {
        "status": status,
        "predictions": predictions
    }
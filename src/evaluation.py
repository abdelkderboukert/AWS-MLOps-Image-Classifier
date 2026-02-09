from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
import pandas as pd

def evaluate_classifier(name, model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(Y_test, Y_pred),
        "F1_micro": f1_score(Y_test, Y_pred, average="micro"),
        "Hamming_Loss": hamming_loss(Y_test, Y_pred)
    }

def calculate_nlp_metrics(references, hypotheses):
    # BLEU
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # CIDEr
    refs = {i: [" ".join(r[0])] for i, r in enumerate(references)}
    hyps = {i: [" ".join(h)] for i, h in enumerate(hypotheses)}
    
    cider = Cider()
    cider_score, _ = cider.compute_score(refs, hyps)
    
    return bleu_1, bleu_4, cider_score
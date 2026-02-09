import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from collections import Counter

from src.config import ZIP_PATH, EXTRACT_PATH, CSV_PATH, TEST_SIZE, RANDOM_STATE
from src.preprocessing import load_images_from_df, clean_text
from src.features import VisualFeatureExtractor
from src.models import train_and_tune
from src.evaluation import evaluate_classifier, calculate_nlp_metrics
from src.utils import get_predicted_keywords, generate_caption

def main():
    # 1. Setup Data
    if not os.path.exists(EXTRACT_PATH):
        print(f"Extracting {ZIP_PATH}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
    
    print("Loading Dataframe...")
    df = pd.read_csv(CSV_PATH)
    
    # 2. Preprocessing
    print("Preprocessing Images...")
    images = load_images_from_df(df)
    
    print("Preprocessing Text...")
    df["clean_tokens"] = df["text"].astype(str).apply(clean_text)
    
    # 3. Vocabulary & Labels
    all_tokens = [t for tokens in df["clean_tokens"] for t in tokens]
    vocab = [word for word, freq in Counter(all_tokens).items() if freq >= 5]
    mlb = MultiLabelBinarizer(classes=vocab)
    Y = mlb.fit_transform(df["clean_tokens"])
    
    # 4. Feature Extraction
    extractor = VisualFeatureExtractor()
    X = extractor.fit_transform(images)
    
    # 5. Split Data
    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
        X, Y, df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # 6. Train Models
    models_to_train = ["Logistic Regression", "Random Forest", "XGBoost"] # Add others as needed
    results = []
    trained_models = {}
    
    for name in models_to_train:
        model, _ = train_and_tune(X_train, Y_train, name)
        trained_models[name] = model
        metrics = evaluate_classifier(name, model, X_test, Y_test)
        results.append(metrics)
        
    results_df = pd.DataFrame(results)
    print("\nModel Performance:")
    print(results_df)
    
    # Select best model (simplified logic: max F1)
    best_model_name = results_df.sort_values("F1_micro", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\nBest Model selected: {best_model_name}")
    
    # 7. Generate Captions & Evaluate NLP Metrics
    print("Generating Captions for Test Set...")
    generated_captions = []
    references = [[df.loc[i, "text"].split()] for i in idx_test]
    
    for i in range(len(X_test)):
        keywords = get_predicted_keywords(best_model, X_test[i], mlb)
        caption = generate_caption(keywords)
        generated_captions.append(caption.split())
        
    bleu1, bleu4, cider = calculate_nlp_metrics(references, generated_captions)
    print(f"\nNLP Metrics -> BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}, CIDEr: {cider:.4f}")

if __name__ == "__main__":
    main()
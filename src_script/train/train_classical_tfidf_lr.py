import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import joblib
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_FILE = os.path.join(BASE_DIR, "data", "train_processed.parquet")
    VAL_FILE = os.path.join(BASE_DIR, "data", "val_processed.parquet")
    MODEL_DIR = os.path.join(BASE_DIR, "src_result", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if args.mode == "train":
        print("Loading data for TF-IDF + LR...")
        df = pd.read_parquet(TRAIN_FILE).sample(500000) # Use subset for speed
        
        # Handle NaN values in comment_text
        df['comment_text'] = df['comment_text'].fillna('')
        
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X = vectorizer.fit_transform(df['comment_text'])
        y = df['y_tox']
        
        print("Training Logistic Regression...")
        clf = LogisticRegression(C=1.0, solver='sag', n_jobs=-1)
        clf.fit(X, y)
        
        # Save model
        import joblib
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        joblib.dump(clf, os.path.join(MODEL_DIR, "lr_model.joblib"))
        print("TF-IDF + LR training completed.")
        
    else:
        import joblib
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        clf = joblib.load(os.path.join(MODEL_DIR, "lr_model.joblib"))
        
        val_df = pd.read_parquet(VAL_FILE)
        X_val = vectorizer.transform(val_df['comment_text'])
        y_val = val_df['y_tox']
        
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        print(f"TF-IDF + LR - F1: {f1_score(y_val, y_pred):.4f}, Acc: {accuracy_score(y_val, y_pred):.4f}")

if __name__ == "__main__":
    main()

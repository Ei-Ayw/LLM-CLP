import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
import random
import nltk
from tqdm import tqdm
import torch

# 尝试导入 transformers，用于回译
try:
    from transformers import MarianMTModel, MarianTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class DataAugmenter:
    """
    实现文档要求的四种数据增强方法：
    a. 同义词替换 (Synonym Replacement)
    b. 随机插入 (Random Insertion)
    c. 随机删除 (Random Deletion)
    d. 回译 (Back-Translation)
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.download_nltk_resources()
        self.bt_model_en_de = None
        self.bt_tok_en_de = None
        self.bt_model_de_en = None
        self.bt_tok_de_en = None

    def download_nltk_resources(self):
        resources = ['wordnet', 'punkt', 'averaged_perceptron_tagger', 'omw-1.4']
        for res in resources:
            try:
                nltk.data.find(f'corpora/{res}' if res in ['wordnet', 'omw-1.4'] else f'tokenizers/{res}' if res == 'punkt' else f'taggers/{res}')
            except LookupError:
                print(f"Downloading NLTK resource: {res}...")
                try:
                    nltk.download(res, quiet=True)
                except Exception as e:
                    print(f"  [Warning] Failed to download {res}: {e}")

    def load_bt_models(self):
        """Lazy load translation models to save memory if not used"""
        if self.bt_model_en_de is None and HAS_TRANSFORMERS:
            print("Loading Back-Translation Models (Helsinki-NLP)...")
            try:
                # En -> De
                self.bt_tok_en_de = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                self.bt_model_en_de = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(self.device).eval()
                # De -> En
                self.bt_tok_de_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
                self.bt_model_de_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(self.device).eval()
            except Exception as e:
                print(f"Failed to load translation models: {e}")
                self.bt_model_en_de = "FAILED"

    def get_synonyms(self, word):
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                if synonym != word:
                    synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement(self, words, n=1):
        """ a. 同义词替换 """
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in nltk.corpus.stopwords.words('english')]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return new_words

    def random_insertion(self, words, n=1):
        """ b. 随机插入 """
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words)-1)
        new_words.insert(random_idx, random_synonym)

    def random_deletion(self, words, p=0.1):
        """ c. 随机删除 """
        if len(words) == 1:
            return words
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            return [words[rand_int]]
        return new_words

    def back_translation(self, text):
        """ d. 回译 (En -> De -> En) """
        if not HAS_TRANSFORMERS: return text
        if self.bt_model_en_de is None: self.load_bt_models()
        if self.bt_model_en_de == "FAILED": return text

        try:
            # En -> De
            batch = self.bt_tok_en_de([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.bt_model_en_de.generate(**batch)
            de_text = self.bt_tok_en_de.batch_decode(gen, skip_special_tokens=True)[0]
            
            # De -> En
            batch = self.bt_tok_de_en([de_text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            gen = self.bt_model_de_en.generate(**batch)
            en_text = self.bt_tok_de_en.batch_decode(gen, skip_special_tokens=True)[0]
            return en_text
        except:
            return text

    def augment(self, text, method='random'):
        try:
            words = nltk.word_tokenize(text)
            # 过滤太短的句子
            if len(words) < 5: return text
            
            # 随机选择一种增强方式 (除了回译, 因为回译太慢，单独控制)
            if method == 'bt':
                return self.back_translation(text)
            
            ops = ['sr', 'ri', 'rd']
            op = random.choice(ops)
            
            if op == 'sr':
                return " ".join(self.synonym_replacement(words))
            elif op == 'ri':
                return " ".join(self.random_insertion(words))
            elif op == 'rd':
                return " ".join(self.random_deletion(words))
        except Exception:
            return text
        return text

def preprocess_data(input_path, output_dir, sample_size=None, seed=42, do_augment=True):
    print(f"Loading data from {input_path}...")
    
    identity_cols = [
        'male', 'female', 'black', 'white', 'muslim', 'jewish', 'christian', 
        'homosexual_gay_or_lesbian', 'psychiatric_or_mental_illness'
    ]
    subtype_cols = [
        'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'
    ]
    
    target_col = 'target'
    text_col = 'comment_text'
    use_cols = [text_col, target_col] + subtype_cols + identity_cols
    
    df = pd.read_csv(input_path, usecols=use_cols)
    
    if sample_size and 0 < sample_size < len(df):
        print(f"Sampling data down to {sample_size}...")
        df = df.sample(n=sample_size, random_state=seed)
    
    print("Processing labels...")
    df[identity_cols] = df[identity_cols].fillna(0)
    df[subtype_cols] = df[subtype_cols].fillna(0)
    df['has_identity'] = (df[identity_cols].max(axis=1) >= 0.5).astype(int)
    df['y_tox'] = (df[target_col] >= 0.5).astype(int)
    
    # Split
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=df['y_tox'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df['y_tox'])
    
    # === Data Augmentation Step ===
    if do_augment:
        print(">>> Starting Data Augmentation using [SR, RI, RD, BT]...")
        augmenter = DataAugmenter()
        
        # 筛选少数类 (Toxic)
        toxic_samples = train_df[train_df['y_tox'] == 1].copy()
        
        # 为了演示，我们只对一部分 Toxic 样本做增强，以免爆炸
        # 如果是服务器全量处理，可以更激进
        aug_list = []
        
        # 混合增强: 80% SR/RI/RD, 20% Back-Translation
        for idx, row in tqdm(toxic_samples.iterrows(), total=len(toxic_samples), desc="Augmenting Toxic Samples"):
            text = str(row[text_col])
            
            # 使用简单的 SR/RI/RD
            aug_text_1 = augmenter.augment(text, method='random')
            row_cp = row.copy()
            row_cp[text_col] = aug_text_1
            aug_list.append(row_cp)
            
            # 只有非常严重的样本才尝试回译 (因为慢)
            if random.random() < 0.1: # 10% chance for BT
                 aug_text_2 = augmenter.augment(text, method='bt')
                 if aug_text_2 != text:
                     row_cp2 = row.copy()
                     row_cp2[text_col] = aug_text_2
                     aug_list.append(row_cp2)

        if aug_list:
            aug_df = pd.DataFrame(aug_list)
            print(f"Generated {len(aug_df)} augmented samples.")
            train_df = pd.concat([train_df, aug_df], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
            
    print(f"Final Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(output_dir, 'train_processed.parquet'), index=False)
    val_df.to_parquet(os.path.join(output_dir, 'val_processed.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_dir, 'test_processed.parquet'), index=False)
    print(f"Preprocessing completed. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_aug", action="store_true", help="Disable augmentation")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_CSV = os.path.join(DATA_DIR, "train.csv")
    OUTPUT_DIR = DATA_DIR
    
    # 自动解压逻辑
    if not os.path.exists(INPUT_CSV):
        import zipfile
        zip_path = os.path.join(DATA_DIR, "jigsaw-unintended-bias-in-toxicity-classification.zip")
        if os.path.exists(zip_path):
            print(f"Detected dataset zip file. Unzipping {zip_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print("Unzip successful.")
            except Exception as e:
                print(f"Failed to unzip: {e}")
                exit(1)
        else:
            print(f"[Error] Data not found! Please upload 'train.csv' or the dataset zip file to: {DATA_DIR}")
            print(f"Expected path: {INPUT_CSV}")
            exit(1)
    
    preprocess_data(INPUT_CSV, OUTPUT_DIR, sample_size=args.sample_size, seed=args.seed, do_augment=not args.no_aug)

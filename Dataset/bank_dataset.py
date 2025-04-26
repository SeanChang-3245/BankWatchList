import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class CategoryEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, emb_mapping, embed_dim):
        self.emb_mapping = emb_mapping
        self.embed_dim    = embed_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.array(X).reshape(-1)
        out = np.zeros((len(arr), self.embed_dim), dtype=float)
        for i, v in enumerate(arr):
            out[i] = self.emb_mapping.get(int(v), np.zeros(self.embed_dim))
        return out

class BankTxnDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.ds      = cfg.dataset[split]
        df_txn  = pd.read_csv(self.ds['transaction'])
        df_acct = pd.read_csv(self.ds['accountInfo'])
        df_id   = pd.read_csv(self.ds['idInfo'])
        df_wl   = pd.read_csv(self.ds['watchList'])
        
        # Drop rows where OWN_TRANS_ID is 'ID99999'
        # df_txn = df_txn[df_txn['OWN_TRANS_ID'] != 'ID99999']

        # 1) Pre‑rename *all* but the key so there’s no duplicate key column
        acct_main = df_acct.rename(
            columns={c: f"{c}_MAIN" for c in df_acct.columns}
        )
        acct_own = df_acct.rename(
            columns={c: f"{c}_OWN" for c in df_acct.columns}
        )
        id_main = df_id.rename(
            columns={c: f"{c}_MAIN" for c in df_id.columns}
        )
        id_own = df_id.rename(
            columns={c: f"{c}_OWN" for c in df_id.columns}
        )

        df = (
            df_txn
            .merge(acct_main, left_on="ACCT_NBR", right_on="ACCT_NBR_MAIN", how="left")
            .merge(acct_own, left_on="OWN_TRANS_ACCT", right_on="ACCT_NBR_OWN", how="left")
            .merge(id_main, left_on="CUST_ID", right_on="CUST_ID_MAIN", how="left")
            .merge(id_own, left_on="OWN_TRANS_ID", right_on="CUST_ID_OWN", how="left")
        )
        
        df = df.drop(columns=[
            'ACCT_NBR_MAIN',
            'ACCT_NBR_OWN',
            'CUST_ID_MAIN_x',
            'CUST_ID_MAIN_y',
            'CUST_ID_OWN_x',
            'CUST_ID_OWN_y'
        ])
        
        date_cols = ['TX_DATE', 'ACCT_OPEN_DT_MAIN', 'ACCT_OPEN_DT_OWN', 'DATE_OF_BIRTH_MAIN', 'DATE_OF_BIRTH_OWN']
        min_date = df[date_cols].min().min()
        max_date = df[date_cols].max().max()
        
        # 3) derive binary label per transaction from watch‑list (by CUST_ID)
        watch_set = set(df_wl['ACCT_NBR'])
        df['LABEL'] = df['ACCT_NBR'].isin(watch_set).astype(int)

        # 4) pick & order features (your real column names here)
        numeric_cols = cfg.dataset["numericalCols"]                          
        categorical_cols = cfg.dataset["categoricalCols"]                    
        feature_cols = numeric_cols + categorical_cols
        
        ######################### TESTING #################################
        # df = df.head()
        ######################### TESTING #################################
        
        # 0) load YAML explanations
        with open(cfg.dataset["categoryEmbedding"]) as f:
            cat_map = yaml.safe_load(f)

        # 1) init text‐embedder
        txt_model = SentenceTransformer(cfg.model["textEmbeddingModel"])
        emb_dim   = txt_model.get_sentence_embedding_dimension()

        # 2) build per‐column {cat_value → vector}
        cat_emb_dict = {}
        for col in categorical_cols:
            info = cat_map.get(col)
            if info:
                m = {}
                for c, exp in zip(info["category"], info["explain"]):
                    m[int(c)] = txt_model.encode(exp, show_progress_bar=False)
                cat_emb_dict[col] = m

        # 3) pipelines
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale",   StandardScaler())
        ])

        cat_transformers = []
        for col in categorical_cols:
            if col in cat_emb_dict:
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                    ("embed",  CategoryEmbedder(cat_emb_dict[col], emb_dim))
                ])
                cat_transformers.append((f"emb_{col}", pipe, [col]))
            else:
                # fallback to one‑hot
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ])
                cat_transformers.append((f"ohe_{col}", pipe, [col]))

        self.ct = ColumnTransformer(
            [("num", num_pipeline, numeric_cols)] + cat_transformers,
            remainder="drop"
        )

        # 4) fit/transform as before…
        if split=="train":
            X_all = self.ct.fit_transform(df[ numeric_cols + categorical_cols ])
        else:
            X_all = self.ct.transform(   df[ numeric_cols + categorical_cols ])

        # if it's sparse, make it dense
        if hasattr(X_all, "toarray"):
            X_all = X_all.toarray()

        y_all = df['LABEL'].values
        df['row_idx'] = np.arange(len(df))
        
        # 6) group into sequences per ACCT_NBR
        sequences = []
        for acct, grp in df.groupby("ACCT_NBR"):
            grp = grp.sort_values('TX_DATE')
            idxs = grp['row_idx'].to_numpy()
            feats = X_all[idxs].astype(np.float32)               # (seq_len, feat_dim)
            lbl   = int(y_all[idxs[0]])                          # same label for whole sequence
            sequences.append((torch.from_numpy(feats), torch.tensor(lbl, dtype=torch.float32)))
        
        self.data = sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns: seq_tensor of shape (seq_len, feat_dim), label float tensor
        return self.data[idx]
    
    def get_label(self):
        df_wl   = pd.read_csv(self.ds['watchList'])
        df_txn  = pd.read_csv(self.ds['transaction'])
        watch_set = set(df_wl['ACCT_NBR'])
        acct_list = sorted(df_txn['ACCT_NBR'].unique())
        df_label = pd.DataFrame({
            'ACCT_NBR': acct_list,
            'in_watchlist': [1 if acct in watch_set else 0
                             for acct in acct_list]
        })

        return df_label


# --- pad‐and‐batch function for your DataLoader ---
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    # batch: list of (seq, label)
    seqs, labels = zip(*batch)
    # pad seqs to (batch, max_seq_len, feat_dim)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    labels  = torch.stack(labels)
    return padded, lengths, labels


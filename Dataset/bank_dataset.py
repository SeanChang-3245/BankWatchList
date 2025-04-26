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

# Update CategoryEmbedder to explicitly handle NaN values 

class CategoryEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, emb_mapping, embed_dim):
        self.emb_mapping = emb_mapping
        self.embed_dim = embed_dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.array(X).reshape(-1)
        out = np.zeros((len(arr), self.embed_dim), dtype=float)
        for i, v in enumerate(arr):
            # Handle NaN or convert to int safely
            if pd.isna(v):
                category_key = 0  # Map NaN to category 0
            else:
                try:
                    category_key = int(v)
                except (ValueError, TypeError):
                    category_key = 0  # Fallback for non-numeric values
            
            # Get embedding or use zero vector if not found
            out[i] = self.emb_mapping.get(category_key, np.zeros(self.embed_dim))
        return out

class BankTxnDataset(Dataset):
    # Class variable to store feature dimension after first fit
    feature_dim = None
    # Store fitted transformation pipeline
    fitted_transformer = None
    
    def __init__(self, cfg, split="train", val_ratio=0.2, random_seed=42):
        """
        Initialize dataset for train, validation, or test
        
        Args:
            cfg: Configuration object
            split: "train", "val", or "test" 
            val_ratio: Portion of training data to use for validation (0-1)
            random_seed: For reproducible validation splits
        """
        self.split = split
        self.random_seed = random_seed
        
        # If validation set, we actually need to load training data first
        data_split = "train" if split == "val" else split
        
        self.ds = cfg.dataset[data_split]
        df_txn = pd.read_csv(self.ds['transaction'])
        df_acct = pd.read_csv(self.ds['accountInfo'])
        df_id = pd.read_csv(self.ds['idInfo'])
        df_wl = pd.read_csv(self.ds['watchList'])
        
        
        # Pre-rename all but the key so there’s no duplicate key column
        acct_main = df_acct.rename(columns={c: f"{c}_MAIN" for c in df_acct.columns})
        acct_own = df_acct.rename(columns={c: f"{c}_OWN" for c in df_acct.columns})
        id_main = df_id.rename(columns={c: f"{c}_MAIN" for c in df_id.columns})
        id_own = df_id.rename(columns={c: f"{c}_OWN" for c in df_id.columns})
        
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
        
        # map DAY_OF_WEEK to int
        if 'DAY_OF_WEEK' in df.columns:
            # Get unique day values and create a mapping dictionary
            unique_days = df['DAY_OF_WEEK'].dropna().unique()
            day_to_int = {day: idx for idx, day in enumerate(sorted(unique_days))}        
            # Map string values to integers
            df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].map(day_to_int).fillna(0).astype(int)
                
        # Derive binary label per transaction from watch-list (by CUST_ID)
        watch_set = set(df_wl['ACCT_NBR'])
        df['LABEL'] = df['ACCT_NBR'].isin(watch_set).astype(int)
    
        # Pick & order features (your real column names here)
        numeric_cols = cfg.dataset["numericalCols"]
        categorical_cols = cfg.dataset["categoricalCols"]
        feature_cols = numeric_cols + categorical_cols
        date_cols = ['TX_DATE', 'ACCT_OPEN_DT_MAIN', 'ACCT_OPEN_DT_OWN', 'DATE_OF_BIRTH_MAIN', 'DATE_OF_BIRTH_OWN']
        
        # Testing
        # df = df.head(10000)
        
        # Load YAML explanations
        with open(cfg.dataset["categoryEmbedding"]) as f:
            cat_map = yaml.safe_load(f)
        
        # Init text-embedder
        txt_model = SentenceTransformer(cfg.model["textEmbeddingModel"])
        emb_dim   = txt_model.get_sentence_embedding_dimension()
        
        # Build per-column {cat_value → vector}
        cat_emb_dict = {}
        for col in categorical_cols:
            col_no_suffix = col
            if col.endswith('_MAIN') or col.endswith('_OWN'):
                col_no_suffix = col.rsplit('_', 1)[0]
            info = cat_map.get(col_no_suffix)
            if info:
                m = {}
                for c, exp in zip(info["category"], info["explain"]):
                    m[int(c)] = txt_model.encode(exp, show_progress_bar=False)
                cat_emb_dict[col] = m
        
        # Before creating the pipeline
        for col in categorical_cols:
            df[col] = df[col].fillna(0)
            if col in cat_emb_dict:
                # Ensure values are integers for embedding lookup
                df[col] = df[col].astype(int)
        
        # Handle categorical values before fitting to ensure consistency
        for col in categorical_cols:
            if col not in cat_emb_dict:
                # For one-hot columns, determine all possible values from config or data
                all_possible_values = cfg.dataset.get(f"all_values_{col}", None)
                if all_possible_values is None:
                    # If not specified, extract from training data
                    if split == "train":
                        # Get unique values and SORT them if they're numeric
                        values = df[col].dropna().unique()
                        # Check if values are numeric 
                        if pd.api.types.is_numeric_dtype(values):
                            all_possible_values = sorted(values.tolist())
                        else:
                            all_possible_values = values.tolist()
        
        # Pipelines
        num_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale",   StandardScaler())
        ])
        
        cat_transformers = []
        for col in categorical_cols:
            if col in cat_emb_dict:
                # For embedded categories, replace NaN with 0 then map to embeddings
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                    ("embed", CategoryEmbedder(cat_emb_dict[col], emb_dim))
                ])
                cat_transformers.append((f"emb_{col}", pipe, [col]))
            else:
                # For one-hot encoding, handle missing values as a separate category
                pipe = Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", 
                        sparse_output=False,
                        categories=[all_possible_values] if all_possible_values else "auto",
                        # Add this parameter to enforce sorting for safety
                        dtype=np.float64
                    ))
                ])
                cat_transformers.append((f"ohe_{col}", pipe, [col]))
        
        self.ct = ColumnTransformer(
            [("num", num_pipeline, numeric_cols)] + cat_transformers,
            remainder="drop"
        )
        
        print(df.shape)
        print(df.columns)
        
        # Ensure consistent feature dimensions
        if split == "train" or (split == "val" and BankTxnDataset.fitted_transformer is None):
            X_all = self.ct.fit_transform(df[numeric_cols + categorical_cols])
            # Store the feature dimension and transformer after first fit
            BankTxnDataset.feature_dim = X_all.shape[1]
            BankTxnDataset.fitted_transformer = self.ct
        else:
            # Use the transformer fitted on training data
            if BankTxnDataset.fitted_transformer is not None:
                self.ct = BankTxnDataset.fitted_transformer
            X_all = self.ct.transform(df[numeric_cols + categorical_cols])
            
        # If it's sparse, make it dense
        if hasattr(X_all, "toarray"):
            X_all = X_all.toarray()
        
        y_all = df['LABEL'].values
        df['row_idx'] = np.arange(len(df))
        
        # Group into sequences per ACCT_NBR
        sequences = []
        for acct, grp in df.groupby("ACCT_NBR"):
            grp = grp.sort_values('TX_DATE')
            idxs = grp['row_idx'].to_numpy()
            feats = X_all[idxs].astype(np.float32)               # (seq_len, feat_dim)
            lbl   = int(y_all[idxs[0]])                          # same label for whole sequence
            sequences.append((torch.from_numpy(feats), torch.tensor(lbl, dtype=torch.float32)))
        
        # Split sequences for validation if needed
        if split == "train" or split == "val":
            np.random.seed(random_seed)
            indices = np.random.permutation(len(sequences))
            split_idx = int(len(sequences) * (1 - val_ratio))
            
            if split == "train":
                # Use only training portion
                self.data = [sequences[i] for i in indices[:split_idx]]
            else:  # split == "val"
                # Use only validation portion
                self.data = [sequences[i] for i in indices[split_idx:]]
        else:  # split == "test"
            # Use all sequences for test set
            self.data = sequences
            
        print(f"Created {split} dataset with {len(self.data)} sequences")
    
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
    
    def get_featDim(self):
        """Return the feature dimension of the processed data"""
        return BankTxnDataset.feature_dim

# --- pad-and-batch function for your DataLoader ---
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    # batch: list of (seq, label)
    seqs, labels = zip(*batch)
    # pad seqs to (batch, max_seq_len, feat_dim)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    labels  = torch.stack(labels)
    return padded, lengths, labels


import torch
from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
from Models.transformer import TransformerClassifier
from Config.config import load_config
from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: \033[92m{device}\033[0m")
    
    train_ds = BankTxnDataset(cfg, split="train")
    print(f"Total number of training data: \033[92m{len(train_ds.data)}\033[0m")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.parameter['batchSize'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,                # speeds host→GPU copies
        collate_fn=pad_collate_fn
    )
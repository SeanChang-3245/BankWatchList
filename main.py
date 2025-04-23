import Dataset.dataset
import Models.transformer
from Config.config import load_config
from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn

if __name__ == "__main__":
    cfg = load_config()
    train_ds = BankTxnDataset(cfg)
    

    # from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
    # from Config.config import load_config
    # from torch.utils.data import DataLoader

    # cfg = load_config()
    # train_ds = BankTxnDataset(cfg, split="train")
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.parameter.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=pad_collate_fn
    # )
    # from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
    # from Config.config import load_config
    # from torch.utils.data import DataLoader

    # cfg = load_config()
    # train_ds = BankTxnDataset(cfg, split="train")
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.parameter.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=pad_collate_fn
    # )
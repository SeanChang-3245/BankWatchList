from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
import Models.transformer
from Config.config import load_config
from Dataset.bank_dataset import BankTxnDataset, pad_collate_fn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    cfg = load_config()
    train_ds = BankTxnDataset(cfg)
    
    print(type(train_ds.data))
    print(len(train_ds.data))
    print(train_ds[0][0].shape)

    # cfg = load_config()
    # train_ds = BankTxnDataset(cfg, split="train")
    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=cfg.parameter.batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=pad_collate_fn
    # )

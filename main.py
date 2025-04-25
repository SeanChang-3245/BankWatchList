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
    print(f"Total number of training data: {len(train_ds.data)}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.parameter['batchSize'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,                # speeds host→GPU copies
        collate_fn=pad_collate_fn
    )

    # 2) model + optimizer
    model = TransformerClassifier(
        feat_dim=train_ds.get_featDim(),
        d_model=cfg.parameter['d_model'],
        nhead=cfg.parameter['attention_head'],
        num_layers=cfg.parameter['num_layers'],
        num_classes=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.parameter['learningRate'])
    scaler    = torch.cuda.amp.GradScaler()  # optional mixed‑precision

    # 3) train loop
    model.train()
    for epoch in range(cfg.parameter['epochs']):
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():    # comment out for full precision
                logits = model(x, src_key_padding_mask=(torch.arange(x.size(1))
                                                       .unsqueeze(0)
                                                       .to(device)
                                                       >= lengths.unsqueeze(1)))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = texts            # list of raw strings   # CHANGE ME
        self.labels = labels          # list of ints          # CHANGE ME
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    masks     = torch.stack([b['attention_mask'] for b in batch])
    labels    = torch.stack([b['label'] for b in batch])
    # src_key_padding_mask expects True for pads
    src_key_padding_mask = masks == 0
    return input_ids, src_key_padding_mask, labels

# USAGE:
# tokenizer = your_tokenizer(e.g. from transformers import BertTokenizer; BertTokenizer.from_pretrained(...))
# texts, labels = load_your_data()      # CHANGE ME: load raw texts & labels
# dataset = TextDataset(texts, labels, tokenizer, max_len=128)
# loader  = DataLoader(
#     dataset,
#     batch_size=32,        # CHANGE ME: batch size
#     shuffle=True,
#     num_workers=4,        # CHANGE ME: num workers
#     collate_fn=collate_fn
# )
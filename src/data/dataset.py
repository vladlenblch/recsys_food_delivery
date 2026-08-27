import torch
from torch.utils.data import Dataset

class SessionDataset(Dataset):
    def __init__(self, train_prefixes):
        if isinstance(train_prefixes, dict):
            self.sessions = list(train_prefixes.values())
        else:
            self.sessions = train_prefixes

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        seq = self.sessions[idx]
        input_seq = seq[:-1]
        target_seq = seq[1:]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            len(input_seq)
        )

def collate_sessions(batch):
    inputs, targets, lengths = zip(*batch)
    max_len = max(lengths)
    batch_size = len(inputs)
    
    padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_targets = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i in range(batch_size):
        padded_inputs[i, :lengths[i]] = inputs[i]
        padded_targets[i, :lengths[i]] = targets[i]
    
    mask = torch.arange(max_len).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    
    return padded_inputs, padded_targets, torch.tensor(lengths, dtype=torch.long), mask

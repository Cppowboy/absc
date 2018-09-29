from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import torch


class SemEval2014(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.data_list = json.load(open(filename, 'r'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item['context_ids'], item['context_label'], item['aspect_mask'], item['aspect_pos'], item['polarity']


def collate(data):
    context_ids, context_label, aspect_mask, aspect_pos, polarity = zip(*data)
    context_ids, context_label, aspect_mask, aspect_pos, polarity = \
        torch.from_numpy(np.array(context_ids)).long(), \
        torch.from_numpy(np.array(context_label)).long(), \
        torch.from_numpy(np.array(aspect_mask)).float(), \
        torch.from_numpy(np.array(aspect_pos)).float(), \
        torch.from_numpy(np.array(polarity)).long()
    return context_ids, context_label, aspect_mask, aspect_pos, polarity


def get_loader(filename, batch_size):
    dataset = SemEval2014(filename)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate)
    return dataloader

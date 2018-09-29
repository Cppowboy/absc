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
        return item['sent_ids'], item['len'], item['sent_mask'], \
               item['aspect_ids'], item['aspect_len'], item['aspect_mask'], \
               item['polarity']


def collate(data):
    sent_ids, lens, sent_masks, aspect_ids, aspect_lens, aspect_masks, polarity = zip(*data)
    sent_ids, lens, sent_masks, aspect_ids, aspect_lens, aspect_masks, polarity = \
        torch.from_numpy(np.array(sent_ids)).long(), \
        torch.from_numpy(np.array(lens)).long(), \
        torch.from_numpy(np.array(sent_masks)).float(), \
        torch.from_numpy(np.array(aspect_ids)).long(), \
        torch.from_numpy(np.array(aspect_lens)).long(), \
        torch.from_numpy(np.array(aspect_masks)).float(), \
        torch.from_numpy(np.array(polarity)).long()
    return sent_ids, lens, sent_masks, aspect_ids, aspect_lens, aspect_masks, polarity


def get_loader(filename, batch_size):
    dataset = SemEval2014(filename)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate)
    return dataloader

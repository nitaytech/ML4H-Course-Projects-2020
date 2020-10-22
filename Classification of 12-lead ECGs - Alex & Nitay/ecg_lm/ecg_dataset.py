import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from ecg_lm.configs import *
import os


class ECGDataset(Dataset):
    def __init__(self, headers_pdf: pd.DataFrame, n_segs: int = None, n_qrs: int = None, abbs_codes: list = None):
        self.headers = headers_pdf
        if abbs_codes is None:
            abbs_codes = headers_pdf.columns.tolist()
        self.abbs_codes = [c for c in abbs_codes if c in ABBS]
        if n_segs is not None:
            self.headers = self.headers[self.headers['n_segs'] >= n_segs]
            self.n_segs = n_segs
        else:
            self.n_segs = self.headers['n_segs'].min()
        if n_qrs is not None:
            self.headers = self.headers[self.headers['n_qrs'] >= n_qrs]
            self.n_qrs = n_qrs
        else:
            self.n_qrs = self.headers['n_qrs'].min()

    def __len__(self):
        return len(self.headers)

    def __getitem__(self, item):
        header = dict(self.headers.iloc[item])
        seg_folder, seg_file, qrs_file = header['seg_folder'], header['seg_file'], header['qrs_file']
        segments = np.load(os.path.join(seg_folder, seg_file))[:, :self.n_segs, :]
        qrs_segments = np.load(os.path.join(seg_folder, qrs_file))[:, :self.n_qrs, :]
        ecg = {'age': header['Age'], 'gender': header['Male'], 'n_qrs': header['n_qrs'],
               'heart_rate': header['heart_rate'], 'segments': segments, 'qrs_segments': qrs_segments}
        ecg.update({abbs: header.get(abbs, False) for abbs in self.abbs_codes})
        return ecg

    def ecg2tensors(self, ecg):  # ecg = {feature: tensor([val_u1, val_u2, ...])}
        demographic = torch.stack([ecg[c].type(torch.float32) for c in ['age', 'gender', 'n_qrs', 'heart_rate']], dim=1)  # shape(batch_size, 4)
        segments = ecg['segments'].type(torch.float32)  # shape(batch_size, 12, n_segs, seg_len)
        qrs_segments = ecg['qrs_segments'].type(torch.float32)  # shape(batch_size, 12, n_qrs, qrs_seg_len)
        abnormalities = torch.stack([ecg[abbs].type(torch.float32) for abbs in self.abbs_codes], dim=1)  # shape(batch_size, n_abbs)
        return demographic, segments, qrs_segments, abnormalities

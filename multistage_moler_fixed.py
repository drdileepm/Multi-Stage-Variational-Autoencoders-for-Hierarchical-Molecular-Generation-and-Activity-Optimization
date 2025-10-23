# multistage_moler_fixed.py

"""
Two-stage hierarchical VAE for molecular generation + activity optimization.

Stage-1: Fingerprint VAE (coarse representation)
Stage-2: SMILES conditional VAE (conditioned on stage-1 latent)
Property predictor on stage-2 latent for activity optimization.
"""

import os
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem

# -------------------------
# Utilities
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smiles_to_morgan(smiles: str, radius: int=2, n_bits: int=2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def canonical_smiles(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None

# -------------------------
# Dataset for SMILES + fingerprints + activity (if present)
# -------------------------
class MoleculeDataset(Dataset):
    def __init__(self, smiles_list: List[str], activities: List[float]=None,
                 fp_bits: int=2048, max_len: int=120, charset: List[str]=None):
        self.raw_smiles = [canonical_smiles(s) for s in smiles_list]
        self.data = [s for s in self.raw_smiles if s is not None]
        self.activities = activities
        if activities is not None:
            self.activities = [a for s,a in zip(self.raw_smiles, activities) if s is not None]

        self.fp_bits = fp_bits
        self.max_len = max_len

        # BLUE: Corrected SMILES tokenization (multi-char tokens, chirality, stereochemistry)
        if charset is None:
            charset = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "b", "c", "n", "o", "p",
                       "[", "]", "(", ")", "=", "#", "1","2","3","4","5","6","7","8","9","+", "-", "/",
                       "\\", "@", "s"]
            charset = sorted(set(charset), key=lambda x: (len(x), x))
            charset = ["<pad>", "<sos>", "<eos>", "<unk>"] + charset
        self.charset = charset
        self.char2idx = {c:i for i,c in enumerate(self.charset)}
        self.idx2char = {i:c for c,i in self.char2idx.items()}

    def __len__(self):
        return len(self.data)

    # BLUE: handle <sos>, <eos>, padding correctly
    def smiles_to_tensor(self, s: str) -> torch.LongTensor:
        chars = ["<sos>"] + list(s) + ["<eos>"]
        ids = [self.char2idx.get(c, self.char2idx["<unk>"]) for c in chars]
        if len(ids) < self.max_len:
            ids = ids + [self.char2idx["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len-1] + [self.char2idx["<eos>"]]
        return torch.LongTensor(ids)

    def __getitem__(self, idx):
        s = self.data[idx]
        fp = smiles_to_morgan(s, n_bits=self.fp_bits)
        smiles_tensor = self.smiles_to_tensor(s)
        if self.activities is not None:
            return smiles_tensor, torch.tensor(fp, dtype=torch.float32), torch.tensor(self.activities[idx], dtype=torch.float32)
        else:
            return smiles_tensor, torch.tensor(fp, dtype=torch.float32)

# -------------------------
# Stage-1: Fingerprint VAE
# -------------------------
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hid_dim, z_dim)
        self.logvar = nn.Linear(hid_dim, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, hid_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class FingerprintVAE(nn.Module):
    def __init__(self, in_dim=2048, hid_dim=1024, z_dim=64):
        super().__init__()
        self.enc = MLPEncoder(in_dim, hid_dim, z_dim)
        self.dec = MLPDecoder(z_dim, hid_dim, in_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar, z

# -------------------------
# Stage-2: Conditional SMILES VAE
# -------------------------
class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, z_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.hidden2mu = nn.Linear(hidden_dim, z_dim)
        self.hidden2logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x, lengths=None):
        emb = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, c) = self.lstm(packed)
        else:
            out, (h, c) = self.lstm(emb)
        h_last = h[-1]
        return self.hidden2mu(h_last), self.hidden2logvar(h_last)

class CondSeqDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, z_dim, cond_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim + z_dim + cond_dim, hidden_dim, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, z, cond, teacher_forcing_ratio=0.5):
        B, T = x.size()
        emb = self.embedding(x)
        z_ex = z.unsqueeze(1).expand(-1, T, -1)
        c_ex = cond.unsqueeze(1).expand(-1, T, -1)
        lstm_in = torch.cat([emb, z_ex, c_ex], dim=-1)
        out, _ = self.lstm(lstm_in)
        logits = self.hidden2out(out)
        return logits

class SeqVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=256, z_dim=64, cond_dim=64, pad_idx=0):
        super().__init__()
        self.encoder = SeqEncoder(vocab_size, emb_dim, hidden_dim, z_dim, pad_idx=pad_idx)
        self.decoder = CondSeqDecoder(vocab_size, emb_dim, hidden_dim, z_dim, cond_dim, pad_idx=pad_idx)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_tokens, cond_vec):
        mu, logvar = self.encoder(x_tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(x_tokens, z, cond_vec)
        return logits, mu, logvar, z

    def sample(self, cond_vec, sos_idx, eos_idx, z_opt=None, max_len=120):
        B = cond_vec.size(0)
        device = cond_vec.device
        z = z_opt if z_opt is not None else torch.randn(B, self.z_dim, device=device)
        inputs = torch.full((B,1), sos_idx, dtype=torch.long, device=device)
        sequences = inputs
        hidden = None
        for t in range(max_len-1):
            emb = self.decoder.embedding(inputs)
            z_ex = z.unsqueeze(1)
            c_ex = cond_vec.unsqueeze(1)
            lstm_in = torch.cat([emb, z_ex, c_ex], dim=-1)
            out, hidden = self.decoder.lstm(lstm_in, hidden)
            logits = self.decoder.hidden2out(out)
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_tokens = probs.argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_tokens], dim=1)
            inputs = next_tokens
            if next_tokens.item() == eos_idx:
                break
        return sequences

class PropertyPredictor(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# -------------------------
# Latent optimization: maximize property predictor wrt z
# -------------------------
def optimize_latent_for_activity(init_z: torch.Tensor, cond_vec: torch.Tensor,
                                 predictor: PropertyPredictor, seq_vae: SeqVAE,
                                 steps: int=100, lr: float=1e-2, pad_idx=0, sos_idx=1, eos_idx=2):
    z_opt = init_z.clone().detach().to(DEVICE)
    z_opt.requires_grad = True
    opt = torch.optim.Adam([z_opt], lr=lr)
    predictor.to(DEVICE)
    seq_vae.to(DEVICE)
    predictor.eval()
    seq_vae.eval()
    for i in range(steps):
        opt.zero_grad()
        pred = predictor(z_opt)
        loss = -pred.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        seq_tokens = seq_vae.sample(cond_vec.to(DEVICE), sos_idx=sos_idx, eos_idx=eos_idx, z_opt=z_opt, max_len=120)
    return seq_tokens

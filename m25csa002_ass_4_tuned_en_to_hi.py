import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import pickle
import time
import nltk
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

print("PyTorch version:", torch.__version__)
print("Ray version:", ray.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Output directory for all saved visualizations ────────────────────────────
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Visualizations will be saved to: {os.path.abspath(PLOTS_DIR)}/")

"""## Data Loading & Preprocessing"""

# Load the TSV dataset
file_id = '1dPWcMzr0H5utKjqa-HUod1_QhwXSsJfI'
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url, sep='\t', header=None, names=["id1", "en", "id2", "hi"])
df = df[["en", "hi"]]
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Total sentence pairs: {len(df)}")
df.head()

df["en_len"] = df["en"].apply(lambda x: len(x.split()))
df["hi_len"] = df["hi"].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["en_len"], bins=30, kde=True, color='skyblue', ax=axes[0])
axes[0].set_title("English Sentence Lengths"); axes[0].set_xlabel("Words")
sns.histplot(df["hi_len"], bins=30, kde=True, color='salmon', ax=axes[1])
axes[1].set_title("Hindi Sentence Lengths"); axes[1].set_xlabel("Words")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_sentence_length_distribution.png"), dpi=150, bbox_inches="tight")
print("Saved: 01_sentence_length_distribution.png")
plt.show()

print("\nEnglish stats:"); print(df["en_len"].describe())
print("\nHindi stats:"); print(df["hi_len"].describe())

"""## Vocabulary Building"""

class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx = 4

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1

    def tokenize(self, sentence):
        return sentence.lower().strip().split()

    def numericalize(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi["<unk>"])

# Build vocabularies
en_vocab = Vocabulary(freq_threshold=2)
hi_vocab = Vocabulary(freq_threshold=2)
en_vocab.build_vocab(df["en"].tolist())
hi_vocab.build_vocab(df["hi"].tolist())
print(f"English vocab size: {len(en_vocab.stoi)}")
print(f"Hindi vocab size:   {len(hi_vocab.stoi)}")

SRC_PAD_IDX = en_vocab["<pad>"]
TGT_PAD_IDX = hi_vocab["<pad>"]

def encode_sentence(sentence, vocab, max_len=50):
    tokens = [vocab.stoi["<sos>"]] + vocab.numericalize(sentence)[:max_len-2] + [vocab.stoi["<eos>"]]
    return tokens + [vocab.stoi["<pad>"]] * (max_len - len(tokens))

"""## Dataset & DataLoader"""

class TranslationDataset(Dataset):
    def __init__(self, df, en_vocab, hi_vocab, max_len=50):
        self.en_sentences = df["en"].tolist()
        self.hi_sentences = df["hi"].tolist()
        self.en_vocab = en_vocab
        self.hi_vocab = hi_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        src = encode_sentence(self.en_sentences[idx], self.en_vocab, self.max_len)
        tgt = encode_sentence(self.hi_sentences[idx], self.hi_vocab, self.max_len)
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    tgt_input  = tgt_batch[:, :-1]
    tgt_output = tgt_batch[:, 1:]
    return src_batch, tgt_input, tgt_output

"""## Transformer Architecture (from scratch)"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear   = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear   = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.query_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output  = torch.matmul(self.dropout(attn_weights), V)
        attn_output  = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(target_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers  = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.embed(x)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, max_len=100, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_subsequent_mask(self, size):
        return torch.tril(torch.ones((size, size))).bool().to(next(self.parameters()).device)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx):
        src_mask     = self.make_pad_mask(src, src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, tgt_pad_idx)
        tgt_sub_mask = self.make_subsequent_mask(tgt.size(1))
        tgt_mask     = tgt_pad_mask & tgt_sub_mask
        enc_out      = self.encoder(src, src_mask)
        dec_out      = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

"""---
## Part 1 — Baseline Training (100 epochs, hardcoded hyperparameters)
"""

# ── PART 1: BASELINE ────────────────────────────────────────────────────────
BASELINE_BATCH_SIZE = 60
BASELINE_LR         = 1e-4
BASELINE_EPOCHS     = 100
MAX_LEN             = 50
D_MODEL             = 512

baseline_dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
baseline_loader  = DataLoader(baseline_dataset, batch_size=BASELINE_BATCH_SIZE,
                               shuffle=True, collate_fn=collate_fn)

baseline_model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=D_MODEL, num_layers=6, num_heads=8,
    d_ff=2048, max_len=MAX_LEN, dropout=0.1
).to(DEVICE)

baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=BASELINE_LR)
baseline_criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

def save_checkpoint(epoch, model, optimizer, loss, path="checkpoint.pt"):
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'loss': loss}, path)
    print(f"Checkpoint saved at epoch {epoch}, loss {loss:.4f}.")

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']} with loss {ckpt['loss']:.4f}")
        return ckpt['epoch']
    print(f"No checkpoint found at {path}. Starting from scratch.")
    return 0

def train_baseline(model, train_loader, optimizer, criterion,
                   start_epoch=0, num_epochs=BASELINE_EPOCHS, checkpoint_path="checkpoint.pt"):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for src, tgt_input, tgt_output in loop:
            src, tgt_input, tgt_output = src.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)
            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(train_loader)
        save_checkpoint(epoch+1, model, optimizer, avg_loss, checkpoint_path)
    return avg_loss

start_epoch = load_checkpoint(baseline_model, baseline_optimizer)
t0 = time.time()
final_baseline_loss = train_baseline(baseline_model, baseline_loader, baseline_optimizer, baseline_criterion, start_epoch=start_epoch)
print(f"\nBaseline training time: {(time.time()-t0)/60:.1f} minutes")
print(f"Final avg loss: {final_baseline_loss:.4f}")
torch.save(baseline_model.state_dict(), "transformer_translation_final.pth")

"""## Evaluation Helpers (BLEU)"""

def translate_sentence(model, sentence, en_vocab, hi_vocab, max_len=50, device=DEVICE):
    model.eval()
    tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tokens = [hi_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, SRC_PAD_IDX, TGT_PAD_IDX)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break
    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]]
    return ' '.join(translated)


smoothie = SmoothingFunction().method4

VAL_DATASET = [
    ("I love you.",              "मैं तुमसे प्यार करता हूँ।"),
    ("How are you?",             "आप कैसे हैं?"),
    ("You should sleep.",        "आपको सोना चाहिए।"),
    ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
    ("Let me tell Tom.",         "मुझे टॉम को बताने दीजिए।"),
]

def evaluate_bleu(model, dataset=VAL_DATASET, en_vocab=en_vocab, hi_vocab=hi_vocab,
                  max_len=50, device=DEVICE):
    references, hypotheses = [], []
    for en_sent, hi_sent in dataset:
        pred = translate_sentence(model, en_sent, en_vocab, hi_vocab, max_len, device)
        hypotheses.append(pred.split())
        references.append([hi_sent.split()])
    score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    print(f"🌐 BLEU Score (NLTK): {score * 100:.2f}")
    return score

# To evaluate baseline (after training):
baseline_score = evaluate_bleu(baseline_model)


# ── PART 2: RAY TUNE TRAINING FUNCTION ───────────────────────────────────────

def train_tune(config):
    """
    Ray Tune-compatible training function.
    Accepts a config dict with all hyperparameters.
    Reports loss after each epoch via ray.train.report().

    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import math, os
    import pandas as pd
    from collections import Counter
    from torch.utils.data import Dataset, DataLoader
    import ray

    lr         = config["lr"]
    batch_size = config["batch_size"]
    num_heads  = config["num_heads"]
    d_ff       = config["d_ff"]
    dropout    = config["dropout"]
    num_layers = config["num_layers"]
    num_epochs = config.get("num_epochs", 30)
    DATA_PATH  = config.get("data_path", url)
    D_MODEL    = 512   # fixed; 512 % 4 == 0 and 512 % 8 == 0 ✓
    MAX_LEN    = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Vocabulary (self-contained copy) ────────────────────────────────────
    class _Vocab:
        def __init__(self, freq_threshold=2):
            self.freq_threshold = freq_threshold
            self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
            self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
            self.idx  = 4
        def build_vocab(self, sentences):
            freq = Counter()
            for s in sentences:
                for w in s.lower().strip().split():
                    freq[w] += 1
            for w, f in freq.items():
                if f >= self.freq_threshold:
                    self.stoi[w] = self.idx
                    self.itos[self.idx] = w
                    self.idx += 1
        def numericalize(self, sentence):
            return [self.stoi.get(w, self.stoi["<unk>"])
                    for w in sentence.lower().strip().split()]
        def __len__(self):       return len(self.stoi)
        def __getitem__(self, t): return self.stoi.get(t, self.stoi["<unk>"])

    def _encode(sentence, vocab, max_len=50):
        tokens = ([vocab["<sos>"]]
                  + vocab.numericalize(sentence)[:max_len - 2]
                  + [vocab["<eos>"]])
        return tokens + [vocab["<pad>"]] * (max_len - len(tokens))

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH, sep="\t", header=None,
                     names=["id1", "en", "id2", "hi"])
    df = df[["en", "hi"]].dropna().reset_index(drop=True)

    _en_v = _Vocab(); _hi_v = _Vocab()
    _en_v.build_vocab(df["en"].tolist())
    _hi_v.build_vocab(df["hi"].tolist())

    src_pad = _en_v["<pad>"]
    tgt_pad = _hi_v["<pad>"]

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    class _Dataset(Dataset):
        def __init__(self):
            self.en = df["en"].tolist()
            self.hi = df["hi"].tolist()
        def __len__(self): return len(self.en)
        def __getitem__(self, i):
            return (torch.tensor(_encode(self.en[i], _en_v, MAX_LEN)),
                    torch.tensor(_encode(self.hi[i], _hi_v, MAX_LEN)))

    def _collate(batch):
        s, t = zip(*batch)
        s = torch.stack(s)          # (B, MAX_LEN)
        t = torch.stack(t)          # (B, MAX_LEN)
        # contiguous() ensures reshape() never fails on non-contiguous slices
        tgt_in  = t[:, :-1].contiguous()
        tgt_out = t[:, 1:].contiguous()
        return s, tgt_in, tgt_out

    loader = DataLoader(_Dataset(), batch_size=batch_size,
                        shuffle=True, collate_fn=_collate, num_workers=0)

    # ── Transformer architecture (self-contained copy) ───────────────────────
    class _PE(nn.Module):
        def __init__(self, d, maxlen=5000):
            super().__init__()
            pe  = torch.zeros(maxlen, d)
            pos = torch.arange(0, maxlen).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))
        def forward(self, x): return x + self.pe[:, :x.size(1)]

    class _MHA(nn.Module):
        def __init__(self, d, h, drop):
            super().__init__()
            self.h = h; self.dk = d // h
            self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d)
            self.Wv = nn.Linear(d, d); self.Wo = nn.Linear(d, d)
            self.drop = nn.Dropout(drop)
        def forward(self, q, k, v, mask=None):
            B = q.size(0)
            Q = self.Wq(q).view(B, -1, self.h, self.dk).transpose(1, 2)
            K = self.Wk(k).view(B, -1, self.h, self.dk).transpose(1, 2)
            V = self.Wv(v).view(B, -1, self.h, self.dk).transpose(1, 2)
            sc = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)
            if mask is not None: sc = sc.masked_fill(mask == 0, -1e9)
            aw = self.drop(torch.softmax(sc, dim=-1))
            out = torch.matmul(aw, V).transpose(1, 2).contiguous()
            return self.Wo(out.view(B, -1, self.h * self.dk))

    class _FF(nn.Module):
        def __init__(self, d, dff, drop):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, dff), nn.ReLU(),
                                     nn.Dropout(drop), nn.Linear(dff, d))
        def forward(self, x): return self.net(x)

    class _LN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.g = nn.Parameter(torch.ones(d))
            self.b = nn.Parameter(torch.zeros(d))
        def forward(self, x):
            m = x.mean(-1, keepdim=True); s = x.std(-1, keepdim=True)
            return self.g * (x - m) / (s + 1e-6) + self.b

    class _ENC(nn.Module):
        def __init__(self, vsz, d, nl, nh, dff, mlen, drop):
            super().__init__()
            self.emb  = nn.Embedding(vsz, d)
            self.pe   = _PE(d, mlen)
            self.drop = nn.Dropout(drop)
            self.layers = nn.ModuleList([
                nn.ModuleList([_MHA(d,nh,drop), _FF(d,dff,drop), _LN(d), _LN(d)])
                for _ in range(nl)])
        def forward(self, x, mask=None):
            x = self.drop(self.pe(self.emb(x)))
            for attn, ff, ln1, ln2 in self.layers:
                x = ln1(x + attn(x, x, x, mask))
                x = ln2(x + ff(x))
            return x

    class _DEC(nn.Module):
        def __init__(self, vsz, d, nl, nh, dff, mlen, drop):
            super().__init__()
            self.emb  = nn.Embedding(vsz, d)
            self.pe   = _PE(d, mlen)
            self.drop = nn.Dropout(drop)
            self.layers = nn.ModuleList([
                nn.ModuleList([_MHA(d,nh,drop), _MHA(d,nh,drop),
                               _FF(d,dff,drop), _LN(d), _LN(d), _LN(d)])
                for _ in range(nl)])
        def forward(self, x, enc, smask=None, tmask=None):
            x = self.drop(self.pe(self.emb(x)))
            for sa, ca, ff, ln1, ln2, ln3 in self.layers:
                x = ln1(x + sa(x, x, x, tmask))
                x = ln2(x + ca(x, enc, enc, smask))
                x = ln3(x + ff(x))
            return x

    class _Transformer(nn.Module):
        def __init__(self, sv, tv, d, nl, nh, dff, mlen, drop):
            super().__init__()
            self.enc = _ENC(sv, d, nl, nh, dff, mlen, drop)
            self.dec = _DEC(tv, d, nl, nh, dff, mlen, drop)
            self.fc  = nn.Linear(d, tv)
        def _pad_mask(self, s, pid):
            return (s != pid).unsqueeze(1).unsqueeze(2)
        def _sub_mask(self, sz):
            return torch.tril(torch.ones(sz, sz)).bool().to(
                next(self.parameters()).device)
        def forward(self, src, tgt, sp, tp):
            sm  = self._pad_mask(src, sp)
            tpm = self._pad_mask(tgt, tp)
            tsm = self._sub_mask(tgt.size(1))
            tm  = tpm & tsm
            return self.fc(self.dec(tgt, self.enc(src, sm), sm, tm))

    model = _Transformer(len(_en_v), len(_hi_v), D_MODEL,
                         num_layers, num_heads, d_ff, MAX_LEN, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for src, tgt_in, tgt_out in loader:
            src    = src.to(device)
            tgt_in = tgt_in.to(device)
            tgt_out = tgt_out.to(device)

            out  = model(src, tgt_in, src_pad, tgt_pad)
            # reshape() is safe on both contiguous and non-contiguous tensors
            loss = criterion(out.reshape(-1, out.shape[-1]),
                             tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        tune.report({"loss": avg_loss, "epoch": epoch + 1})

print("✅ train_tune() defined successfully.")

# ── PART 2.2 + 2.3: Configure & Run Hyperparameter Search ──────────────────

# 1. Search space — 6 hyperparameters tuned
search_space = {
    # (1) Learning rate — log scale recommended for optimizers
    "lr":         tune.loguniform(1e-5, 1e-3),

    # (2) Batch size — larger = noisier gradients but faster epochs
    "batch_size": tune.choice([32, 64, 128]),

    # (3) Attention heads — 512 % 4 == 0, 512 % 8 == 0  ✓
    "num_heads":  tune.choice([4, 8]),

    # (4) Feed-forward dimension — controls model capacity
    "d_ff":       tune.choice([1024, 2048, 4096]),

    # (5) Dropout — regularization strength
    "dropout":    tune.uniform(0.05, 0.4),

    # (6) Number of encoder/decoder layers
    "num_layers": tune.choice([4, 6]),

    # Fixed params passed via config
    "num_epochs": 30,
    "data_path":  url,
}

# 2. Optuna search algorithm (TPE sampler, minimise loss)
optuna_search = OptunaSearch(metric="loss", mode="min")

# 3. ASHA Scheduler — kills bad trials after grace_period epochs
#    Trials not in the top 50% at each rung are terminated early
asha_scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=30,           # maximum epochs per trial
    grace_period=5,     # minimum epochs before pruning kicks in
    reduction_factor=2, # at each rung, keep top 1/reduction_factor trials
)

# 4. Initialise Ray (if not already running)
# Shutdown stale Ray session if running, then reinit
if ray.is_initialized():
    ray.shutdown()

import psutil

# ── Suppress Ray noise ───────────────────────────────────────────────────────
os.environ["RAY_DISABLE_METRICS_EXPORTER"]       = "1"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # silence FutureWarning

# ── Memory & resource budget ─────────────────────────────────────────────────
total_ram_mb = psutil.virtual_memory().total // 1024 // 1024
avail_ram_mb = psutil.virtual_memory().available // 1024 // 1024

# On HPC nodes Ray can see ALL CPUs on the node (e.g. 64-128).
# Cap at SLURM allocation (--cpus-per-task) to avoid over-spawning workers.
slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
slurm_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE",
                  "1" if torch.cuda.is_available() else "0"))

# Run at most 4 trials in parallel regardless of CPU count.
# Each trial loads ~2-3 GB (dataset + model), so cap to avoid OOM.
MAX_PARALLEL    = min(slurm_cpus, 4)

# Give Ray only 15% of total RAM for its object store.
# The remaining 85% is left for PyTorch training workers.
object_store_mb = int(total_ram_mb * 0.15)

ray.init(
    ignore_reinit_error=True,
    num_cpus=MAX_PARALLEL,           # Ray spawns at most this many workers
    num_gpus=slurm_gpus,             # expose SLURM-allocated GPU(s) to Ray
    object_store_memory=object_store_mb * 1024 * 1024,
    logging_level="error",
    log_to_driver=False,
)
print(f"Ray initialised")
print(f"  CPUs visible to Ray : {MAX_PARALLEL}  (SLURM alloc: {slurm_cpus})")
print(f"  GPUs visible to Ray : {slurm_gpus}")
print(f"  Object store cap    : {object_store_mb} MB  (15% of {total_ram_mb} MB total)")
print(f"  Available RAM now   : {avail_ram_mb} MB")

# 5. Configure and launch the Tuner
tuner = tune.Tuner(
    tune.with_resources(
        train_tune,
        resources={
            "cpu": 1,
            # Each of the 4 parallel trials gets 0.25 of the GPU.
            # Set to 1.0 to run trials sequentially on the full GPU.
            "gpu": 0.25 if torch.cuda.is_available() else 0,
        }
    ),
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        scheduler=asha_scheduler,
        num_samples=20,
    ),
    param_space=search_space,
)

print("🚀 Starting Ray Tune sweep (20 trials × ≤30 epochs each)...")
t0 = time.time()
results = tuner.fit()  # set verbose=0 inside TuneConfig to silence 30s status if desired
elapsed = time.time() - t0
print(f"\n✅ Sweep complete in {elapsed/60:.1f} minutes")

"""### 2.4 — Analyse Results & Extract Best Config"""

# ── Analyse sweep results ────────────────────────────────────────────────────

results_df = results.get_dataframe()

# Show all trials sorted by final loss
print("=== All Trial Results (sorted by loss) ===")
display_cols = ["loss", "epoch", "config/lr", "config/batch_size",
                "config/num_heads", "config/d_ff", "config/dropout", "config/num_layers"]
available = [c for c in display_cols if c in results_df.columns]
print(results_df[available].sort_values("loss").to_string(index=False))

# Best result
best_result = results.get_best_result(metric="loss", mode="min")
best_config  = best_result.config
best_loss    = best_result.metrics["loss"]

print(f"\n🏆 Best Trial:")
print(f"   Final loss : {best_loss:.4f}")
for k, v in best_config.items():
    if k not in ("data_path",):
        print(f"   {k:12s}: {v}")

# ── Plot loss curves for all trials ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
for result in results:
    if result.metrics_dataframe is not None:
        df_m = result.metrics_dataframe
        if "loss" in df_m.columns and "epoch" in df_m.columns:
            ax.plot(df_m["epoch"], df_m["loss"], alpha=0.4, linewidth=1)

ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Loss Curves for All Ray Tune Trials\n(ASHA pruned underperforming trials early)")
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_all_trial_loss_curves.png"), dpi=150, bbox_inches="tight")
print("Saved: 02_all_trial_loss_curves.png")
plt.show()

# Highlight best trial
best_df = best_result.metrics_dataframe
if best_df is not None and "loss" in best_df.columns:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(best_df["epoch"], best_df["loss"], color="green", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Training Loss")
    ax2.set_title(f"Best Trial Loss Curve\n(lr={best_config['lr']:.2e}, "
                  f"bs={best_config['batch_size']}, heads={best_config['num_heads']}, "
                  f"d_ff={best_config['d_ff']}, dropout={best_config['dropout']:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_best_trial_loss_curve.png"), dpi=150, bbox_inches="tight")
    print("Saved: 03_best_trial_loss_curve.png")
    plt.show()

"""---
## Part 3 — Retrain Best Config & BLEU Evaluation
"""

# ── Retrain with best hyperparameters ────────────────────────────────────────

BEST_LR         = best_config["lr"]
BEST_BS         = best_config["batch_size"]
BEST_HEADS      = best_config["num_heads"]
BEST_DFF        = best_config["d_ff"]
BEST_DROPOUT    = best_config["dropout"]
BEST_LAYERS     = best_config["num_layers"]
BEST_EPOCHS     = 30         # same cap used during search
MAX_LEN         = 50
D_MODEL         = 512

print(f"Retraining with best config:")
print(f"  lr={BEST_LR:.2e}  batch_size={BEST_BS}  num_heads={BEST_HEADS}")
print(f"  d_ff={BEST_DFF}  dropout={BEST_DROPOUT:.3f}  num_layers={BEST_LAYERS}")
print(f"  epochs={BEST_EPOCHS}")

# DataLoader with best batch size
best_dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
best_loader  = DataLoader(best_dataset, batch_size=BEST_BS, shuffle=True, collate_fn=collate_fn)

# Model
best_model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=D_MODEL,
    num_layers=BEST_LAYERS,
    num_heads=BEST_HEADS,
    d_ff=BEST_DFF,
    max_len=MAX_LEN,
    dropout=BEST_DROPOUT
).to(DEVICE)

best_optimizer = optim.Adam(best_model.parameters(), lr=BEST_LR)
best_criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)

# Training loop with loss tracking
train_losses = []
t0 = time.time()

for epoch in range(BEST_EPOCHS):
    best_model.train()
    epoch_loss = 0.0
    for src, tgt_in, tgt_out in tqdm(best_loader, desc=f"Epoch [{epoch+1}/{BEST_EPOCHS}]", leave=False):
        src, tgt_in, tgt_out = src.to(DEVICE), tgt_in.to(DEVICE), tgt_out.to(DEVICE)
        out = best_model(src, tgt_in, SRC_PAD_IDX, TGT_PAD_IDX)
        out = out.reshape(-1, out.shape[-1])
        loss = best_criterion(out, tgt_out.reshape(-1))
        best_optimizer.zero_grad(); loss.backward(); best_optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(best_loader)
    train_losses.append(avg)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{BEST_EPOCHS} | avg loss: {avg:.4f}")

elapsed = time.time() - t0
print(f"\n⏱  Retrain time: {elapsed/60:.1f} minutes")
print(f"Final loss (epoch {BEST_EPOCHS}): {train_losses[-1]:.4f}")

# Loss curve for best-config retrain
plt.figure(figsize=(8, 4))
plt.plot(range(1, BEST_EPOCHS+1), train_losses, color='darkorange', linewidth=2, marker='o', markersize=3)
plt.axhline(y=train_losses[-1], color='gray', linestyle='--', alpha=0.5, label=f'Final: {train_losses[-1]:.4f}')
plt.xlabel("Epoch"); plt.ylabel("Avg Training Loss")
plt.title(f"Best-Config Retrain ({BEST_EPOCHS} epochs)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_best_config_retrain_loss.png"), dpi=150, bbox_inches="tight")
print("Saved: 04_best_config_retrain_loss.png")
plt.show()

# ── BLEU Evaluation ─────────────────────────────────────────────────────────

print("\n=== Translation Samples (Best Model) ===")
example_sentences = [
    "I love you.",
    "What is your name?",
    "How are you?",
    "The weather is nice today.",
    "She is a good teacher.",
]
for sent in example_sentences:
    pred = translate_sentence(best_model, sent, en_vocab, hi_vocab)
    print(f"\n🗣  EN: {sent}")
    print(f"🇮🇳  HI: {pred}")

print("\n=== BLEU Score Comparison ===")
BASELINE_BLEU = baseline_score
print(f"📌 Baseline BLEU (100 epochs): {BASELINE_BLEU:.2f}")
best_bleu = evaluate_bleu(best_model) * 100
print(f"🚀 Best-Config BLEU ({BEST_EPOCHS} epochs): {best_bleu:.2f}")
delta = best_bleu - BASELINE_BLEU
print(f"\n{'✅ IMPROVEMENT' if delta >= 0 else '⚠️  Gap'}: {delta:+.2f} BLEU points")
if best_bleu >= BASELINE_BLEU:
    print(f"🎯 Goal achieved: BLEU ≥ {BASELINE_BLEU} in just {BEST_EPOCHS} epochs!")
else:
    print(f"Close! Further training or wider search space may close the gap.")

# ── BLEU Comparison Bar Chart ────────────────────────────────────────────────
fig_bleu, ax_bleu = plt.subplots(figsize=(6, 4))
labels = [f"Baseline\n(100 epochs)", f"Ray Tune + Optuna\n({BEST_EPOCHS} epochs)"]
values = [BASELINE_BLEU, best_bleu]
colors = ["#4878CF", "#6ACC65" if best_bleu >= BASELINE_BLEU else "#D65F5F"]
bars = ax_bleu.bar(labels, values, color=colors, width=0.4, edgecolor="white", linewidth=1.2)
ax_bleu.axhline(y=BASELINE_BLEU, color="#4878CF", linestyle="--", alpha=0.5, linewidth=1)
for bar, val in zip(bars, values):
    ax_bleu.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax_bleu.set_ylabel("BLEU Score"); ax_bleu.set_ylim(0, max(values) * 1.15)
ax_bleu.set_title("BLEU Score: Baseline vs Ray Tune + Optuna")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_bleu_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved: 05_bleu_comparison.png")
plt.show()

# ── Hyperparameter importance plot (top configs by loss) ─────────────────────

try:
    results_df_plot = results.get_dataframe()
    hp_cols = [c for c in results_df_plot.columns if c.startswith("config/") and c != "config/data_path"]
    if hp_cols and "loss" in results_df_plot.columns:
        plot_df = results_df_plot[hp_cols + ["loss"]].dropna()
        plot_df.columns = [c.replace("config/", "") for c in plot_df.columns]
        fig_hp, axes_hp = plt.subplots(1, len(hp_cols), figsize=(3 * len(hp_cols), 4), sharey=False)
        if len(hp_cols) == 1:
            axes_hp = [axes_hp]
        for ax_hp, col in zip(axes_hp, [c.replace("config/", "") for c in hp_cols]):
            ax_hp.scatter(plot_df[col], plot_df["loss"], alpha=0.6, s=40)
            ax_hp.set_xlabel(col, fontsize=9)
            ax_hp.set_ylabel("Loss" if col == plot_df.columns[0] else "")
            ax_hp.set_title(col, fontsize=9)
        fig_hp.suptitle("Hyperparameter vs Final Loss (all trials)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "06_hyperparameter_scatter.png"), dpi=150, bbox_inches="tight")
        print("Saved: 06_hyperparameter_scatter.png")
        plt.show()
except Exception as e:
    print(f"(Skipped hyperparameter scatter: {e})")

# ── Save best model & vocabs ─────────────────────────────────────────────────

torch.save(best_model.state_dict(), "m25csa002_ass_4_best_model.pth")

with open("en_vocab.pkl", "wb") as f:
    pickle.dump(en_vocab, f)
with open("hi_vocab.pkl", "wb") as f:
    pickle.dump(hi_vocab, f)

# Save best hyperparameters for reproducibility
import json as _json
best_config_save = {k: v for k, v in best_config.items() if k != "data_path"}
with open("best_hyperparams.json", "w") as f:
    _json.dump(best_config_save, f, indent=2)

print("✅ Saved:")
print("   • m25csa002_ass_4_best_model.pth  — model weights")
print("   • en_vocab.pkl / hi_vocab.pkl   — vocabularies")
print("   • best_hyperparams.json         — best hyperparameters")
print(f"   • {PLOTS_DIR}/01_sentence_length_distribution.png")
print(f"   • {PLOTS_DIR}/02_all_trial_loss_curves.png")
print(f"   • {PLOTS_DIR}/03_best_trial_loss_curve.png")
print(f"   • {PLOTS_DIR}/04_best_config_retrain_loss.png")
print(f"   • {PLOTS_DIR}/05_bleu_comparison.png")
print(f"   • {PLOTS_DIR}/06_hyperparameter_scatter.png")
print(f"\nBest config summary:")
for k, v in best_config_save.items():
    print(f"   {k:12s}: {v}")
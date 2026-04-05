# ================================
# 1) Mount Google Drive
# ================================
from google.colab import drive
print("Mounting Google Drive...")
drive.mount('/content/drive')
!ls "/content/drive/MyDrive/AFND"

# ================================
# 2) Install / Import
# ================================
!pip install -q transformers scikit-learn torch arabert

import os, json, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast          # Mixed-precision
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ================================
# 3) Load Dataset
# ================================
DATA_ROOT    = "/content/drive/MyDrive/AFND/Dataset"
SOURCES_PATH = "/content/drive/MyDrive/AFND/sources.json"

with open(SOURCES_PATH, "r", encoding="utf-8") as f:
    source_labels = json.load(f)

all_articles = []
for folder_name in sorted(os.listdir(DATA_ROOT), key=lambda x: int(x.split('_')[-1])):
    folder_path = os.path.join(DATA_ROOT, folder_name)
    json_path   = os.path.join(folder_path, "scraped_articles.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data  = json.load(f)
            label = source_labels.get(folder_name, None)
            if   isinstance(data, list): articles = data
            elif isinstance(data, dict) and "articles" in data: articles = data["articles"]
            else: articles = [data]
            for article in articles:
                if not isinstance(article, dict): continue
                title = article.get("title", "")
                text  = article.get("text",  "")
                date  = article.get("date",  article.get("publication_date", ""))
                all_articles.append({"source": folder_name, "title": title,
                                     "text": text, "date": date, "label": label})
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

df = pd.DataFrame(all_articles)
print(f"Total raw articles: {len(df)}")

# ================================
# 4) Clean Labels
# ================================
df["label"] = df["label"].astype(str).str.strip().str.lower()
label_alias = {
    "credable": "credible",         "not credable": "not credible",
    "not_credible": "not credible", "nondecided":   "undecided",
    "non decided": "undecided",     "non_decided":  "undecided",
}
df["label"] = df["label"].replace(label_alias)
df = df[df["label"].isin(["credible", "not credible"])].reset_index(drop=True)
print(f"After label filter: {len(df)}  |  {df['label'].value_counts().to_dict()}")

# ================================
# 5) AraBERT-Specific Preprocessing
# ================================
# Consistent with aubmindlab/bert-base-arabertv02 pretraining norms
def arabert_preprocess(text):
    """Normalize Arabic text to match AraBERT pretraining conventions."""
    if not isinstance(text, str): return ""
    # Remove diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Normalize Alef variants -> bare Alef
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize Yeh variants -> dotless Yeh
    text = re.sub(r"[يى]", "ي", text)
    # Normalize Teh Marbuta -> Heh
    text = re.sub(r"ة", "ه", text)
    # Normalize Waw with hamza
    text = re.sub(r"ؤ", "و", text)
    # Remove tatweel (kashida)
    text = re.sub(r"\u0640", "", text)
    # Remove non-Arabic characters (Latin, digits)
    text = re.sub(r"[A-Za-z0-9]", " ", text)
    # Remove Arabic-Indic digits
    text = re.sub(r"[٠-٩]", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["title"] = df["title"].fillna("").astype(str)
df["text"]  = df["text"].fillna("").astype(str)
df["combined_text"] = (
    df["title"].str.strip() + " " + df["text"].str.strip()
).str.strip().apply(arabert_preprocess)

df["combined_length"] = df["combined_text"].str.len()
df = df[df["combined_length"] >= 80].reset_index(drop=True)
print(f"After length filter: {len(df)}")

# ================================
# 6) Encode Labels
# ================================
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])
print("Classes:", le.classes_)  # 0=credible, 1=not credible

# ================================
# 7) Balanced Sampling (10%)
# ================================
FRACTION    = 0.10
target_each = int(len(df) * FRACTION) // 2
df_c = df[df["label"] == "credible"]
df_n = df[df["label"] == "not credible"]
target_each = min(target_each, len(df_c), len(df_n))

df_balanced = pd.concat([
    df_c.sample(target_each, random_state=SEED),
    df_n.sample(target_each, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Balanced size: {len(df_balanced)}  |  {df_balanced['label'].value_counts().to_dict()}")

# ================================
# 8) Class Weights / Focal Loss
# ================================
class_counts = df_balanced["label_encoded"].value_counts().sort_index().values
class_weights = torch.tensor(
    [len(df_balanced) / (2 * c) for c in class_counts],
    dtype=torch.float
).to(device)
print(f"Class weights: {class_weights.tolist()}")

class FocalLoss(nn.Module):
    """Focal Loss — downweights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha   = alpha    # class weight tensor
        self.gamma   = gamma
        self.ls      = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets,
            weight=self.alpha,
            label_smoothing=self.ls,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

# ================================
# 9) Train / Val / Test Split
# ================================
train_df, temp_df = train_test_split(
    df_balanced, test_size=0.2, random_state=SEED,
    stratify=df_balanced["label_encoded"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=SEED,
    stratify=temp_df["label_encoded"]
)
print(f"Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")

# ================================
# 10) Tokenizer + Dataset
#     MAX_LEN = 384 for more context
# ================================
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
MAX_LEN    = 384    # ↑ from 256 — more article context
BATCH_SIZE = 8      # keep small for T4 VRAM
ACCUM_STEPS = 4     # effective batch = 8 × 4 = 32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["combined_text"].tolist()
        self.labels = df["label_encoded"].tolist()

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc  = tokenizer(
            self.texts[idx], truncation=True,
            padding="max_length", max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_loader = DataLoader(NewsDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(NewsDataset(val_df),   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(NewsDataset(test_df),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ================================
# 11) Model  (AraBERT + Multi-scale CNN + BiLSTM + Attention Pool)
#     CNN filters reduced 128→64 for T4 stability
#     Dropout increased to 0.4 / 0.3 to reduce overfitting
# ================================
class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):                        # (B, T, H)
        w = torch.softmax(self.attn(x), dim=1)   # (B, T, 1)
        return (x * w).sum(dim=1)                # (B, H)


class AraBertCNNLSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        # Gradient checkpointing: saves ~40% VRAM on T4
        # use_cache must be disabled when checkpointing
        self.bert.config.use_cache = False
        self.bert.gradient_checkpointing_enable()

        hidden = self.bert.config.hidden_size    # 768

        # Multi-scale CNN — 64 filters (down from 128 for T4 stability)
        self.conv3 = nn.Conv1d(hidden, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden, 64, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()

        # 2-layer BiLSTM with increased dropout (0.3)
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3
        )

        self.pool = AttentionPool(256)

        # Deeper head with increased dropout (0.4)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x   = out.last_hidden_state               # (B, T, 768)

        xt  = x.transpose(1, 2)                   # (B, 768, T)
        c3  = self.relu(self.conv3(xt))            # (B, 64, T)
        c5  = self.relu(self.conv5(xt))            # (B, 64, T)
        x   = torch.cat([c3, c5], dim=1)           # (B, 128, T)
        x   = x.transpose(1, 2)                    # (B, T, 128)

        lstm_out, _ = self.lstm(x)                # (B, T, 256)
        x = self.pool(lstm_out)                   # (B, 256)

        return self.classifier(x)                 # (B, 2)


model = AraBertCNNLSTMClassifier().to(device)

# ================================
# 12) Optimizer + Scheduler + Mixed-Precision Scaler
#     Higher LR for CNN/LSTM (2e-4), lower for BERT (2e-5)
# ================================
EPOCHS = 5

optimizer = torch.optim.AdamW([
    {"params": model.bert.parameters(),       "lr": 2e-5,  "weight_decay": 0.01},
    {"params": model.conv3.parameters(),      "lr": 2e-4},
    {"params": model.conv5.parameters(),      "lr": 2e-4},
    {"params": model.lstm.parameters(),       "lr": 2e-4},
    {"params": model.pool.parameters(),       "lr": 2e-4},
    {"params": model.classifier.parameters(), "lr": 2e-4},
])

total_steps = (len(train_loader) // ACCUM_STEPS) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# AMP scaler for mixed-precision training
scaler = GradScaler()

# ================================
# 13) Training / Eval Functions
#     FIX: .detach().cpu().numpy() on all tensors before numpy()
# ================================
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0
    accum_count = 0

    if train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)

        if train:
            # ---- Mixed-precision forward ----
            with autocast():
                logits = model(ids, mask)
                loss   = criterion(logits, lbls) / ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count == ACCUM_STEPS:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            # Recover true loss for logging
            total_loss += loss.item() * ACCUM_STEPS

        else:
            with torch.no_grad():
                with autocast():
                    logits = model(ids, mask)
                    loss   = criterion(logits, lbls)
            total_loss += loss.item()

        # ---- FIX: detach before numpy() ----
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]   # prob of class 1
        preds = torch.argmax(logits.detach(), dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.detach().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / len(loader), acc, f1, auc


# ================================
# 14) Training Loop + Early Stopping + 🔥 AUTO SAVE
# ================================
best_val_f1 = 0
patience    = 2
no_improve  = 0

SAVE_PATH = "/content/drive/MyDrive/AFND/best_model.pt"

for epoch in range(EPOCHS):
    t_loss, t_acc, t_f1, t_auc = run_epoch(train_loader, train=True)
    v_loss, v_acc, v_f1, v_auc = run_epoch(val_loader,   train=False)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train  ->  Loss:{t_loss:.4f}  Acc:{t_acc:.4f}  F1:{t_f1:.4f}  AUC:{t_auc:.4f}")
    print(f"  Val    ->  Loss:{v_loss:.4f}  Acc:{v_acc:.4f}  F1:{v_f1:.4f}  AUC:{v_auc:.4f}")

    # 🔥 BEST MODEL CHECK + SAVE مباشرة
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        no_improve  = 0

        # 🔥 حفظ مباشر على Drive (أفضل حل)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_f1": best_val_f1,
            "epoch": epoch + 1
        }, SAVE_PATH)

        print(f"  ✅ Best model UPDATED & saved to {SAVE_PATH}")

    else:
        no_improve += 1
        print(f"  ⚠️  No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print("  🛑 Early stopping triggered.")
            break

print(f"\n🏆 Best Val F1: {best_val_f1:.4f}")

# ================================
# 15) ROC-AUC + Optimal Threshold (Youden J)
# ================================
model.eval()
all_probs_val, all_labels_val = [], []

with torch.no_grad():
    for batch in val_loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        with autocast():
            logits = model(ids, mask)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        all_probs_val.extend(probs.cpu().numpy())
        all_labels_val.extend(lbls.detach().cpu().numpy())

fpr_c, tpr_c, thresholds = roc_curve(all_labels_val, all_probs_val)
youden_j   = tpr_c - fpr_c
opt_idx    = np.argmax(youden_j)
opt_thresh = thresholds[opt_idx]
val_auc    = roc_auc_score(all_labels_val, all_probs_val)

print(f"\nVal ROC-AUC      : {val_auc:.4f}")
print(f"Optimal threshold: {opt_thresh:.4f}  (Youden J = {youden_j[opt_idx]:.4f})")

plt.figure(figsize=(7, 5))
plt.plot(fpr_c, tpr_c, lw=2, label=f"AUC = {val_auc:.4f}")
plt.scatter(fpr_c[opt_idx], tpr_c[opt_idx], color="red", zorder=5,
            label=f"Optimal thresh = {opt_thresh:.3f}")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (Validation)"); plt.legend(); plt.show()

# ================================
# 16) Test Evaluation
# ================================
all_preds_test, all_labels_test, all_probs_test = [], [], []

with torch.no_grad():
    for batch in test_loader:
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)
        with autocast():
            logits = model(ids, mask)
        probs  = torch.softmax(logits.detach(), dim=1)[:, 1]
        preds  = (probs >= opt_thresh).long()
        all_preds_test.extend(preds.cpu().numpy())
        all_labels_test.extend(lbls.detach().cpu().numpy())
        all_probs_test.extend(probs.cpu().numpy())

test_auc = roc_auc_score(all_labels_test, all_probs_test)
test_acc = accuracy_score(all_labels_test, all_preds_test)
test_f1  = f1_score(all_labels_test, all_preds_test, average="macro")

print("\n========== TEST RESULTS ==========")
print(f"Accuracy : {test_acc:.4f}")
print(f"Macro F1 : {test_f1:.4f}")
print(f"ROC-AUC  : {test_auc:.4f}")
print()
print(classification_report(all_labels_test, all_preds_test, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(all_labels_test, all_preds_test))

# ================================
# 17) Inference Speed Benchmark
# ================================
model.eval()
dummy_ids  = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN)).to(device)
dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long).to(device)

# Warm-up
for _ in range(5):
    with torch.no_grad():
        with autocast(): _ = model(dummy_ids, dummy_mask)

N_RUNS  = 100
t_start = time.perf_counter()
for _ in range(N_RUNS):
    with torch.no_grad():
        with autocast(): _ = model(dummy_ids, dummy_mask)
t_end = time.perf_counter()

ms_per_sample = (t_end - t_start) / N_RUNS * 1000
print(f"\n⚡ Single-sample  : {ms_per_sample:.2f} ms  ({1000/ms_per_sample:.1f} samples/sec)")

# Batch throughput
bs_bench   = 32
bench_data = next(iter(DataLoader(NewsDataset(test_df.head(bs_bench)), batch_size=bs_bench)))
ids_b  = bench_data["input_ids"].to(device)
mask_b = bench_data["attention_mask"].to(device)

t0 = time.perf_counter()
for _ in range(20):
    with torch.no_grad():
        with autocast(): _ = model(ids_b, mask_b)
t1 = time.perf_counter()

batch_ms = (t1 - t0) / 20 * 1000
print(f"⚡ Batch-{bs_bench} throughput : {batch_ms:.2f} ms  ({bs_bench / batch_ms * 1000:.0f} samples/sec)")

# ================================
# 18) Cross-Dataset Generalization (5-Fold CV)
# ================================
print("\n========== 5-FOLD CROSS-VALIDATION ==========")
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
y_all = df_balanced["label_encoded"].values

fold_accs, fold_f1s, fold_aucs = [], [], []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(df_balanced, y_all)):
    tr_fold = df_balanced.iloc[tr_idx].reset_index(drop=True)
    vl_fold = df_balanced.iloc[vl_idx].reset_index(drop=True)

    tr_ldr = DataLoader(NewsDataset(tr_fold), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    vl_ldr = DataLoader(NewsDataset(vl_fold), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    fold_model = AraBertCNNLSTMClassifier().to(device)
    fold_opt   = torch.optim.AdamW(fold_model.parameters(), lr=2e-5, weight_decay=0.01)
    fold_steps = (len(tr_ldr) // ACCUM_STEPS) * 3
    fold_sched = get_linear_schedule_with_warmup(
        fold_opt, num_warmup_steps=int(0.1 * fold_steps),
        num_training_steps=fold_steps
    )
    fold_scaler = GradScaler()
    fold_accum  = 0

    for ep in range(3):
        fold_model.train()
        fold_opt.zero_grad()
        for batch in tr_ldr:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            with autocast():
                logits = fold_model(ids, mask)
                loss   = criterion(logits, lbls) / ACCUM_STEPS
            fold_scaler.scale(loss).backward()
            fold_accum += 1
            if fold_accum == ACCUM_STEPS:
                fold_scaler.unscale_(fold_opt)
                nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                fold_scaler.step(fold_opt); fold_scaler.update()
                fold_sched.step(); fold_opt.zero_grad(); fold_accum = 0

    fold_model.eval()
    fp, fl, fpr_list = [], [], []
    with torch.no_grad():
        for batch in vl_ldr:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            with autocast():
                logits = fold_model(ids, mask)
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=1)
            fp.extend(preds.cpu().numpy())
            fl.extend(lbls.detach().cpu().numpy())
            fpr_list.extend(probs.cpu().numpy())

    f_acc = accuracy_score(fl, fp)
    f_f1  = f1_score(fl, fp, average="macro")
    f_auc = roc_auc_score(fl, fpr_list)
    fold_accs.append(f_acc); fold_f1s.append(f_f1); fold_aucs.append(f_auc)
    print(f"  Fold {fold+1}  ->  Acc:{f_acc:.4f}  F1:{f_f1:.4f}  AUC:{f_auc:.4f}")

    # Free fold model memory
    del fold_model, fold_opt, fold_sched, fold_scaler
    torch.cuda.empty_cache()

print(f"\n  Mean Acc : {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
print(f"  Mean F1  : {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
print(f"  Mean AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
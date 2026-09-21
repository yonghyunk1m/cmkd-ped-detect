"""Proper LOSO teacher evaluation — each fold uses its own checkpoint."""
import os, re, glob, torch, numpy as np, pandas as pd
import torch.nn as nn

class TeacherMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress_mlp = nn.Sequential(nn.Linear(258, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.classifier_exist = nn.Linear(128, 2)
    def forward(self, x):
        return self.classifier_exist(self.compress_mlp(x))

EMBED_DIR = '/path/to/teacher_embeddings'
LABEL_ROOT = '/path/to/ASPED_v.a'
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'work_dir/teacher_video_only')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

pattern = re.compile(r'^(Session_\d+)_(\w+)_(\d+_6m_ellipse)_(\d+)_embed_258\.pt$')
sessions = sorted([d for d in os.listdir(CKPT_DIR) if d.startswith('Session_')])

results = []
for ts in sessions:
    ckpt = glob.glob(os.path.join(CKPT_DIR, ts, 'best-*.ckpt'))[0]
    model = TeacherMLP().to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE).get('state_dict', {})
    model.load_state_dict({k: v for k, v in sd.items()
                           if 'compress_mlp' in k or 'classifier_exist' in k}, strict=True)
    model.eval()

    probs_all, labels_all = [], []
    for fname in sorted(os.listdir(EMBED_DIR)):
        m = pattern.match(fname)
        if not m or m.group(1) != ts:
            continue
        session, location, clip_stem, rec_idx = m.groups()
        csv_path = os.path.join(LABEL_ROOT, session, location, 'Labels',
                                clip_stem.split('_')[0] + '.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        col = 'recorder' + rec_idx + '_6m'
        if col not in df.columns:
            continue
        labels = (df[col].values > 0).astype(int)
        data = torch.load(os.path.join(EMBED_DIR, fname), map_location=DEVICE)
        emb = data['embeddings'] if isinstance(data, dict) else data
        T = min(len(labels), emb.shape[0])
        with torch.no_grad():
            logits = model(emb[:T])
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs_all.extend(p[:T].tolist())
        labels_all.extend(labels[:T].tolist())

    P, L = np.array(probs_all), np.array(labels_all)

    # Default t=0.5
    pred = (P >= 0.5).astype(int)
    tn = ((pred == 0) & (L == 0)).sum()
    tp = ((pred == 1) & (L == 1)).sum()
    fp = ((pred == 1) & (L == 0)).sum()
    fn = ((pred == 0) & (L == 1)).sum()
    a0 = tn / (tn + fp); a1 = tp / (tp + fn)
    ma = (a0 + a1) / 2
    pr = tp / (tp + fp) if tp + fp > 0 else 0
    f1 = 2 * pr * a1 / (pr + a1) if pr + a1 > 0 else 0

    # Optimal threshold (sweep)
    best_ma, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        p2 = (P >= t).astype(int)
        t0 = ((p2 == 0) & (L == 0)).sum() / (((p2 == 0) & (L == 0)).sum() + ((p2 == 1) & (L == 0)).sum())
        t1 = ((p2 == 1) & (L == 1)).sum() / (((p2 == 1) & (L == 1)).sum() + ((p2 == 0) & (L == 1)).sum())
        m2 = (t0 + t1) / 2
        if m2 > best_ma:
            best_ma, best_t = m2, t

    print(f'{ts}: macro_acc={ma:.4f} (t=0.5) | best={best_ma:.4f} (t={best_t:.2f}) | F1={f1:.4f} acc0={a0:.4f} recall={a1:.4f}')
    results.append({'session': ts, 'macro_acc': ma, 'best_macro_acc': best_ma, 'best_t': best_t, 'f1': f1})

print(f'\nMean macro_acc (t=0.5):    {np.mean([r["macro_acc"] for r in results]):.4f}')
print(f'Mean macro_acc (t=optimal): {np.mean([r["best_macro_acc"] for r in results]):.4f}')
print(f'Mean F1 (t=0.5):           {np.mean([r["f1"] for r in results]):.4f}')

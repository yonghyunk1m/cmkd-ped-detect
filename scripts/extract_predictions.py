"""Extract per-sample prediction probabilities for visualization.

Runs the (cached) best ckpt for each (method, session) and saves the held-out
test probabilities together with the GT labels into a single .npz file per
method. Output schema:

    work_dir/predictions/<method>.npz
        session_5242023_probs : (N,) float32, P(Ped | x)
        session_5242023_labels: (N,) int8, {0,1}
        session_6012023_probs : ...
        ...

The visualization scripts (`scripts/plot_distributions.py`) consume these.

Usage:
    python scripts/extract_predictions.py \
        --methods baseline_binary_v1 kd_logit_unw kd_logit_unw_immune \
                  kd_hybrid_logit_cosine_immune
"""
import os, sys, glob, argparse, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import get_train_test_splits
from data.dataset import ASPEDDataset
from models.seq2seq import ASPEDLightningModel
from models.kd_model import ASPEDKDModel

FEATURE_ROOT = "/media/ykim/Linux/ASPED_v1_npy"
WORK_DIR = "work_dir"
SESSIONS = [
    "Session_5242023", "Session_6012023", "Session_6072023",
    "Session_6212023", "Session_6282023",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_best_ckpt(exp_name, session):
    base = os.path.join(WORK_DIR, exp_name, session)
    if not os.path.isdir(base):
        return None
    ckpts = (glob.glob(os.path.join(base, "best-*.ckpt"))
             + glob.glob(os.path.join(base, "best-*", "*.ckpt")))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def build_model(config, is_kd):
    mc = config["model"]
    kwargs = dict(
        token_dim=mc.get("token_dim", 128),
        nhead=mc.get("nhead", 4),
        dropout=mc.get("dropout", 0.2),
        num_classes=mc["n_classes"],
        num_layers=mc.get("num_layers", 1),
        task_type=mc.get("task", "classification"),
        backbone_name=mc.get("backbone_name", "vggish"),
        backbone_finetune=mc.get("backbone_finetune", False),
    )
    return ASPEDKDModel(kd_cfg=mc["kd"], **kwargs) if is_kd else ASPEDLightningModel(**kwargs)


@torch.no_grad()
def collect(model, session, n_classes=2):
    model.eval().to(DEVICE)
    _, test_list = get_train_test_splits(
        feature_root_dir=FEATURE_ROOT, test_session=session,
        window_size_sec=10, test_stride_sec=10)
    ds = ASPEDDataset(test_list, task_type="classification", max_class=n_classes - 1)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    probs, labels = [], []
    for batch in loader:
        audio, y = batch[0], batch[-1]
        logits = model(audio.to(DEVICE)).squeeze(-1)
        probs.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        labels.append(y.numpy().reshape(-1))
    return np.concatenate(probs).astype(np.float32), np.concatenate(labels).astype(np.int8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--out_dir", default="work_dir/predictions")
    args = p.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(args.out_dir, exist_ok=True)

    for method in args.methods:
        cfg_path = f"configs/{method}.yaml"
        if not os.path.exists(cfg_path):
            print(f"[skip] {method}: no config")
            continue
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        is_kd = (config.get("model", {}).get("kd") is not None
                 and not method.startswith("baseline"))

        save = {}
        for session in SESSIONS:
            ckpt = find_best_ckpt(method, session)
            if ckpt is None:
                print(f"  [skip] {method}/{session}: no ckpt")
                continue
            print(f"  [eval] {method}/{session} <- {os.path.basename(ckpt)}")
            model = build_model(config, is_kd)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=False)
            probs, labels = collect(model, session, config["model"]["n_classes"])
            # NaN guard (in case any ckpt has lingering instability)
            n_nan = int(np.isnan(probs).sum())
            if n_nan > 0:
                print(f"    [warn] {n_nan} NaN probabilities; clipped to 0.5")
                probs = np.nan_to_num(probs, nan=0.5)
            sid = session.lower()
            save[f"{sid}_probs"] = probs
            save[f"{sid}_labels"] = labels
            del model; torch.cuda.empty_cache()

        if save:
            out = os.path.join(args.out_dir, f"{method}.npz")
            np.savez_compressed(out, **save)
            sizes = ", ".join(f"{k.split('_')[1]}={len(v)}" for k, v in save.items()
                               if k.endswith("_probs"))
            print(f"[saved] {out}  ({sizes})")


if __name__ == "__main__":
    main()

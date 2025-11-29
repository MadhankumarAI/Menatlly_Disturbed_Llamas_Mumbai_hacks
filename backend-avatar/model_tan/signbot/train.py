# server/train.py
import argparse, os, json, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from model import KPTransformer
from dataset import KeypointDataset

def build_label_encoder(manifest_csv):
    df = pd.read_csv(manifest_csv)
    le = LabelEncoder()
    le.fit(df['label'])
    return le

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

def train_loop(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # build encoders
    train_df = pd.read_csv(args.train_csv)
    le = LabelEncoder(); le.fit(train_df['label'])
    label_map = {lab: int(i) for i, lab in enumerate(le.classes_)}

    train_ds = KeypointDataset(args.train_csv, label_map, seq_len=args.seq_len, augment=True)
    val_ds = KeypointDataset(args.val_csv, label_map, seq_len=args.seq_len, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=collate_fn)

    sample = np.load(train_df.iloc[0]['file'])['keypoints']
    input_dim = sample.shape[1]
    num_classes = len(le.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KPTransformer(input_dim=input_dim, num_classes=num_classes,
                          d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
                          ff_dim=args.ff_dim, dropout=args.dropout)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0; total=0; correct=0
        for X,y in train_loader:
            X = X.to(device); y = y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()) * X.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += X.size(0)
        train_acc = correct/total
        train_loss = total_loss/total

        # validation
        model.eval()
        total=0; correct=0
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(device); y = y.to(device)
                logits = model(X)
                preds = logits.argmax(1)
                total += X.size(0)
                correct += (preds==y).sum().item()
        val_acc = correct/total
        print(f"Epoch {epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model_state": model.state_dict(),
                "label_classes": list(le.classes_),
                "input_dim": input_dim,
                "seq_len": args.seq_len,
                "args": vars(args)
            }
            torch.save(ckpt, args.out)
            print("Saved best checkpoint to", args.out)

    # also write label file separate
    with open(args.out + ".labels.json", "w") as f:
        json.dump(list(le.classes_), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=192, dest="d_model")
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512, dest="ff_dim")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--out", default="./models/recognition.pth")
    args = parser.parse_args()
    train_loop(args)

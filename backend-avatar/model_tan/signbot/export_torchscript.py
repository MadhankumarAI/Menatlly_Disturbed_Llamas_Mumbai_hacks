# server/export_torchscript.py
import torch, argparse, numpy as np
from infer import Recognizer

def export(ckpt_path, out_path, sample_seq_len=64, input_dim=None):
    rec = Recognizer(ckpt_path)
    model = rec.model.eval().cpu()
    # build a dummy input (B=1)
    if input_dim is None:
        input_dim = rec.model.input_linear.in_features if hasattr(rec.model, 'input_linear') else model.input_linear.in_features
    dummy = torch.randn(1, sample_seq_len, input_dim)
    traced = torch.jit.trace(model, dummy)
    traced.save(out_path)
    print("Saved TorchScript to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    export(args.ckpt, args.out, sample_seq_len=args.seq_len)

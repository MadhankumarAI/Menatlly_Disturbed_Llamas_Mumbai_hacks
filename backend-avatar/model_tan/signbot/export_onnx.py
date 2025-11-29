# server/export_onnx.py
import torch, argparse
from infer import Recognizer

def export_onnx(ckpt_path, out_path, seq_len=64):
    rec = Recognizer(ckpt_path)
    model = rec.model.eval().cpu()
    dummy = torch.randn(1, seq_len, model.input_linear.in_features)
    torch.onnx.export(model, dummy, out_path,
                      input_names=['input'],
                      output_names=['logits'],
                      dynamic_axes={'input': {0: 'batch', 1: 'time'}, 'logits': {0:'batch'}},
                      opset_version=12)
    print("Saved ONNX to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()
    export_onnx(args.ckpt, args.out, seq_len=args.seq_len)

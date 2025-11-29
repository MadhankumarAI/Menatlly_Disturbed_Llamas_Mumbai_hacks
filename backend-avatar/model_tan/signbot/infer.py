# server/infer.py
import torch
import numpy as np
from model import KPTransformer

class Recognizer:
    def __init__(self, ckpt_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu")
        self.label_classes = ck['label_classes']
        input_dim = ck.get('input_dim')
        seq_len = ck.get('seq_len', 64)
        args = ck.get('args', {})
        # create model with saved hyperparams if present
        self.model = KPTransformer(input_dim=input_dim, num_classes=len(self.label_classes),
                                   d_model=args.get('d_model', 192),
                                   nhead=args.get('nhead', 6),
                                   num_layers=args.get('num_layers', 4),
                                   ff_dim=args.get('ff_dim', 512),
                                   dropout=args.get('dropout', 0.1))
        self.model.load_state_dict(ck['model_state'])
        self.model.to(self.device)
        self.model.eval()
        self.seq_len = seq_len

    def _resample(self, arr):
        # arr: (T, D)
        T = arr.shape[0]
        if T == self.seq_len:
            return arr
        if T < 2:
            return np.repeat(arr, self.seq_len, axis=0)[:self.seq_len]
        idx = np.linspace(0, T-1, self.seq_len).astype(int)
        return arr[idx]

    def predict(self, keypoints_sequence):
        """
        keypoints_sequence: np.array shape (T, D) or list
        returns: (label, confidence, probs)
        """
        arr = np.asarray(keypoints_sequence, dtype=np.float32)
        arr = self._resample(arr)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, T, D)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            return self.label_classes[idx], float(probs[idx]), probs

# quick test
if __name__ == "__main__":
    import sys, numpy as np
    r = Recognizer(sys.argv[1])
    sample = np.load("./data/keypoints_example.npz")['keypoints']
    label, conf, probs = r.predict(sample)
    print("Predicted:", label, conf)

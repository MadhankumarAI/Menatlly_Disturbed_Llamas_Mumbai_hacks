Model package: KPTransformer keypoint recognizer
-----------------------------------------------

Input format:
- keypoints: sequence of frames; each frame is flattened [x,y,z] for:
  - 33 pose landmarks (MediaPipe)
  - 21 left hand
  - 21 right hand
- So per-frame feature dimension D = (33+21+21)*3 = 225 (if using these counts)
- The model resamples sequences to seq_len (default 64). Keep consistent preprocessing.

Quick steps:
1. Prepare manifests CSVs: columns 'file','label' where file points to .npz containing array 'keypoints'
2. Train:
   python server/train.py --train-csv ./data/manifest_train.csv --val-csv ./data/manifest_val.csv --epochs 30 --out ./models/recognition.pth
3. Infer (python wrapper):
   from infer import Recognizer
   r = Recognizer('./models/recognition.pth')
   label, conf, probs = r.predict(numpy_array_of_shape_TxD)
4. Export to TorchScript or ONNX (see export scripts) for mobile/web embedding.
5. Serve via FastAPI (server/serve_fastapi.py) for app integration.

Latency & on-device considerations:
- A small model (d_model=128, 2-4 layers) runs fast on CPU for single inferences (target <100ms for seq_len=64 on modern server CPU). Measure and tune.
- For mobile/on-device, prefer TorchScript + quantization or ONNX + ONNX Runtime Mobile.
- Reduce seq_len and model size for lower latency.

Tips:
- Keep preprocessing identical between train/inference (normalization, joint ordering).
- Save 'label_classes' with checkpoint (done in train.py) to map indices to gloss tokens.
- Use confidence thresholding at app side to avoid false positives.


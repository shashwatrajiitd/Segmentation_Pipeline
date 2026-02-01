Place your local BiRefNet model file here as:

  models/birefnet.torchscript

This pipeline expects the file to be a TorchScript module saved with:

  torch.jit.save(scripted_or_traced_model, "birefnet.torchscript")

Notes:
- Pure state_dict checkpoints (torch.save(model.state_dict(), ...)) are NOT directly loadable here
  because they require the original model architecture source code.
- File extension does not matter; it can be .pth as long as it is TorchScript.

Quick setup (recommended):

  python get_model.py --out models/birefnet.torchscript --input-res 1088


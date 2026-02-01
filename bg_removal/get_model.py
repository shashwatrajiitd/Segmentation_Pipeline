from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_birefnet_torchscript(
    *,
    out_path: Path,
    hf_repo: str = "ZhengPeng7/BiRefNet",
    input_res: int = 1088,
) -> None:
    """
    Export BiRefNet to TorchScript (trace).

    Ground rules (do not violate):
    - Trace on CPU only (avoid MPS constant pinning / device capture issues).
    - float32 only.
    - batch size 1.
    """
    try:
        from transformers import AutoModelForImageSegmentation
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("transformers is not installed. Run: pip install transformers") from e

    torch.set_default_dtype(torch.float32)

    # IMPORTANT:
    # Some recent `transformers` versions may initialize modules on the "meta" device
    # (via low_cpu_mem_usage / init_empty_weights). BiRefNet's __init__ does real tensor ops
    # and calls `.item()`, which fails on meta tensors.
    #
    # Fix: disable low_cpu_mem_usage so initialization happens on real CPU tensors.
    # BiRefNet's backbone init computes a Python list via `torch.linspace(...).item()`.
    # When `transformers` uses meta-device initialization internally, `torch.linspace` may
    # produce a meta tensor, and `.item()` crashes ("cannot be called on meta tensors").
    # We defensively force `torch.linspace` to materialize on CPU during model construction.
    _orig_linspace = torch.linspace

    def _cpu_linspace(start, end, steps, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("device", None)
        return _orig_linspace(start, end, steps, *args, device="cpu", **kwargs)

    torch.linspace = _cpu_linspace  # type: ignore[assignment]
    try:
        core = AutoModelForImageSegmentation.from_pretrained(
            hf_repo,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            _fast_init=False,
            dtype=torch.float32,
            device_map=None,
        )
    finally:
        torch.linspace = _orig_linspace  # type: ignore[assignment]
    core.eval()
    for p in core.parameters():
        p.requires_grad_(False)
    core = core.to(dtype=torch.float32).to("cpu")

    # TorchScript serialization can break if a module's qualified name contains
    # non-identifier path segments (HF dynamic modules include a hash directory
    # like `...BiRefNet.<hash>.birefnet` where `<hash>` starts with a digit).
    #
    # TorchScript emits those names into generated Python-like source, and the
    # parser then fails during `torch.jit.load`. We rewrite *all* classes coming
    # from that dynamic module to a safe identifier.
    try:
        import sys

        orig_mod_name = core.__class__.__module__
        mod = sys.modules.get(orig_mod_name, None)
        if mod is not None:
            for name in dir(mod):
                obj = getattr(mod, name, None)
                if isinstance(obj, type) and getattr(obj, "__module__", None) == orig_mod_name:
                    try:
                        obj.__module__ = "birefnet_local"
                    except Exception:
                        pass

        core.__class__.__module__ = "birefnet_local"
    except Exception:
        pass

    class BiRefNetWrapper(torch.nn.Module):
        """
        Normalize BiRefNet outputs to a single logits tensor.
        """

        def __init__(self, m: torch.nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.m(x)

            # HF ModelOutput / dataclass commonly uses `.logits`
            if hasattr(out, "logits"):
                y = out.logits
            elif isinstance(out, dict):
                y = out.get("logits", None)
                if y is None:
                    # fall back to last tensor value
                    y = None
                    for v in out.values():
                        if isinstance(v, torch.Tensor):
                            y = v
                    if y is None:
                        raise RuntimeError("BiRefNet dict output contained no tensors.")
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                # final stage output contains the matte/logits
                y = out[-1]
            else:
                y = out

            if isinstance(y, (list, tuple)) and len(y) > 0:
                y = y[-1]

            if not isinstance(y, torch.Tensor):
                raise RuntimeError(f"BiRefNet output is not a tensor: {type(y)}")

            # Ensure single-channel logits if model returns multi-channel
            if y.ndim == 4 and y.shape[1] != 1:
                y = y[:, :1, :, :]

            return y.to(dtype=torch.float32)

    wrapped = BiRefNetWrapper(core).eval().to("cpu")
    dummy = torch.randn(1, 3, int(input_res), int(input_res), dtype=torch.float32, device="cpu")

    traced = torch.jit.trace(wrapped, dummy, strict=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BiRefNet to TorchScript (CPU trace, float32, batch=1).")
    parser.add_argument(
        "--hf-repo",
        default="ZhengPeng7/BiRefNet",
        help="Hugging Face repo id (default: ZhengPeng7/BiRefNet).",
    )
    parser.add_argument(
        "--input-res",
        default=1088,
        type=int,
        help="Square trace resolution (must match pipeline TARGET_SIZE).",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "models" / "birefnet.torchscript"),
        help="Destination TorchScript path used by the pipeline.",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    export_birefnet_torchscript(out_path=out_path, hf_repo=str(args.hf_repo), input_res=int(args.input_res))

    _ = torch.jit.load(str(out_path), map_location="cpu")
    print(f"OK. Saved TorchScript model to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


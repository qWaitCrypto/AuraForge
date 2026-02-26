from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .builtins import _maybe_bool, _maybe_int, _require_str, _resolve_in_project
from .runtime import ToolRuntimeError


def _require_text_or_path(args: dict[str, Any]) -> tuple[str | None, str | None]:
    text = args.get("text")
    path = args.get("path")
    if text is None and path is None:
        raise ValueError("Missing input: provide exactly one of 'text' or 'path'.")
    if text is not None and path is not None:
        raise ValueError("Invalid input: provide only one of 'text' or 'path', not both.")
    if text is not None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid 'text' (expected non-empty string).")
        return text, None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Invalid 'path' (expected non-empty string).")
    return None, path.strip()


def _parse_device(args: dict[str, Any]) -> str:
    device = args.get("device") or "auto"
    if not isinstance(device, str):
        raise ValueError("Invalid 'device' (expected string: auto|cpu|cuda).")
    device = device.strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Invalid 'device' (expected: auto|cpu|cuda).")
    return device


def _labels_for_binary_classifier() -> tuple[dict[int, str], dict[int, str]]:
    labels_zh = {0: "人类", 1: "AI"}
    labels_en = {0: "Human", 1: "AI"}
    return labels_zh, labels_en


def _pick_torch_device(*, torch, requested: str) -> tuple[Any, str | None]:
    """
    Return (torch.device, note).

    For requested=auto:
    - prefers CUDA if available AND supported by this torch build for the current GPU
    - otherwise falls back to CPU
    """

    if requested == "cpu":
        return torch.device("cpu"), None

    def _cuda_supported() -> tuple[bool, str | None]:
        import warnings

        # torch.cuda.is_available() may emit noisy warnings about unsupported GPU archs.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cuda capability.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*not compatible with the current PyTorch installation.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*no kernel image is available.*", category=UserWarning)
            try:
                if not torch.cuda.is_available():
                    return False, "CUDA is not available."
            except Exception as e:
                return False, f"CUDA probe failed: {type(e).__name__}: {e}"

        try:
            major, minor = torch.cuda.get_device_capability(0)
            gpu_arch = f"sm_{major}{minor}"
        except Exception as e:
            return False, f"Could not read GPU capability: {type(e).__name__}: {e}"

        try:
            arch_list = list(torch.cuda.get_arch_list() or [])
        except Exception:
            arch_list = []

        if arch_list and gpu_arch not in arch_list:
            return (
                False,
                f"GPU arch {gpu_arch} is not supported by this PyTorch build (supported: {' '.join(arch_list)}).",
            )
        return True, None

    if requested == "cuda":
        ok, note = _cuda_supported()
        if not ok:
            raise ToolRuntimeError(
                "Requested CUDA but this environment cannot run CUDA kernels on the detected GPU.\n"
                f"- detail: {note}\n"
                "- Fix: install a PyTorch build that supports your GPU, or run with device='cpu'."
            )
        return torch.device("cuda"), None

    if requested == "auto":
        ok, note = _cuda_supported()
        if ok:
            return torch.device("cuda"), None
        return torch.device("cpu"), note

    raise ValueError(f"Unsupported device: {requested!r}")


def _is_cuda_no_kernel_image_error(exc: BaseException) -> bool:
    msg = str(exc)
    return ("no kernel image is available" in msg) or ("cudaErrorNoKernelImageForDevice" in msg)


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        root_resolved = root.resolve()
        path_resolved = path.resolve()
    except Exception:
        return False
    return path_resolved == root_resolved or root_resolved in path_resolved.parents


def _find_aura_install_root() -> Path | None:
    """
    Best-effort: find a directory above the installed package that contains AIGC_detector/.
    This is used as the default model location so writing projects don't need to vendor the model.
    """

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "AIGC_detector").is_dir():
            return parent
    return None


def _ensure_torchvision_interpolation_mode() -> None:
    """
    Work around extremely old torchvision installs (e.g. 0.2.0) that break modern transformers imports.

    transformers may import `torchvision.transforms.InterpolationMode` during package import even for text-only usage.
    If torchvision is present but too old to define that symbol, patch a minimal compatible enum in-process.
    """

    try:
        import enum
        import sys
        import types

        import torchvision  # type: ignore
        import torchvision.transforms as _tv_transforms  # type: ignore

        # transformers.video_utils may do `from torchvision import io as torchvision_io`.
        if not hasattr(torchvision, "io"):
            io_mod = types.ModuleType("torchvision.io")

            def _not_supported(*_a, **_kw):
                raise RuntimeError("torchvision.io is not available in this environment (shim injected for transformers).")

            io_mod.read_image = _not_supported  # type: ignore[attr-defined]
            io_mod.read_video = _not_supported  # type: ignore[attr-defined]
            io_mod.write_video = _not_supported  # type: ignore[attr-defined]
            torchvision.io = io_mod  # type: ignore[attr-defined]
            sys.modules.setdefault("torchvision.io", io_mod)

        existing = getattr(_tv_transforms, "InterpolationMode", None)
        if existing is not None and hasattr(existing, "NEAREST_EXACT"):
            return

        class InterpolationMode(enum.Enum):
            NEAREST = 0
            NEAREST_EXACT = 1
            BILINEAR = 2
            BICUBIC = 3
            BOX = 4
            HAMMING = 5
            LANCZOS = 6

        _tv_transforms.InterpolationMode = InterpolationMode  # type: ignore[attr-defined]
    except Exception:
        # If torchvision isn't installed, or patching fails, just continue.
        # transformers will behave as usual; if it crashes, we surface the import error.
        return


def _infer_max_length(tokenizer, model) -> int:
    max_len = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = getattr(tokenizer, "model_max_length", 512)
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max > 0:
        max_len = min(max_len, tokenizer_max)
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = 512
    return int(max_len)


def _encode_chunked(*, tokenizer, text: str, max_length: int, stride: int):
    import torch

    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    special_n = int(getattr(tokenizer, "num_special_tokens_to_add")(pair=False))
    body_len = int(max_length) - special_n
    if body_len <= 0:
        raise RuntimeError(f"Model max_length={max_length} is too small for {special_n} special tokens.")

    stride_i = max(0, int(stride))
    if stride_i >= body_len:
        raise ValueError(f"Invalid 'stride' ({stride_i}); must be < {body_len}.")
    step = body_len - stride_i

    chunks: list[list[int]] = []
    for start in range(0, max(1, len(token_ids)), step):
        body = token_ids[start : start + body_len]
        if not body and start > 0:
            break
        chunks.append(body)

    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    token_type_ids_list: list[list[int]] = []
    token_counts: list[int] = []

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, int):
        pad_id = 0

    for body in chunks:
        full_ids: list[int] = tokenizer.build_inputs_with_special_tokens(body)
        tti: list[int] = tokenizer.create_token_type_ids_from_sequences(body)
        if len(full_ids) != len(tti):
            tti = [0] * len(full_ids)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            tti = tti[:max_length]

        pad_n = max_length - len(full_ids)
        token_counts.append(len(full_ids))
        input_ids_list.append(full_ids + [pad_id] * pad_n)
        attention_mask_list.append([1] * len(full_ids) + [0] * pad_n)
        token_type_ids_list.append(tti + [0] * pad_n)

    batch: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids_list, dtype=torch.long),
    }
    return batch, token_counts


@dataclass(slots=True)
class ProjectAIGCDetectTool:
    name: str = "project__aigc_detect"
    description: str = (
        "Detect whether a text is likely AI-generated using a local (offline) classifier under the project root.\n"
        "\n"
        "Inputs (provide exactly one):\n"
        "- text: inline string\n"
        "- path: relative UTF-8 file path under the project root\n"
        "\n"
        "Model loading:\n"
        "- Loads from model_dir with local_files_only=True (no network).\n"
        "- If model_dir is omitted, it defaults to AIGC_detector/ under the Aura installation root.\n"
        "- model_dir may be a project-relative path, or an absolute path under the project root / Aura install root.\n"
        "- The model is cached in-memory per session after first use to speed up subsequent calls.\n"
        "\n"
        "Long text handling:\n"
        "- If the text exceeds the model window, it is split into token chunks.\n"
        "- Each chunk produces probabilities via softmax(logits).\n"
        "- Final probabilities are a token-weighted average across chunks (weights=non-pad tokens).\n"
        "- stride controls token overlap between adjacent chunks (0 = no overlap).\n"
        "\n"
        "Output interpretation:\n"
        "- This is a best-effort signal for writing review, not factual proof.\n"
        "- probs is ordered as [Human, AI]. pred_label/pred_score reflect the max-prob class."
    )
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Inline text to detect (exclusive with path)."},
                "path": {"type": "string", "description": "Relative file path under project root (exclusive with text)."},
                "model_dir": {
                    "type": "string",
                    "description": "Relative directory containing the local HuggingFace model (default: AIGC_detector).",
                },
                "device": {
                    "type": "string",
                    "enum": ["auto", "cpu", "cuda"],
                    "description": "Device selection (auto picks CUDA if available).",
                },
                "fp16": {"type": "boolean", "description": "Use fp16 on CUDA (default false). Ignored on CPU."},
                "stride": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Token overlap between adjacent chunks (default 0). Must be < chunk body length.",
                },
            },
            "additionalProperties": False,
        }
    )

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cache: dict[tuple[str, str, bool], tuple[Any, Any, int]] = field(default_factory=dict, init=False, repr=False)

    def execute(self, *, args: dict[str, Any], project_root: Path) -> dict[str, Any]:
        text, rel_path = _require_text_or_path(args)
        device_choice = _parse_device(args)

        model_dir_given = "model_dir" in args and args.get("model_dir") is not None
        model_dir_rel: str
        if model_dir_given:
            model_dir_rel = args.get("model_dir")  # type: ignore[assignment]
            if not isinstance(model_dir_rel, str) or not model_dir_rel.strip():
                raise ValueError("Invalid 'model_dir' (expected non-empty string).")
            model_dir_rel = model_dir_rel.strip()
        else:
            install_root = _find_aura_install_root()
            if install_root is None:
                raise FileNotFoundError(
                    "Default model directory not found under the Aura installation. "
                    "Provide 'model_dir' explicitly."
                )
            model_dir_rel = str((install_root / "AIGC_detector").resolve())

        stride = _maybe_int(args, "stride") or 0
        if stride < 0:
            raise ValueError("Invalid 'stride' (expected integer >= 0).")
        fp16 = _maybe_bool(args, "fp16") or False

        model_dir_path = Path(model_dir_rel).expanduser()
        if model_dir_path.is_absolute():
            model_dir = model_dir_path.resolve()
            if not _is_under_root(model_dir, project_root):
                install_root = _find_aura_install_root()
                if install_root is None or not _is_under_root(model_dir, install_root):
                    raise PermissionError(
                        "Absolute model_dir must be under the project root or the Aura installation root."
                    )
        else:
            model_dir = _resolve_in_project(project_root, model_dir_rel)
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir_rel}")

        source: dict[str, Any]
        if rel_path is not None:
            file_path = _resolve_in_project(project_root, rel_path)
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"File not found: {rel_path}")
            data = file_path.read_bytes()
            text = data.decode("utf-8", errors="replace")
            source = {"kind": "file", "path": str(Path(rel_path))}
        else:
            source = {"kind": "inline"}

        # Lazy imports: keep this tool optional.
        try:
            import torch
        except Exception as e:  # pragma: no cover
            import sys

            raise ToolRuntimeError(
                "project__aigc_detect requires the optional dependency 'torch', but it could not be imported.\n"
                f"- python: {sys.executable}\n"
                f"- import error: {type(e).__name__}: {e}"
            ) from e

        _ensure_torchvision_interpolation_mode()

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as e:  # pragma: no cover
            import sys

            raise ToolRuntimeError(
                "project__aigc_detect requires the optional dependency 'transformers', but it could not be imported.\n"
                f"- python: {sys.executable}\n"
                f"- import error: {type(e).__name__}: {e}\n"
                "Install it in the same environment used to run 'aura chat'."
            ) from e

        device, device_note = _pick_torch_device(torch=torch, requested=device_choice)

        cache_key = (str(model_dir), str(device), bool(fp16 and device.type == "cuda"))
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is None:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                use_safetensors = (model_dir / "model.safetensors").is_file() or (model_dir / "pytorch_model.safetensors").is_file()
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_dir,
                        local_files_only=True,
                        use_safetensors=use_safetensors,
                    )
                except ValueError as e:
                    msg = str(e)
                    if "CVE-2025-32434" in msg or "check_torch_load_is_safe" in msg or "upgrade torch to at least v2.6" in msg:
                        raise ToolRuntimeError(
                            "Failed to load the local model weights with the current torch/transformers versions.\n"
                            "- transformers is refusing to call torch.load on .bin weights unless torch>=2.6 (CVE gate).\n"
                            "- Fix options:\n"
                            "  1) Upgrade torch to >=2.6, or\n"
                            "  2) Downgrade transformers to a version that doesn't enforce this, or\n"
                            "  3) Convert AIGC_detector/pytorch_model.bin to safetensors (model.safetensors) and retry.\n"
                            f"- model_dir: {model_dir}\n"
                            f"- original error: {type(e).__name__}: {e}"
                        ) from e
                    raise
                model.eval()
                try:
                    model.to(device)
                except Exception as e:
                    # If auto-selected CUDA but runtime kernels are not supported, transparently fall back to CPU.
                    msg = str(e)
                    if device_choice == "auto" and device.type == "cuda" and (
                        "no kernel image is available" in msg or "cudaErrorNoKernelImageForDevice" in msg
                    ):
                        device = torch.device("cpu")
                        device_note = "Fell back to CPU due to unsupported CUDA kernels for this GPU/torch build."
                        cache_key = (str(model_dir), str(device), False)
                        model.to(device)
                    else:
                        raise
                if cache_key[2]:
                    model.half()
                max_len = _infer_max_length(tokenizer, model)
                self._cache[cache_key] = (tokenizer, model, max_len)
            else:
                tokenizer, model, max_len = cached

        batch_cpu, token_counts = _encode_chunked(tokenizer=tokenizer, text=text or "", max_length=max_len, stride=stride)
        model_input_names = getattr(tokenizer, "model_input_names", None) or ["input_ids", "attention_mask", "token_type_ids"]
        batch = {k: batch_cpu[k].to(device) for k in model_input_names if k in batch_cpu}
        if "input_ids" not in batch:
            raise RuntimeError("Tokenizer did not produce input_ids for model inputs.")

        with torch.inference_mode():
            logits = model(**batch).logits
            probs_t = torch.softmax(logits, dim=-1).detach().float().cpu()

            if "attention_mask" in batch:
                weights = batch["attention_mask"].detach().float().cpu().sum(dim=1)
            else:
                weights = torch.ones(probs_t.shape[0], dtype=torch.float32)

            denom = float(weights.sum().item()) if weights.numel() else 0.0
            if denom > 0:
                probs_weighted = (probs_t * weights.unsqueeze(1)).sum(dim=0) / denom
            else:
                probs_weighted = probs_t.mean(dim=0)

            probs = probs_weighted.tolist()
            pred = int(max(range(len(probs)), key=lambda i: probs[i]))
            score = float(probs[pred])

        labels_zh, labels_en = _labels_for_binary_classifier()
        return {
            "ok": True,
            "source": source,
            "model_dir": str(Path(model_dir_rel)),
            "resolved_model_dir": str(model_dir),
            "device_requested": device_choice,
            "device": str(device),
            "device_note": device_note,
            "max_length": int(max_len),
            "stride": int(stride),
            "n_chunks": int(len(token_counts)),
            "chunk_token_counts": [int(x) for x in token_counts],
            "total_tokens_used": int(sum(int(x) for x in token_counts)),
            "aggregation": "weighted_mean(probabilities, by_attention_mask_tokens)",
            "label_order": [labels_en[0], labels_en[1]],
            "label_order_zh": [labels_zh[0], labels_zh[1]],
            "pred_label_id": pred,
            "pred_label": labels_zh.get(pred, str(pred)),
            "pred_label_en": labels_en.get(pred, str(pred)),
            "pred_score": score,
            "probs": probs,
            "text_preview": (text or "")[:200],
        }

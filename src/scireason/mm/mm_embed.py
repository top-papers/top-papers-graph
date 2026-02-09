from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from ..config import settings


MMBackend = Literal["none", "open_clip"]


def _require(pkg: str) -> None:
    raise RuntimeError(
        f"Для MM-эмбеддингов нужна зависимость '{pkg}'.\n"
        "Установите extras: pip install -e '.[mm]'\n"
    )


@lru_cache(maxsize=1)
def _open_clip_model():
    try:
        import open_clip  # type: ignore
        import torch  # type: ignore
    except Exception:
        _require("open_clip_torch/torch")

    model_name = getattr(settings, "open_clip_model", "ViT-B-32")
    pretrained = getattr(settings, "open_clip_pretrained", "laion2b_s34b_b79k")
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device


def embed_text(texts: Sequence[str], backend: Optional[MMBackend] = None) -> List[List[float]]:
    backend = backend or getattr(settings, "mm_embed_backend", "none")
    if backend == "none":
        raise RuntimeError("MM embeddings backend=none. Set MM_EMBED_BACKEND=open_clip and install extras.[mm]")
    if backend != "open_clip":
        raise ValueError(f"Unknown backend: {backend}")

    model, _preprocess, tokenizer, device = _open_clip_model()
    import torch  # type: ignore

    with torch.no_grad():
        tokens = tokenizer(list(texts)).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()


def embed_images(image_paths: Sequence[Path], backend: Optional[MMBackend] = None) -> List[List[float]]:
    backend = backend or getattr(settings, "mm_embed_backend", "none")
    if backend == "none":
        raise RuntimeError("MM embeddings backend=none. Set MM_EMBED_BACKEND=open_clip and install extras.[mm]")
    if backend != "open_clip":
        raise ValueError(f"Unknown backend: {backend}")

    model, preprocess, _tokenizer, device = _open_clip_model()
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    imgs = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()

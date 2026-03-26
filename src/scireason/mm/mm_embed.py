from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from ..config import settings
from ..llm import embed as text_embed


MMBackend = Literal['none', 'open_clip', 'hash']


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
        _require('open_clip_torch/torch')

    model_name = getattr(settings, 'open_clip_model', 'ViT-B-32')
    pretrained = getattr(settings, 'open_clip_pretrained', 'laion2b_s34b_b79k')
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer, device


def _hash_image_embed(image_paths: Sequence[Path]) -> List[List[float]]:
    from PIL import Image  # type: ignore

    # 16 x 8 x 3 = 384 dims, matching the default hash text embedding size.
    size = (16, 8)
    vecs: List[List[float]] = []
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize(size)
        raw = list(img.getdata())
        vec = [float(channel) / 255.0 for pixel in raw for channel in pixel]
        vecs.append(vec)
    return vecs


def embed_text(texts: Sequence[str], backend: Optional[MMBackend] = None) -> List[List[float]]:
    backend = backend or getattr(settings, 'mm_embed_backend', 'none')
    if backend == 'none':
        raise RuntimeError("MM embeddings backend=none. Set MM_EMBED_BACKEND=open_clip or hash.")
    if backend == 'hash':
        return text_embed(list(texts))
    if backend != 'open_clip':
        raise ValueError(f'Unknown backend: {backend}')

    model, _preprocess, tokenizer, device = _open_clip_model()
    import torch  # type: ignore

    with torch.no_grad():
        tokens = tokenizer(list(texts)).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()


def embed_images(image_paths: Sequence[Path], backend: Optional[MMBackend] = None) -> List[List[float]]:
    backend = backend or getattr(settings, 'mm_embed_backend', 'none')
    if backend == 'none':
        raise RuntimeError("MM embeddings backend=none. Set MM_EMBED_BACKEND=open_clip or hash.")
    if backend == 'hash':
        return _hash_image_embed(image_paths)
    if backend != 'open_clip':
        raise ValueError(f'Unknown backend: {backend}')

    model, preprocess, _tokenizer, device = _open_clip_model()
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    imgs = [preprocess(Image.open(p).convert('RGB')) for p in image_paths]
    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().tolist()

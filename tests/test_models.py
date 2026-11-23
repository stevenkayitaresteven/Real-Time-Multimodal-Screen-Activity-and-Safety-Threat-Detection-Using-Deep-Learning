import pytest
import torch

from screenguard.config import ModelConfig
from screenguard.models.losses import MultiTaskLoss
from screenguard.models.multitask import ExportWrapper, build_model
from screenguard.taxonomy import IGNORE_INDEX, TASKS

IMG = (64, 112)
L = 128


def _batch(b=3, empty_text_row=True):
    x = torch.randn(b, 3, *IMG)
    ids = torch.randint(4, 200, (b, L))
    ids[:, 0] = 1
    ids[:, 80:] = 0
    if empty_text_row:
        ids[0, 1:] = 0  # a frame where OCR found nothing
    return x, ids


def _cfg(**kw):
    base = dict(dim=32, text_depth=1, text_heads=2, fusion_depth=1, fusion_heads=2, vit_dim=48, vit_depth=1, vit_heads=2)
    base.update(kw)
    return ModelConfig(**base)


@pytest.mark.parametrize("vision", ["tiny_cnn", "vit_tiny", "mobilenet_v3_small"])
@pytest.mark.parametrize("fusion", ["concat", "gated", "attention"])
def test_forward_shapes_and_backward(vision, fusion):
    torch.manual_seed(0)
    m = build_model(_cfg(vision=vision, fusion=fusion), TASKS, IMG, L)
    x, ids = _batch()
    out = m(x, ids)
    for t, classes in TASKS.items():
        assert out[t].shape == (3, len(classes))
        assert torch.isfinite(out[t]).all()
    sum(o.sum() for o in out.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.vision.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.text.parameters())


@pytest.mark.parametrize("modalities", [["image"], ["text"]])
@pytest.mark.parametrize("fusion", ["concat", "gated", "attention"])
def test_single_modality_ablations(modalities, fusion):
    m = build_model(_cfg(fusion=fusion, modalities=modalities), TASKS, IMG, L).eval()
    x, ids = _batch()
    out = m(x if "image" in modalities else None, ids if "text" in modalities else None)
    assert out["threat"].shape == (3, len(TASKS["threat"])) and torch.isfinite(out["threat"]).all()
    assert (m.vision is None) == ("image" not in modalities) and (m.text is None) == ("text" not in modalities)


def test_modality_dropout_only_in_training():
    m = build_model(_cfg(modality_dropout=1.0), TASKS, IMG, L)
    x, ids = _batch()
    m.eval()
    a = m(x, ids)["threat"]
    b = m(x, ids)["threat"]
    assert torch.allclose(a, b)
    m.train()
    out = m(x, ids)
    assert torch.isfinite(out["threat"]).all()


def test_export_wrapper_returns_tuple_in_task_order():
    m = build_model(_cfg(), TASKS, IMG, L).eval()
    x, ids = _batch()
    outs = ExportWrapper(m)(x, ids)
    assert len(outs) == len(TASKS) and outs[2].shape[-1] == len(TASKS["threat"])


def test_masked_multitask_loss():
    logits = {t: torch.randn(4, len(c), requires_grad=True) for t, c in TASKS.items()}
    targets = {"activity": torch.tensor([0, 1, IGNORE_INDEX, 2]),
               "nsfw": torch.full((4,), IGNORE_INDEX),  # no NSFW labels in this batch
               "threat": torch.tensor([0, 1, 2, 3])}
    crit = MultiTaskLoss(TASKS, "uncertainty")
    loss, parts = crit(logits, targets)
    assert torch.isfinite(loss) and set(parts) == {"activity", "threat"}
    loss.backward()
    assert logits["nsfw"].grad is None or logits["nsfw"].grad.abs().sum() == 0
    assert crit.log_vars["activity"].grad is not None
    fixed, _ = MultiTaskLoss(TASKS, "fixed", {"activity": 2.0})(logits, targets)
    assert torch.isfinite(fixed)


@pytest.mark.parametrize("stem", ["ngram", "strided"])
def test_text_stems_and_masks(stem):
    from screenguard.models.text import ByteTextEncoder

    enc = ByteTextEncoder(32, 1, 2, max_len=256, downsample=8, stem=stem)
    ids = torch.zeros(2, 256, dtype=torch.long)
    ids[:, 0] = 1
    ids[0, 1:200] = 70
    tokens, valid, has_text = enc(ids)
    assert tokens.shape[:2] == valid.shape and torch.isfinite(tokens).all()
    assert has_text.tolist() == [True, False]
    assert valid[0].sum() > valid[1].sum() == 1  # empty row keeps only the CLS token

import pytest
import torch

from screenguard.config import Config
from screenguard.models.multitask import build_model
from screenguard.taxonomy import TASKS
from screenguard.training import save_checkpoint

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = Config()
    cfg.data.img_size = [72, 128]
    cfg.data.max_text_len = 256
    cfg.model.dim = 32
    cfg.model.text_depth = 1
    cfg.model.fusion_depth = 1
    cfg.model.text_heads = 2
    cfg.model.fusion_heads = 2
    return cfg


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, tiny_cfg):
    """An untrained but structurally complete checkpoint (fast to build)."""
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model, TASKS, tuple(tiny_cfg.data.img_size), tiny_cfg.data.max_text_len)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    save_checkpoint(str(path), model, tiny_cfg, {"tasks_trained": ["activity", "threat"],
                                                 "temperatures": {"activity": 1.0, "threat": 1.0}})
    return str(path)

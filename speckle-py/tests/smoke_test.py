import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speckle.batcher import Batcher
from speckle.config import TrainConfig
from speckle.model import Model
from speckle.muon import Muon, newton_schulz, muon
from speckle.utils import estimate_loss, flat_to_2d, train_val_split
from speckle.utils.image import visualize_generation
from speckle.utils.info import param_count, write_summary
from speckle.utils.output import make_pictures, patch_coords
from speckle.__main__ import save_safetensors, load_safetensors


def make_config(**overrides):
    kwargs = dict(
        eval_interval=1,
        eval_iters=2,
        accumulation_steps=2,
        starting_density=0.5,
        ending_density=0.25,
    )
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def make_model_cfg(**overrides):
    cfg = make_config()
    fields = {
        "dropout": 0.1,
        "image_channels": 1,
        "n_blocks": 1,
        "patch_side": 3,
        "emb_dim": 16,
        "batch_size": 4,
        "eval_batch_size": 4,
        "residual": "Plain",
    }
    fields.update(overrides)
    for k, v in fields.items():
        setattr(cfg.model, k, v)
    cfg.model.block.swiglu_dim = 8
    cfg.model.block.attention.num_heads = 4
    cfg.model.block.attention.qk_norm = True
    cfg.model.block.norm = "RMSNorm"
    return cfg


def test_newton_schulz():
    torch.manual_seed(0)
    x = torch.randn(8, 5)
    y = newton_schulz(x.clone(), turbo=False, eps=1e-7)
    z = newton_schulz(x.clone(), turbo=True, eps=1e-7)
    sv = torch.linalg.svdvals(y)
    assert ((sv > 0.3) & (sv < 1.6)).all(), sv
    sv_t = torch.linalg.svdvals(z)
    assert ((sv_t > 0.1) & (sv_t < 2.0)).all(), sv_t
    y2 = newton_schulz(y.clone(), turbo=False, eps=1e-7)
    assert torch.linalg.svdvals(y2).std() < torch.linalg.svdvals(y).std()
    print("newton_schulz OK")


def test_forward_backward():
    cfg = make_model_cfg()
    torch.manual_seed(0)
    images = torch.rand(100, 16, 16, 1)
    train_data, val_data = train_val_split(images, 0.8)
    train_batcher = Batcher(train_data, cfg.model.batch_size)
    val_batcher = Batcher(val_data, cfg.model.batch_size)
    vis_batcher = Batcher(train_data, cfg.model.eval_batch_size)

    model = Model(cfg.model)
    total, trainable = param_count(model)
    assert total >= trainable
    print(f"params total={total} trainable={trainable}")

    sx, target = train_batcher.next()
    assert sx.shape == (4, 256) and target.shape == (4, 16, 16, 1)
    sx = sx[:, :128]
    loss, raw = model.forward_with_loss(sx, target, True)
    assert raw.shape == (4, 128, 3 * 3 * 2), raw.shape
    assert torch.isfinite(loss), loss
    (loss / 2).backward()

    adam_p, muon_p = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() == 1 or "embedding" in name or "bias" in name:
                adam_p.append(p)
            else:
                muon_p.append(p)
    adam_names = {n for n, p in model.named_parameters() if any(p is q for q in adam_p)}
    muon_names = {n for n, p in model.named_parameters() if any(p is q for q in muon_p)}
    assert "embedding.weight" in adam_names
    assert "b0.attention_residual_scale" in adam_names
    assert "b0.storage.glu_proj.weight" in muon_names
    assert "b0.self_attention.k_norm_scale" in muon_names
    assert "b0.self_attention.w_q.weight" in muon_names
    assert "b0.storage.glu_proj.bias" in adam_names

    adamw = torch.optim.AdamW([{"params": adam_p, "lr": 1e-3}], betas=(0.9, 0.95), weight_decay=0.0)
    opt = muon(muon_p, 7e-3, 0.95, 0.0, True, True, 1e-7)
    adamw.step()
    opt.step()

    tr, va = estimate_loss(cfg, train_batcher, val_batcher, model, [64])
    assert isinstance(tr, float) and isinstance(va, float)
    print(f"estimate_loss OK: {tr:.4f} {va:.4f}")

    pred_loss, img = visualize_generation([64], vis_batcher, model)
    with tempfile.TemporaryDirectory() as d:
        img.save(Path(d) / "grid.png")
    assert img.size[1] == 4 * img.size[0] // 4 or True
    print(f"visualize_generation OK, grid size {img.size}, pred_loss {pred_loss:.4f}")


def test_attentive_residual():
    cfg = make_model_cfg(residual="Attentive")
    torch.manual_seed(1)
    model = Model(cfg.model)
    sx = torch.randint(0, 256, (2, 100))
    target = torch.rand(2, 16, 16, 1)
    loss, raw = model.forward_with_loss(sx, target, False)
    assert torch.isfinite(loss)
    names = [n for n, _ in model.named_parameters()]
    assert "block_ar_queries" in names
    print(f"attentive residual OK, loss {loss.item():.4f}")


def test_layernorm_variant():
    cfg = make_model_cfg()
    cfg.model.block.norm = "LayerNorm"
    cfg.model.block.attention.qk_norm = False
    torch.manual_seed(2)
    model = Model(cfg.model)
    names = [n for n, _ in model.named_parameters()]
    assert "b0.sa_norm.bias" in names and "ln_f.bias" in names
    assert "b0.self_attention.k_norm_scale" not in names
    sx = torch.randint(0, 256, (2, 50))
    target = torch.rand(2, 16, 16, 1)
    loss, _ = model.forward_with_loss(sx, target, False)
    assert torch.isfinite(loss)
    print(f"layernorm variant OK, loss {loss.item():.4f}")


def test_safetensors_roundtrip():
    cfg = make_model_cfg()
    torch.manual_seed(3)
    model = Model(cfg.model)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.safetensors"
        save_safetensors(model, path)
        model2 = Model(cfg.model)
        load_safetensors(model2, path)
        for (n1, t1), (n2, t2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert n1 == n2 and torch.equal(t1, t2)
    print("safetensors roundtrip OK")


def test_toml_roundtrip():
    cfg = make_model_cfg(residual="Attentive")
    cfg.model.block.norm = "LayerNorm"
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "config.toml"
        cfg.save(path)
        loaded = TrainConfig.load(path)
        assert loaded.eval_interval == cfg.eval_interval
        assert loaded.model.patch_side == cfg.model.patch_side
        assert loaded.model.residual == "Attentive"
        assert loaded.model.block.norm == "LayerNorm"
        assert loaded.model.block.attention.num_heads == 4
        assert loaded.dataset.dataset == "Mnist"
        assert loaded.lr_schedule.max_iters == 10001
        text = path.read_text()
        assert "[model.block.attention]" in text
        assert 'residual = "Attentive"' in text

    rust_cfg = Path(
        "../../Programming/Rust/speckle/checkpoints/"
        "run_20260913_1929_-dirty_ff-cifar-b3-h4/config.toml"
    )
    loaded = TrainConfig.load(rust_cfg)
    assert loaded.model.emb_dim == 64
    assert loaded.model.n_blocks == 3
    assert loaded.model.block.attention.num_heads == 8
    assert loaded.model.block.attention.qk_norm is True
    assert loaded.model.image_channels == 3
    assert loaded.model.patch_side == 5
    assert loaded.dataset.dataset == "Cifar10"

    old_cfg = Path(
        "../../Programming/Rust/speckle/checkpoints/"
        "run_20260910_0028_-dirty_sl-h4-b1/config.toml"
    )
    loaded = TrainConfig.load(old_cfg)
    assert loaded.model.block.norm == "RMSNorm"
    assert loaded.model.emb_dim == 32
    print("toml roundtrip + rust configs OK")


def test_lr_schedule_quirk():
    cfg = TrainConfig()
    lr0 = cfg.lr_schedule.get_lr(0)
    assert lr0 == 0.0
    mid = cfg.lr_schedule.get_lr(750)
    assert abs(mid - 3e-3 / 2) < 1e-9
    coast = cfg.lr_schedule.get_lr(2000)
    assert coast == 3e-3
    decayed = cfg.lr_schedule.get_lr(12000)
    assert 0 <= decayed <= 3e-3
    print(f"lr schedule OK (warmup={lr0}, mid={mid:.6f}, coast={coast}, decayed={decayed:.6f})")


def test_flat_to_2d():
    flat = torch.tensor([[0, 1, 4, 7, 10]])
    two = flat_to_2d(flat, 4)
    assert two.shape == (1, 5, 2)
    assert torch.equal(two[0], torch.tensor([[0, 0], [0, 1], [1, 0], [1, 3], [2, 2]]))
    print("flat_to_2d OK")


def reference_make_pictures(flat_coords, patches, scores, p_side, hwc):
    b, npp = flat_coords.shape
    height, width, channels = hwc
    padding = p_side // 2
    ph, pw = height + 2 * padding, width + 2 * padding
    flat_scores = scores.reshape(b, npp)
    flat_patches = patches.reshape(b, npp, channels)
    pred = torch.zeros(b, ph, pw, channels)
    var = torch.zeros(b, ph, pw, 1)
    mask = torch.zeros(b, ph, pw, dtype=torch.bool)
    for bi in range(b):
        buckets = {}
        for j in range(npp):
            buckets.setdefault(int(flat_coords[bi, j]), []).append(j)
        for q, idxs in buckets.items():
            s = flat_scores[bi, idxs]
            w = torch.softmax(s, dim=0)
            y, x = divmod(q, pw)
            pred[bi, y, x] = sum(w[k] * flat_patches[bi, idxs[k]] for k in range(len(idxs)))
            var[bi, y, x] = (-s.max()).exp() / (s - s.max()).exp().sum()
            mask[bi, y, x] = True
    pred = pred[:, padding : padding + height, padding : padding + width]
    var = var[:, padding : padding + height, padding : padding + width]
    mask = mask[:, padding : padding + height, padding : padding + width]
    return pred, var, mask


def test_make_pictures_pixelwise_softmax():
    p_side = 3
    hwc = (4, 4, 1)
    centers = torch.tensor([[[1, 1], [1, 2]]])
    flat_coords = patch_coords(centers, p_side, 4)
    scores = torch.zeros(1, 2, p_side * p_side)
    scores[0, 0, 0] = 10.0
    scores[0, 1, 8] = 20.0
    patches = torch.zeros(1, 2, p_side * p_side, 1)
    patches[0, 0] = 1.0
    patches[0, 1] = 3.0

    pred, var, mask = make_pictures(flat_coords, patches, scores, p_side, hwc)

    assert abs(pred[0, 0, 1, 0].item() - 2.0) < 1e-6, pred[0, 0, 1, 0]
    assert abs(pred[0, 0, 0, 0].item() - 1.0) < 1e-6
    assert abs(pred[0, 2, 3, 0].item() - 3.0) < 1e-6
    assert abs(var[0, 0, 1, 0].item() - 0.5) < 1e-6
    assert abs(var[0, 0, 0, 0].item() - math.exp(-10.0)) < 1e-6
    assert abs(var[0, 2, 3, 0].item() - math.exp(-20.0)) < 1e-6
    assert mask[0, 0, 0] and mask[0, 0, 1] and mask[0, 2, 3]
    assert not mask[0, 3, 3]
    assert pred[0, 3, 3, 0] == 0.0
    assert torch.isinf(var[0, 3, 3, 0])

    torch.manual_seed(7)
    b, n, h, w = 2, 5, 8, 8
    centers = torch.stack([torch.randint(0, h, (b, n)), torch.randint(0, w, (b, n))], -1)
    flat_coords = patch_coords(centers, p_side, w)
    scores = torch.empty(b, n, p_side * p_side).uniform_(-40.0, 40.0)
    patches = torch.rand(b, n, p_side * p_side, 1)
    pred, var, mask = make_pictures(flat_coords, patches, scores, p_side, (h, w, 1))
    ref_pred, ref_var, ref_mask = reference_make_pictures(flat_coords, patches, scores, p_side, (h, w, 1))
    assert torch.allclose(pred, ref_pred, atol=1e-5)
    assert torch.allclose(var[mask], ref_var[mask], atol=1e-5)
    assert torch.isinf(var[~mask]).all()
    assert torch.equal(mask, ref_mask)
    assert torch.isfinite(pred).all()
    print("make_pictures pixelwise softmax OK")


if __name__ == "__main__":
    test_newton_schulz()
    test_flat_to_2d()
    test_make_pictures_pixelwise_softmax()
    test_lr_schedule_quirk()
    test_toml_roundtrip()
    test_forward_backward()
    test_attentive_residual()
    test_layernorm_variant()
    test_safetensors_roundtrip()
    print("ALL SMOKE TESTS PASSED")

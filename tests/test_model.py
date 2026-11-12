import numpy as np
import torch

from src.model import count_parameters, make_model


def test_forward_pass_shape():
    model = make_model(seed=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(8, 1, 64, 64))
    assert out.shape == (8, 9)


def test_forward_is_deterministic_in_eval():
    model = make_model(seed=0)
    model.eval()
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.equal(a, b)


def test_softmax_probabilities_sum_to_one():
    model = make_model(seed=0)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(5, 1, 64, 64))
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(1), torch.ones(5), atol=1e-5)


def test_embedding_shape():
    model = make_model(seed=0)
    Z = model.embed(np.zeros((6, 64, 64), dtype=np.float32))
    assert Z.shape == (6, 256)
    assert np.isfinite(Z).all()


def test_model_has_learnable_parameters():
    model = make_model(seed=0)
    n = count_parameters(model)
    assert n > 100_000  # small CNN, sanity bound
    assert any(p.requires_grad for p in model.parameters())

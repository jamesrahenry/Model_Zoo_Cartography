"""Unit tests for train/train_mlp.py (ArcMLP) and train/train_mlp_batched.py
(init_stacked) — the architecture spec itself (README: "must stay analytically
comparable to the null baseline — never change these to rescue convergence")
and FINDINGS.md's "init bit-identical per seed" claim for the batched trainer.

CPU-only (widths/depths kept tiny; no GPU required — torch runs these fine on
CPU, just slower at real corpus scale, which is out of scope for unit tests).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_mlp import ArcMLP, evaluate
from train_mlp_batched import init_stacked


class TestArcMLPArchitecture:
    def test_no_biases_anywhere(self):
        model = ArcMLP(width=8, depth=3, init_seed=0, head_classes=4)
        for lin in model.layers:
            assert lin.bias is None
        assert model.head.bias is None

    def test_layer_shapes(self):
        model = ArcMLP(width=6, depth=5, init_seed=0)
        assert len(model.layers) == 5
        for lin in model.layers:
            assert lin.weight.shape == (6, 6)

    def test_he_gaussian_init_variance(self):
        # N(0, 2/fan_in) per element — check empirically over many layers/seeds
        width = 128
        stds = []
        for seed in range(20):
            model = ArcMLP(width=width, depth=1, init_seed=seed)
            stds.append(model.layers[0].weight.detach().numpy().std())
        expected = (2.0 / width) ** 0.5
        assert np.mean(stds) == pytest.approx(expected, rel=0.03)

    def test_same_seed_gives_bit_identical_init(self):
        m1 = ArcMLP(width=10, depth=3, init_seed=7)
        m2 = ArcMLP(width=10, depth=3, init_seed=7)
        for l1, l2 in zip(m1.layers, m2.layers):
            torch.testing.assert_close(l1.weight, l2.weight)

    def test_different_seeds_give_different_init(self):
        m1 = ArcMLP(width=10, depth=3, init_seed=1)
        m2 = ArcMLP(width=10, depth=3, init_seed=2)
        assert not torch.allclose(m1.layers[0].weight, m2.layers[0].weight)

    def test_head_mode_forward_shape_and_relu_before_head(self):
        model = ArcMLP(width=6, depth=3, init_seed=0, head_classes=4)
        x = torch.randn(5, 6)
        out = model(x)
        assert out.shape == (5, 4)

    def test_columns_mode_returns_pre_relu_full_width(self):
        model = ArcMLP(width=6, depth=3, init_seed=0, head_classes=None)
        x = torch.randn(5, 6)
        out = model(x)
        assert out.shape == (5, 6)  # full width, no head slicing inside forward

    def test_relu_applied_after_every_layer_except_final_in_head_mode(self):
        # architecture doc: "ReLU after every layer" for the censused stack;
        # the head reads the POST-relu final-layer output.
        model = ArcMLP(width=4, depth=2, init_seed=0, head_classes=3)
        x = torch.randn(20, 4)
        # manually replicate: relu after layer0, relu after layer1, then head
        h = torch.relu(model.layers[0](x))
        h = torch.relu(model.layers[1](h))
        expected = model.head(h)
        torch.testing.assert_close(model(x), expected)

    def test_weights_arc_convention_matches_x_at_w_forward(self):
        # weights_arc_convention() stores (in, out); analytic_vacuum.py and
        # manifold_detector.py both assume forward = x @ W with that layout.
        model = ArcMLP(width=5, depth=2, init_seed=0)
        arc_w = model.weights_arc_convention()
        x = torch.randn(7, 5)
        torch_out = torch.relu(model.layers[0](x))
        manual_out = np.maximum(x.numpy() @ arc_w[0], 0.0)
        np.testing.assert_allclose(torch_out.detach().numpy(), manual_out, atol=1e-5)

    def test_head_init_does_not_perturb_stack_init(self):
        # head is documented as initialized AFTER the stack, from the same
        # generator, so two nets with the same seed have identical STACK
        # weights whether or not a head is attached.
        no_head = ArcMLP(width=8, depth=4, init_seed=3, head_classes=None)
        with_head = ArcMLP(width=8, depth=4, init_seed=3, head_classes=5)
        for l1, l2 in zip(no_head.layers, with_head.layers):
            torch.testing.assert_close(l1.weight, l2.weight)


class TestEvaluate:
    def test_perfect_predictions_give_accuracy_one(self):
        model = ArcMLP(width=4, depth=2, init_seed=0, head_classes=4)
        x = torch.randn(50, 4)
        with torch.no_grad():
            y = model(x).argmax(dim=1)  # labels model already "gets right"
        acc = evaluate(model, x, y, n_classes=4)
        assert acc == pytest.approx(1.0)

    def test_random_labels_give_chance_level_accuracy(self):
        torch.manual_seed(0)
        model = ArcMLP(width=8, depth=3, init_seed=0, head_classes=4)
        x = torch.randn(4000, 8)
        y = torch.randint(0, 4, (4000,))
        acc = evaluate(model, x, y, n_classes=4)
        assert 0.15 < acc < 0.35  # near chance (0.25), generous band

    def test_model_left_in_train_mode_after_evaluate(self):
        model = ArcMLP(width=4, depth=2, init_seed=0, head_classes=3)
        model.train()
        evaluate(model, torch.randn(10, 4), torch.randint(0, 3, (10,)), n_classes=3)
        assert model.training is True


class TestBatchedInitMatchesSequential:
    """FINDINGS.md: 'batched trainer validated ... init bit-identical per seed
    across the sequential and batched trainer.' This is the regression test
    for that specific, load-bearing claim."""

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_init_stacked_matches_arc_mlp_stack_layers(self, seed):
        width, depth, n_classes = 12, 4, 5
        layers, heads = init_stacked(width, depth, n_classes, seeds=[seed])
        ref = ArcMLP(width, depth, init_seed=seed, head_classes=n_classes)
        for l in range(depth):
            # ArcMLP stores (out, in) nn.Linear weight; the batched trainer's
            # forward is x @ W with W = weight.T (see init_stacked docstring).
            expected = ref.layers[l].weight.detach().T
            torch.testing.assert_close(layers[l][0], expected)

    def test_init_stacked_matches_arc_mlp_head(self):
        width, depth, n_classes, seed = 10, 3, 6, 5
        layers, heads = init_stacked(width, depth, n_classes, seeds=[seed])
        ref = ArcMLP(width, depth, init_seed=seed, head_classes=n_classes)
        torch.testing.assert_close(heads[0], ref.head.weight.detach().T)

    def test_batch_of_seeds_matches_per_seed_sequential_init(self):
        width, depth, n_classes = 8, 3, 4
        seeds = [11, 22, 33]
        layers, heads = init_stacked(width, depth, n_classes, seeds)
        for b, seed in enumerate(seeds):
            ref = ArcMLP(width, depth, init_seed=seed, head_classes=n_classes)
            for l in range(depth):
                torch.testing.assert_close(layers[l][b], ref.layers[l].weight.detach().T)
            torch.testing.assert_close(heads[b], ref.head.weight.detach().T)

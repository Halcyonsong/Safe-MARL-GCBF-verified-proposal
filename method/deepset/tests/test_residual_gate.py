"""Unit tests for residual-gate policy mechanics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from core.paths import ensure_import_path

ensure_import_path()

import jax
import jax.numpy as jnp
import numpy as np

from stage3.residual_policy import (
    GATE_FEAT_DIM,
    apply_residual,
    danger_features,
    init_train_params,
    make_modules,
    physics_gate,
)


class ResidualGateTest(unittest.TestCase):
    def setUp(self):
        self.modules = make_modules(base_hidden=8, gate_hidden=8, correction_hidden=8)
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        ego = jnp.zeros((1, 8))
        ent = jnp.zeros((1, 4, 7))
        self.base_params = self.modules.base.init(k1, ego, ent)
        self.train_params = init_train_params(
            self.modules, k2, ego_dim=8, entity_dim=7, entity_slots=4
        )

    def test_danger_feature_dim(self):
        ego = jnp.zeros((2, 8))
        ent = jnp.zeros((2, 4, 7))
        # one agent neighbor close
        ent = ent.at[0, 0, :].set(jnp.array([0.12, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
        feat = danger_features(ego, ent)
        self.assertEqual(feat.shape, (2, GATE_FEAT_DIM))
        self.assertLess(float(feat[0, 0]), 0.2)
        # Missing AA is capped far and has_aa=0 (not a huge sentinel).
        self.assertGreaterEqual(float(feat[1, 0]), 0.99)
        self.assertEqual(float(feat[1, 8]), 0.0)
        self.assertGreater(float(feat[0, 8]), 0.5)

    def test_physics_gate_near_vs_far(self):
        # Layout: d_aa, d_ao, close_aa, close_ao, v, g, n_aa, n_ao, has_aa, has_ao
        # near AA: d=0.10 -> n=1.0 -> gate ~1
        feat_near = jnp.array([[0.10, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        # far AA: d=0.40 -> n=4.0 -> gate 0 with open=2.5
        feat_far = jnp.array([[0.40, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        # absent both: has flags 0 -> gate 0
        feat_none = jnp.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        g_near = float(physics_gate(feat_near, open_mult=2.5, close_mult=1.0)[0])
        g_far = float(physics_gate(feat_far, open_mult=2.5, close_mult=1.0)[0])
        g_none = float(physics_gate(feat_none, open_mult=2.5, close_mult=1.0)[0])
        self.assertGreaterEqual(g_near, 0.99)
        self.assertLessEqual(g_far, 0.01)
        self.assertLessEqual(g_none, 0.01)

    def test_hard_gate_far_equals_base(self):
        ego = jnp.zeros((1, 8))
        # no nearby entities
        ent = jnp.zeros((1, 4, 7))
        a_base = self.modules.base.apply(self.base_params, ego, ent)
        a_res, g, delta = apply_residual(
            self.modules,
            self.base_params,
            self.train_params,
            ego,
            ent,
            gate_mode="hard",
            open_mult=2.0,
            close_mult=1.0,
        )
        self.assertLessEqual(float(g[0]), 1e-5)
        np.testing.assert_allclose(np.asarray(a_res), np.clip(np.asarray(a_base), -1, 1), atol=1e-5)

    def test_correction_can_move_action_when_gate_open(self):
        # Force a nearby entity so hard gate opens.
        ego = jnp.zeros((1, 8))
        ent = jnp.zeros((1, 4, 7))
        ent = ent.at[0, 0, :].set(jnp.array([0.11, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]))
        # zero correction should match base when delta~0; after init delta may be nonzero
        a0 = jnp.clip(self.modules.base.apply(self.base_params, ego, ent), -1.0, 1.0)
        a1, g, delta = apply_residual(
            self.modules,
            self.base_params,
            self.train_params,
            ego,
            ent,
            gate_mode="hard",
        )
        self.assertGreater(float(g[0]), 0.5)
        # residual formula holds pre-clip
        recon = jnp.clip(a0 + g[..., None] * delta, -1.0, 1.0)
        np.testing.assert_allclose(np.asarray(a1), np.asarray(recon), atol=1e-5)


if __name__ == "__main__":
    unittest.main()

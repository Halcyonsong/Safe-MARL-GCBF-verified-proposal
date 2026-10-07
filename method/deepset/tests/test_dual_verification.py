"""Regression checks for dual-threat proof semantics and verifier precedence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from core.paths import ensure_import_path

ensure_import_path()

import numpy as np

from stage2.bounds import Box, BucketSpec
from stage2.l2_query import l2_ibp, marabou_l2_square_crash
from stage2.marabou_query import QueryResult
from stage2.verify_l2 import verify_bucket


def point_spec(aa=0.15, ao=0.08):
    x = np.array([2, 2, 0, 0, 0.2, 0, 0, 0, aa, 0, 0, 0, ao, 0, 0, 0], dtype=float)
    return BucketSpec(name="dual_test", k=1, m=1, box=Box(x, x), property="l2_aa_ao")


class DualVerificationTest(unittest.TestCase):
    def test_ibp_requires_both_disks_avoided(self):
        with patch("stage2.l2_query.ibp_clipped_action_from_flat_box", return_value=(np.zeros(2), np.zeros(2))):
            self.assertEqual(l2_ibp(None, point_spec()).status, "proven")
            self.assertEqual(l2_ibp(None, point_spec(aa=0.09)).status, "unknown")
            self.assertEqual(l2_ibp(None, point_spec(ao=0.04)).status, "unknown")

    def test_marabou_checks_both_threats_with_outer_squares(self):
        class Net:
            def __init__(self):
                self.var = 18
                self.bounds = {}

            def getNewVariable(self):
                self.var += 1
                return self.var

            def addEquation(self, equation):
                pass

            def setLowerBound(self, var, value):
                self.bounds[(var, "lo")] = value

            def setUpperBound(self, var, value):
                self.bounds[(var, "hi")] = value

        nets = [Net(), Net()]
        with patch("stage2.l2_query.export_bucket_onnx"), patch(
            "stage2.l2_query._load_onnx_with_box",
            side_effect=[(n, list(range(16)), [16, 17]) for n in nets],
        ), patch("stage2.l2_query._solve", return_value=("unsat", {}, None)) as solve:
            result = marabou_l2_square_crash(None, point_spec(), Path("unused"))
        self.assertEqual(result.status, "proven_outer")
        self.assertEqual(solve.call_count, 2)
        self.assertEqual(nets[0].bounds[(19, "hi")], 0.1)
        self.assertEqual(nets[1].bounds[(19, "hi")], 0.05)
        self.assertEqual(len(result.subqueries), 2)

    def test_dual_timeout_is_not_a_proof(self):
        class Net:
            def getNewVariable(self):
                return 18

            def addEquation(self, equation):
                pass

            def setLowerBound(self, var, value):
                pass

            def setUpperBound(self, var, value):
                pass

        with patch("stage2.l2_query.export_bucket_onnx"), patch(
            "stage2.l2_query._load_onnx_with_box", return_value=(Net(), list(range(16)), [16, 17])
        ), patch("stage2.l2_query._solve", side_effect=[("unsat", {}, None), ("timeout", {}, None)]):
            result = marabou_l2_square_crash(None, point_spec(), Path("unused"))
        self.assertEqual(result.status, "timeout")
        self.assertEqual([r["status"] for r in result.subqueries], ["proven_outer", "timeout"])

    def test_sampling_violation_survives_formal_timeout(self):
        class Weights:
            hidden = 512

        with patch("stage2.encode.forward_bucket", return_value=np.zeros(2)), patch(
            "stage2.verify_l2.check_torch_matches_numpy"
        ), patch("stage2.verify_l2.l2_sampling_attack", return_value=QueryResult("dual_test", "sampling", "violated", "CE", 0)), patch(
            "stage2.verify_l2.l2_ibp", return_value=QueryResult("dual_test", "ibp", "unknown", "overlap", 0)
        ), patch("stage2.verify_l2.marabou_l2_square_crash", return_value=QueryResult("dual_test", "marabou", "timeout", "timeout", 0)) as marabou:
            result = verify_bucket(Weights(), point_spec(), onnx_dir=Path("unused"), timeout_s=1, n_samples=1, skip_marabou=False, continue_formal=True)
        self.assertEqual(result["verdict"], "violated")
        self.assertEqual(len(result["results"]), 3)
        marabou.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import numpy as np
import pytest

from support_classes import SupportClass
from provenance_schema import ProvenanceClass
from validation_runner import compute_support_class_metrics, run_ablation_matrix


def test_compute_support_class_metrics_reports_by_class_and_family():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pred = np.array([[1.0, 1.0], [4.0, 4.0]], dtype=np.float32)
    support = np.array([
        [int(SupportClass.AUTHORITATIVE_LOCKED), int(SupportClass.GUIDANCE_CONDITIONED_SDB)],
        [int(SupportClass.GUIDANCE_CONDITIONED_RIVER), int(SupportClass.LOW_CONFIDENCE_CONTINUOUS_FILL)],
    ], dtype=np.uint8)
    metrics = compute_support_class_metrics(pred=pred, truth=truth, support_class=support)
    assert metrics["overall"]["count"] == 4
    assert metrics["by_class"][str(int(SupportClass.GUIDANCE_CONDITIONED_SDB))]["label"] == "guidance_conditioned_sdb"
    assert "guidance_conditioned" in metrics["by_family"]


def test_run_ablation_matrix_rejects_shape_mismatch():
    truth = np.ones((2, 2), dtype=np.float32)
    support = np.full((2, 2), int(SupportClass.AUTHORITATIVE_LOCKED), dtype=np.uint8)
    prov = np.full((2, 2), int(ProvenanceClass.AUTHORITATIVE_LOCKED), dtype=np.uint8)
    with pytest.raises(ValueError):
        run_ablation_matrix(
            truth=truth,
            support_class=support,
            provenance_class=prov,
            cases={"bad": np.ones((3, 3), dtype=np.float32)},
        )

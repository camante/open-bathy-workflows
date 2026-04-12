from validation_runner import compute_support_class_metrics, compute_provenance_class_metrics
from support_classes import SupportClass
from provenance_schema import ProvenanceClass
import numpy as np


def test_support_metrics_group_by_canonical_family():
    pred = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    truth = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    support = np.array([[
        int(SupportClass.GUIDANCE_CONDITIONED_SDB),
        int(SupportClass.GUIDANCE_CONDITIONED_RIVER),
        int(SupportClass.SCAFFOLD_INFERRED),
    ]], dtype=np.uint8)
    metrics = compute_support_class_metrics(pred=pred, truth=truth, support_class=support)
    assert "guidance_conditioned" in metrics["by_family"]
    assert "scaffold_inferred" in metrics["by_family"]


def test_provenance_metrics_group_by_canonical_family():
    pred = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    truth = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    provenance = np.array([[
        int(ProvenanceClass.SDB_CONDITIONED_FILL),
        int(ProvenanceClass.RIVER_CONDITIONED_FILL),
        int(ProvenanceClass.RIVER_SCAFFOLD_DOMINANT_FILL),
    ]], dtype=np.uint8)
    metrics = compute_provenance_class_metrics(pred=pred, truth=truth, provenance_class=provenance)
    assert "guidance_conditioned" in metrics["by_family"]
    assert "scaffold_inferred" in metrics["by_family"]

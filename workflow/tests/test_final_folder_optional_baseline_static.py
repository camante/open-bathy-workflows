from pathlib import Path


def test_final_folder_validator_uses_receipt_required_outputs():
    root = Path(__file__).resolve().parents[1]
    text = (root / "active_pipeline.py").read_text(encoding="utf-8")
    assert "Validate final/ against the receipt-defined deliverables" in text
    assert 'receipt.get("required_outputs")' in text
    assert "baseline_skipped_reason" in text
    assert "final_output_folder_incomplete" in text

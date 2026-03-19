from pathlib import Path
import json

from xs_infer_bathy_raster import _dominant_xs_rejection_reason, _write_xs_rejection_receipts


def test_write_xs_rejection_receipts(tmp_path: Path):
    acct = {
        "n_xs_total": 10,
        "n_xs_rejected_missing_banks": 7,
        "n_xs_rejected_bad_width": 2,
        "n_xs_rejected_bad_wse": 1,
    }
    reason = _dominant_xs_rejection_reason(acct)
    receipts = _write_xs_rejection_receipts(tmp_path / "xs_rejection_receipt", acct, reason=reason)
    data = json.loads(receipts["json"].read_text(encoding="utf-8"))
    assert data["dominant_rejection_reason"] == "missing_bank_picks"
    assert receipts["csv"].exists()

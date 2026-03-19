import pytest

from precedence_audit import validate_precedence_audit


def test_validate_precedence_audit_rejects_changed_locked_cells():
    with pytest.raises(ValueError, match="Changed locked authoritative cells"):
        validate_precedence_audit({
            "authoritative_lock": {
                "lock_preserved": False,
                "changed_locked_cell_count": 1,
                "guidance_nonzero_on_locked_count": 0,
            },
            "gap_fill": {"remaining_gap_nodata_count": 0},
        })

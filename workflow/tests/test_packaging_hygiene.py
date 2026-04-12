from pathlib import Path

import pytest

from packaging_hygiene import clean_repo_hygiene, require_clean_repo_tree


def test_packaging_hygiene_detects_and_cleans_cache_artifacts(tmp_path):
    bad_dir = tmp_path / '__pycache__'
    bad_dir.mkdir()
    bad_file = tmp_path / 'x.pyc'
    bad_file.write_bytes(b'0')

    with pytest.raises(RuntimeError):
        require_clean_repo_tree(tmp_path)

    report = clean_repo_hygiene(tmp_path)
    assert report.has_issues
    require_clean_repo_tree(tmp_path)

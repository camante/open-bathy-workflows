from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from packaging_hygiene import clean_repo_hygiene, require_clean_repo_tree


def main() -> int:
    p = argparse.ArgumentParser(description="Create a clean workflow zip without cache artifacts.")
    p.add_argument('--source', default='.', help='Workflow repo directory to package.')
    p.add_argument('--output', default='workflow_clean.zip', help='Output zip path.')
    p.add_argument('--clean-first', action='store_true', help='Remove cache artifacts before packaging.')
    args = p.parse_args()

    source = Path(args.source).resolve()
    if args.clean_first:
        clean_repo_hygiene(source)
    require_clean_repo_tree(source)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_base = output.with_suffix('')
    made = shutil.make_archive(str(archive_base), 'zip', root_dir=str(source.parent), base_dir=source.name)
    made_path = Path(made)
    if made_path != output:
        if output.exists():
            output.unlink()
        made_path.replace(output)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
items = ['python', 'venv', 'mlx-runtime', 'models']
for name in items:
    src = source / name
    if not src.exists():
        raise SystemExit(f"Missing dependency: {name}")
    dst = target / name
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.cache', '*.lock'))
    else:
        shutil.copy2(src, dst)
frameworks = source.parent / 'Frameworks'
if frameworks.is_dir():
    shutil.copytree(frameworks, target.parent / 'Frameworks', symlinks=True)
licenses = source / 'Licenses'
if licenses.is_dir():
    shutil.copytree(licenses, target / 'Licenses', dirs_exist_ok=True)
repo = Path(__file__).resolve().parents[1]
for package in ('mineru', 'mineru_vl_utils'):
    shutil.copytree(repo / 'vendor' / package, target / 'venv/lib/python3.11/site-packages' / package, dirs_exist_ok=True)
shutil.copytree(repo / 'vendor/mlx_vlm', target / 'mlx-runtime/mlx_vlm', dirs_exist_ok=True)

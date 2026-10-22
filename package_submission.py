"""Create a source archive without caches, Git internals or nested archives."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(__file__).resolve().parent
with ZipFile(root/'tredence-aqua-submission.zip','w',ZIP_DEFLATED) as archive:
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if path.is_file() and not any(p in {'.git','.pytest_cache','__pycache__','.venv'} for p in relative.parts) and path.suffix not in {'.zip','.pyc'}:
            archive.write(path,relative)
print('Submission archive rebuilt')

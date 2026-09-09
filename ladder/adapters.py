"""Explicit adapters for reviewed, revision-pinned custom model implementations."""
from pathlib import Path
from .corpus import sha_file

KOLIBER_REPO = 'OrisTeam/Koliber-v1.1-Base-Preview'
KOLIBER_REVISION = '10bfff5fec23a43cd798b80645f95673a234f39a'
KOLIBER_FILES = {
    'modeling_koliber.py': 'd44f08f2fffe7025145341cfe8543be749b8899d358e97217c5bcf82f50dfcba',
    'configuration_koliber.py': 'e8c85bf39fdb409d139c3cf377e651edb87b348a2a5cf010a22dfb0cfa82a0bc',
    '__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
    'config.json': '8806a4fa76061ba4790396cacad8ed371a41484f1abf3ff527ff8a3dcb89e033',
}


def koliber_snapshot(model_path, revision):
    if model_path != KOLIBER_REPO or revision != KOLIBER_REVISION:
        raise ValueError('Koliber adapter requires the reviewed repository and exact pinned revision')
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(model_path, revision=revision,
                allow_patterns=[*KOLIBER_FILES, 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']))
    for name, expected in KOLIBER_FILES.items():
        if sha_file(root / name) != expected:
            raise ValueError(f'Reviewed Koliber source hash mismatch: {name}')
    return str(root)

"""Contained, atomic, versioned artifacts for the new shared-data study."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    path.relative_to(root)
    return path


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_contract(code):
    code = Path(code)
    execution = read(code / 'config/ectt_execution_v1.json')
    path = code / execution['scientific_contract']
    if digest(path) != execution['scientific_contract_sha256']:
        raise ValueError('Frozen scientific contract changed; execution amendment required')
    scientific = read(path)
    if scientific['seeds'] != [11, 22, 33] or len(scientific['datasets']) != 8:
        raise ValueError('Unexpected target/seed registry')
    return scientific, execution


def identity(code):
    code = Path(code)
    paths = ['config/ectt_protocol_v1.json', 'config/ectt_execution_v1.json',
             'config/sources.lock.json', 'config/ectt_gpu_requirements.txt', 'research/MASTER_PROTOCOL_RECORD.json']
    paths += [str(p.relative_to(code)) for p in sorted((code / 'scripts').glob('ectt_*.py'))]
    return hashlib.sha256(json.dumps({p: digest(code / p) for p in paths}, sort_keys=True).encode()).hexdigest()


def artifact_manifest(store):
    return {str(p.relative_to(store)): digest(p) for p in sorted(Path(store).rglob('*'))
            if p.is_file() and p.name not in ('COMPLETE.json', '.lock') and not p.name.endswith('.tmp')}


def completed(store, expected_identity):
    store = Path(store)
    marker = store / 'COMPLETE.json'
    if not marker.exists():
        return False
    record = read(marker)
    if record['identity'] != expected_identity:
        raise ValueError('Completion identity mismatch; previous results preserved')
    for name, sha in record['artifacts'].items():
        path = inside(store, store / name)
        if not path.is_file() or digest(path) != sha:
            raise ValueError('Missing/corrupt completed artifact: ' + str(path))
    return True


def finish(store, expected_identity):
    write(Path(store) / 'COMPLETE.json', {'identity': expected_identity,
          'state': 'COMPLETED', 'artifacts': artifact_manifest(store)})

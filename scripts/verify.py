import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
    failures = []
    missing_models = []
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file():
            (missing_models if item['path'].startswith('checkpoints/') else failures).append(item['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(item['path'])
    if failures or missing_models:
        print('Mismatched/missing payload:', failures)
        print('Missing checkpoints:', len(missing_models), '- extract evaluated-checkpoints-v1.zip at repository root')
        raise SystemExit(1)
    print(f'PASS: {len(manifest["files"])} source, configuration, data, result and checkpoint files verified.')


if __name__ == '__main__':
    main()

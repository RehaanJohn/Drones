#!/usr/bin/env python3
"""Download existing scanner weights once, outside mission startup."""
import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('/tmp/addc-qr-models'))
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]/'QR_Scanning'/'scanner.py'
    spec = importlib.util.spec_from_file_location('existing_qr_scanner', source)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    scanner.MODEL_DIR = args.directory.expanduser()
    scanner.ensure_models_exist()
    print(f'QR weights ready in {scanner.MODEL_DIR}')


if __name__ == '__main__':
    main()

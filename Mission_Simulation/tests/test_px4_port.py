import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('px4_port', Path(__file__).resolve().parents[1]/'tools/patch_px4_fortress.py')
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)


class PortGuardTests(unittest.TestCase):
    def test_backups_idempotence_and_local_edit_protection(self):
        original = b'#include <gz/transport.hh>\ngz::transport::Node node;\n'
        hashes = {'GZBridge.hpp': hashlib.sha256(original).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch.dict(port.HASHES, hashes, clear=True):
            root = Path(directory)
            file = root/'src/modules/simulation/gz_bridge/GZBridge.hpp'
            file.parent.mkdir(parents=True)
            file.write_bytes(original)
            port.patch(root, check=True)
            self.assertEqual(file.read_bytes(), original)
            port.patch(root)
            transformed = file.read_bytes()
            self.assertIn(b'ignition::transport', transformed)
            self.assertEqual(file.with_name(file.name+'.addc-original').read_bytes(), original)
            port.patch(root)
            self.assertEqual(file.read_bytes(), transformed)
            file.write_bytes(transformed+b'// Local changes\n')
            with self.assertRaises(ValueError):
                port.patch(root)

    def test_unknown_source_is_rejected_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root/'src/modules/simulation/gz_bridge/CMakeLists.txt'
            file.parent.mkdir(parents=True)
            file.write_text('unsupported version\n')
            with self.assertRaises(ValueError):
                port.patch(root)
            self.assertEqual(file.read_text(), 'unsupported version\n')
            self.assertFalse(file.with_name(file.name+'.addc-original').exists())

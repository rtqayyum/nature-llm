"""Synthetic workbook/RAR/ZIP tests; no released data or external extraction."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import source_inspect as audit
import source_audit_env as environment


def synthetic_rar(name="../../not_extracted.npy"):
    def header(kind, flags, data):
        body = struct.pack("<BHH", kind, flags, 7 + len(data)) + data
        return struct.pack("<H", zlib.crc32(body) & 0xffff) + body
    encoded, payload = name.encode(), b"synthetic payload"
    fields = struct.pack("<LLBLLBBHL", len(payload), len(payload), 3, zlib.crc32(payload) & 0xffffffff,
                         0x50210000, 20, 0x30, len(encoded), 0x20)
    return (b"Rar!\x1a\x07\x00" + header(0x73, 0, b"\0" * 6) + header(0x74, 0x8000, fields + encoded)
            + payload + header(0x7b, 0, b""))


class SourceCoreTests(unittest.TestCase):
    def test_copy_resume_cache_and_corruption_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"source content" * 10000
            archive, destination = root / "source.zip", root / "cache/container.rar"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
                stream.writestr("../../container.rar", payload)
            destination.parent.mkdir()
            partial = destination.with_suffix(".rar.part")
            partial.write_bytes(payload[:1103])
            with zipfile.ZipFile(archive) as stream:
                metadata = audit.copy_member(stream, "../../container.rar", destination, "outer", lambda **kw: None)
                self.assertEqual(metadata["resumed_prefix_bytes"], 1103)
                self.assertEqual(destination.read_bytes(), payload)
                self.assertTrue(audit.copy_member(stream, "../../container.rar", destination, "outer", lambda **kw: None)["reused_existing"])
                destination.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "preserved"):
                    audit.copy_member(stream, "../../container.rar", destination, "outer", lambda **kw: None)

    def test_complete_copy_without_manifest_is_recovered_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "cache.rar"
            destination.write_bytes(b"complete")
            with zipfile.ZipFile(root / "outer.zip", "w") as stream:
                stream.writestr("cache.rar", b"complete")
            with zipfile.ZipFile(root / "outer.zip") as stream:
                result = audit.copy_member(stream, "cache.rar", destination, "outer", lambda **kw: None)
            self.assertEqual(result["resumed_prefix_bytes"], 8)
            self.assertEqual(destination.read_bytes(), b"complete")

    def test_inspection_environment_installs_into_its_own_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, report = root / "code", root / "report"
            (code / "config").mkdir(parents=True)
            (code / "config/source_audit_requirements.txt").write_text("rarfile==4.2\n")
            def run(args, **kwargs):
                if kwargs.get("stdout"):
                    kwargs["stdout"].write("synthetic freeze\n")
                return subprocess.CompletedProcess(args, 1 if kwargs.get("capture_output") else 0)
            with mock.patch.object(environment.subprocess, "run", side_effect=run) as execute:
                python = environment.ensure(root, code, report)
            self.assertEqual(python, root / ".venv-source-audit/bin/python")
            install = next(call.args[0] for call in execute.call_args_list if "install" in call.args[0])
            self.assertIn("--no-user", install)
            self.assertEqual(str(install[0]), str(python))
            self.assertFalse((root / ".venv-reference-cpu").exists())


@unittest.skipUnless(importlib.util.find_spec("rarfile") and importlib.util.find_spec("openpyxl"), "Needs source-inspection test dependencies")
class SourceParserTests(unittest.TestCase):
    def create_workbook(self, path):
        import openpyxl
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Table 1"
        sheet.append(["Dataset", "Model", "MAE"])
        sheet.append(["Shanghai", "LLM-UTP", 26.79])
        sheet.append([None, "Beijing", 23.48])
        sheet["D4"] = "=SUM(C2:C3)"
        workbook.save(path)

    def test_workbook_preserves_formulas_missing_caches_and_blank_leading_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.create_workbook(root / "source.xlsx")
            result = audit.inspect_workbook(root / "source.xlsx", root / "report")
            self.assertEqual(result["sheets"][0]["formulas_without_cached_values"], 1)
            rows = [json.loads(row) for row in (root / "report/sheet-000.jsonl").read_text().splitlines()]
            self.assertEqual(rows[2]["cells"][0]["coordinate"], "B3")
            self.assertEqual(rows[3]["cells"][0]["source_value"], "=SUM(C2:C3)")

    def test_rar_headers_are_parsed_without_extracting_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.rar").write_bytes(synthetic_rar())
            result = audit.inspect_rar(root / "source.rar", root / "inventory.json")
            self.assertEqual(result["member_count"], 1)
            self.assertEqual(result["members"][0]["name"], "../../not_extracted.npy")
            self.assertFalse((root.parent / "not_extracted.npy").exists())

    def test_complete_inspection_reuses_containers_and_reports_required_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "code"
            (code / "config").mkdir(parents=True)
            original = root / "work/reference/snapshots/synthetic/Source Data/Source Data.zip"
            original.parent.mkdir(parents=True)
            workbook = root / "workbook.xlsx"
            self.create_workbook(workbook)
            with zipfile.ZipFile(original, "w") as stream:
                stream.write(workbook, "Source Data/Zhang_Source_Data.xlsx")
                stream.writestr("Source Data/Supplementary Fig4/sh_outputs.rar", synthetic_rar("predictions.npy"))
                stream.writestr("Source Data/Supplementary Fig4/bus_outputs.rar", b"not a RAR")
            asset = {"path": "Source Data/Source Data.zip", "size": original.stat().st_size, "sha256": audit.file_sha(original)}
            (code / "config/sources.lock.json").write_text(json.dumps({"reference_commit": "synthetic", "lfs_files": [asset]}))
            for number in (1, 2):
                report = root / "reports" / str(number)
                self.assertEqual(audit.run(root, code, report), 2)
                result = json.loads((report / "source_inspection.json").read_text())
                self.assertEqual(result["items"]["workbook"]["state"], "COMPLETED")
                self.assertEqual(result["items"]["shanghai_archive"]["state"], "COMPLETED")
                self.assertEqual(result["items"]["bus_archive"]["state"], "FAILED")
                if number == 2:
                    self.assertTrue(result["items"]["workbook"]["container_metadata"]["reused_existing"])
            # A separate valid synthetic release exercises the all-required-items pass.
            with zipfile.ZipFile(original, "w") as stream:
                stream.write(workbook, "Source Data/Zhang_Source_Data.xlsx")
                stream.writestr("Source Data/Supplementary Fig4/sh_outputs.rar", synthetic_rar("predictions.npy"))
                stream.writestr("Source Data/Supplementary Fig4/bus_outputs.rar", synthetic_rar("bus.npy"))
            asset.update(size=original.stat().st_size, sha256=audit.file_sha(original))
            (code / "config/sources.lock.json").write_text(json.dumps({"reference_commit": "synthetic", "lfs_files": [asset]}))
            self.assertEqual(audit.run(root, code, root / "reports/3"), 3)
            final = json.loads((root / "reports/3/source_inspection.json").read_text())
            self.assertTrue(all(row["state"] == "COMPLETED" for row in final["items"].values()))


if __name__ == "__main__":
    unittest.main()

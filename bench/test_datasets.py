import io
import os
import tarfile
import tempfile
import unittest

from datasets import _safe_extract


class SafeExtractTests(unittest.TestCase):
    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as root:
            archive_path = os.path.join(root, "bad.tar")
            with tarfile.open(archive_path, "w") as archive:
                member = tarfile.TarInfo("../escaped")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))

            destination = os.path.join(root, "data")
            os.mkdir(destination)
            with tarfile.open(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                    _safe_extract(archive, destination)
            self.assertFalse(os.path.exists(os.path.join(root, "escaped")))


if __name__ == "__main__":
    unittest.main()

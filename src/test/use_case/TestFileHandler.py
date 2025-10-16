import unittest
import os
import tempfile
import shutil

from src.main.application.use_case.FileHandler import FileHandler

class TestFileHandler(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_write_and_read_file(self):
        content_to_write = "This is a test."
        file_path = FileHandler.write_file(content_to_write, self.test_dir, "", ".txt")

        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))

        read_content = FileHandler.read_file(file_path)
        self.assertEqual(read_content, content_to_write)

    def test_find_files_by_name(self):
        subdir = os.path.join(self.test_dir, "subdir")
        os.makedirs(subdir)

        # Create test files
        with open(os.path.join(self.test_dir, "report_2023.txt"), 'w') as f: f.write('a')
        with open(os.path.join(self.test_dir, "image.jpg"), 'w') as f: f.write('b')
        with open(os.path.join(subdir, "final_report.doc"), 'w') as f: f.write('c')

        found_files = FileHandler.find_files_by_name(self.test_dir, "report")

        self.assertEqual(len(found_files), 2)
        self.assertIn(os.path.join(self.test_dir, "report_2023.txt"), found_files)
        self.assertIn(os.path.join(subdir, "final_report.doc"), found_files)

if __name__ == '__main__':
    unittest.main()

import os
from datetime import datetime
from typing import Optional


class FileHandler:
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    @staticmethod
    def write_file(
            content: str,
            directory_path: str,
            file_name: Optional[str] = None,
            extension: str = ".txt",
            timestamp_format: str = "%Y%m%d_%H%M%S"
    ) -> Optional[str]:
        try:
            os.makedirs(directory_path, exist_ok=True)

            final_file_name: str

            if file_name is None or file_name == "":
                timestamp = datetime.now().strftime(timestamp_format)
                final_file_name = f"{timestamp}{extension}"
            else:
                if not file_name.endswith(extension):
                    final_file_name = f"{file_name}{extension}"
                else:
                    final_file_name = file_name

            full_path = os.path.join(directory_path, final_file_name)

            with open(full_path, 'w') as f:
                f.write(content)

            print(f"File written successfully to {full_path}")
            return full_path
        except Exception as e:
            print(f"Error writing file: {e}")
            return None

    @staticmethod
    def read_file(file_path: str) -> str | None:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    @staticmethod
    def find_files_by_name(directory_path: str, file_name_part: str) -> list[str]:
        matching_files = []
        try:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file_name_part in file:
                        matching_files.append(os.path.join(root, file))
            return matching_files
        except Exception as e:
            print(f"Error finding files: {e}")
            return []

import os
from datetime import datetime

class FileHandler:
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    @staticmethod
    def write_file(directory_path: str, content: str, extension: str = '.txt') -> str | None:
        try:
            os.makedirs(directory_path, exist_ok=True)
            timestamp = datetime.now().strftime(FileHandler.TIMESTAMP_FORMAT)
            filename = f"{timestamp}{extension}"
            full_path = os.path.join(directory_path, filename)

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

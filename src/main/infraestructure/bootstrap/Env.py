import dataclasses
import os
from typing import Optional


@dataclasses.dataclass
class Env:
    app_host: Optional[str] = None
    app_version: Optional[str] = None
    neo4j_host: Optional[str] = None
    neo4j_port: Optional[int] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_db: Optional[str] = None

    def __post_init__(self):
        self.load_from_env()

    def load_from_file(self, file_path: str):
        if not os.path.exists(file_path):
            print(f"Warning: Env file not found at {file_path}")
            return

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip().upper()
                value = value.strip()

                if key == "APP_HOST":
                    self.app_host = value
                elif key == "APP_VERSION":
                    self.app_version = value
                elif key == "NEO4J_HOST":
                    self.neo4j_host = value
                elif key == "NEO4J_PORT" and value.isdigit():
                    self.neo4j_port = int(value)
                elif key == "NEO4J_USER":
                    self.neo4j_user = value
                elif key == "NEO4J_PASSWORD":
                    self.neo4j_password = value
                elif key == "NEO4J_DB":
                    self.neo4j_db = value

    def load_from_env(self):
        self.app_host = os.getenv("APP_HOST", "0.0.0.0")
        self.app_version = os.getenv("APP_VERSION", "0.1.0")
        self.neo4j_host = os.getenv("NEO4J_HOST", "localhost")
        port_str = os.getenv("NEO4J_PORT")
        self.neo4j_port = int(port_str) if port_str and port_str.isdigit() else 7687
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_db = os.getenv("NEO4J_DB", "neo4j")

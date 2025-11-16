import dataclasses

from src.main.infraestructure.bootstrap.Constant import Constant
from src.main.infraestructure.bootstrap.Env import Env


@dataclasses.dataclass
class Bootstrap:
    env: Env
    constant: Constant

    def __init__(self):
        self.env = Env()
        self.constant = Constant()

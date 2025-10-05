import dataclasses


@dataclasses.dataclass
class Constant:
    DATE: str = "%Y-%m-%d"
    TIME: str = "T%H:%M:%SZ"
    DATE_TIME: str = "%Y-%m-%dT%H:%M:%SZ"

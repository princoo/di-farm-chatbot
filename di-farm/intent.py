import typing
from dataclasses import dataclass

@dataclass
class IntentModel:
    intent: str
    text: typing.List[str]
    responses: typing.List[str]
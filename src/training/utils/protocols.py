from typing import Any, List, Protocol


class ScalerProtocol(Protocol):
    def scale(self, outputs: Any) -> Any:
        ...

    def unscale_(self, optimizer: Any) -> None:
        ...

    def step(self, optimizer: Any) -> None:
        ...

    def update(self) -> None:
        ...


class SchedulerProtocol(Protocol):
    def step(self) -> None:
        ...

    def get_last_lr(self) -> List[float]:
        ...

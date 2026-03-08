"""Bridge-specific error types."""


class BridgeError(RuntimeError):
    """Base class for RL <-> sim bridge failures."""


class BridgeContractError(BridgeError):
    """Raised when the RL contract and live sim contract diverge."""


class SimArtifactNotFoundError(BridgeError):
    """Raised when the expected packaged sim artifact is missing."""


class SimPrerequisiteError(BridgeError):
    """Raised when the checked-out sim workspace is not runtime-ready."""


class BridgeJVMStateError(BridgeError):
    """Raised when the embedded JVM is in an unusable state."""


class BatchSlotExecutionError(BridgeError):
    """Raised when one slot in a bridge batch fails."""

    def __init__(self, slot_index: int, operation: str, message: str) -> None:
        super().__init__(f"Batch slot {slot_index} failed during {operation}: {message}")
        self.slot_index = int(slot_index)
        self.operation = str(operation)

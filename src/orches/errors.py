"""Errors raised for invalid ORCHES inputs."""


class OrchesInputError(ValueError):
    """Base class for invalid user-controlled ORCHES inputs."""


class ConfigurationError(OrchesInputError):
    """Raised when a configuration cannot be parsed or is inconsistent."""


class WorkloadTraceError(OrchesInputError):
    """Raised when a TTC workload trace is malformed or inconsistent."""


class SimulationError(RuntimeError):
    """Raised when a simulator backend cannot execute or report valid output."""

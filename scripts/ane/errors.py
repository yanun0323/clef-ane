"""Request errors shared by the service and HTTP adapters."""


class RequestError(ValueError):
    """The request is invalid (HTTP 400)."""


class TooLong(RequestError):
    """The schema alone does not fit the largest context (HTTP 413)."""

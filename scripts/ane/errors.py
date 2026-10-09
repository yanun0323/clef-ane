"""Request errors shared by the service and HTTP adapters."""


class RequestError(ValueError):
    """The request is invalid (HTTP 400)."""


class TooLong(RequestError):
    """The request exceeds an input size, pixel, or context limit (HTTP 413)."""

from contextvars import ContextVar

user_context: ContextVar[dict] = ContextVar("user_context", default={})

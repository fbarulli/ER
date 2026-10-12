"""One terminal-and-transcript entry point for Colab progress frames."""
from __future__ import annotations

from collections.abc import Callable

from cli.log_capture import progress_frames_to_lines


class ColabProgressOutput:
    """Forward remote output to the terminal and the shared lane transcript."""

    def __init__(self, write_log: Callable[[str], None],
                 write_terminal: Callable[[str], None] | None = None) -> None:
        self._write_log = write_log
        self._write_terminal = write_terminal

    @classmethod
    def build(cls, write_log: Callable[[str], None],
              write_terminal: Callable[[str], None] | None = None) -> ColabProgressOutput:
        return cls(write_log, write_terminal)

    def forward(self, text: str, *, prefix: str = "", capture: bool = True) -> None:
        """Preserve carriage-return bars on-screen and as readable log lines."""
        if not text:
            return
        if capture:
            logged = progress_frames_to_lines(text)
            if prefix:
                logged = "".join(prefix + line for line in logged.splitlines(keepends=True))
            self._write_log(logged)
        start = 0
        while start < len(text):
            endings = [
                position for position in (text.find("\r", start), text.find("\n", start))
                if position >= 0
            ]
            end = min(endings) + 1 if endings else len(text)
            rendered = f"{prefix}{text[start:end]}"
            if self._write_terminal is None:
                print(rendered, end="", flush=True)
            else:
                self._write_terminal(rendered)
            start = end

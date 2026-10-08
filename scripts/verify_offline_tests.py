"""Run pytest with pipe-based asyncio wakeups when sandbox socket IPC is denied.

This changes only event-loop wakeup transport in this verification process.
Network permissions and application code are unchanged; Python 3.12/Linux only.
"""
import asyncio
import os
import sys

import pytest


class _PipeEnd:
    def __init__(self, fd): self.fd = fd
    def fileno(self): return self.fd
    def close(self): os.close(self.fd)
    def recv(self, size): return os.read(self.fd, size)
    def send(self, data): return os.write(self.fd, data)


class PipeWakeupLoop(asyncio.SelectorEventLoop):
    def _make_self_pipe(self):
        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False); os.set_blocking(write_fd, False)
        self._ssock, self._csock = _PipeEnd(read_fd), _PipeEnd(write_fd)
        self._internal_fds += 1
        self._add_reader(read_fd, self._read_from_self)


class PipeWakeupPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self): return PipeWakeupLoop()


def main(args=None):
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Pipe wakeup verification requires Linux/Python 3.12')
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(PipeWakeupPolicy())
    try: return pytest.main(args)
    finally: asyncio.set_event_loop_policy(previous)


if __name__ == '__main__': sys.exit(main(sys.argv[1:]))

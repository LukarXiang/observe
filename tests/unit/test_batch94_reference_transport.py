"""Verify both patched low-level entry points without allocating a real socket."""
import socket

import pytest

from scripts import recover_strategy_batch94_reference as recovery


def test_socket_entrypoints_blocked_without_kernel_socket():
    # An arbitrary receiver proves the blocker runs before the original descriptor
    # could check the receiver or invoke an operating-system transport.
    receiver = object()
    with recovery.network_blocked():
        for method in (socket.socket.connect, socket.socket.connect_ex):
            with pytest.raises(AssertionError, match='network forbidden'):
                method(receiver, ('127.0.0.1', 1))

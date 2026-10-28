"""Network attempts fail before transport and the patch does not persist."""
import socket

import pytest
import requests

from scripts import recover_strategy_batch94_reference as recovery


@pytest.mark.parametrize('attempt', [
    lambda: socket.getaddrinfo('example.invalid', 443),
    lambda: socket.create_connection(('127.0.0.1', 1)),
    lambda: requests.get('https://example.invalid'),
])
def test_numerical_worker_network_blocked_before_transport(attempt):
    with recovery.network_blocked(), pytest.raises(AssertionError, match='network forbidden'): attempt()


def test_socket_methods_blocked_before_transport():
    with socket.socket() as sock, recovery.network_blocked():
        with pytest.raises(AssertionError, match='network forbidden'): sock.connect(('127.0.0.1', 1))
        with pytest.raises(AssertionError, match='network forbidden'): sock.connect_ex(('127.0.0.1', 1))


def test_network_context_restores_original_functions_after_exception():
    original = socket.getaddrinfo
    with pytest.raises(RuntimeError), recovery.network_blocked():
        assert socket.getaddrinfo is not original
        raise RuntimeError('inside context')
    assert socket.getaddrinfo is original

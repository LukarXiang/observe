import asyncio
import os
import sys

from anyio.from_thread import start_blocking_portal
import pytest

from scripts.verify_offline_tests import PipeWakeupPolicy


@pytest.mark.skipif(sys.platform != 'linux' or sys.version_info[:2] != (3, 12), reason='Linux/Python3.12 verification adapter')
def test_pipe_loop_wakes_cross_thread_without_socket_transport():
    previous = asyncio.get_event_loop_policy(); policy = PipeWakeupPolicy()
    asyncio.set_event_loop_policy(policy)
    try:
        with start_blocking_portal() as portal:
            assert portal.call(lambda: 42) == 42
            async def check():
                loop = asyncio.get_running_loop()
                assert not os.get_blocking(loop._ssock.fileno())
                assert not os.get_blocking(loop._csock.fileno())
                await asyncio.sleep(.001)
                return 'awake'
            assert portal.call(check) == 'awake'
    finally: asyncio.set_event_loop_policy(previous)
    assert asyncio.get_event_loop_policy() is previous

"""Sandbox guard for HumanEval program execution.

Loaded via PYTHONPATH (a directory containing this file, importable as `sitecustomize`)
into the child interpreter that runs a model-generated program.  It makes outbound
network access fail loudly instead of silently reaching the tailnet/Internet.  This host
denies unprivileged user namespaces (`unshare -n -r` -> EPERM), so a kernel network
namespace is not available; this in-process guard plus the rlimits set by the parent are
the isolation we have.  It is deliberately narrow: sockets only, nothing else is patched.
"""
import socket as _socket

_MSG = "network disabled by the HumanEval sandbox"


def _deny(*_a, **_k):
    raise OSError(_MSG)


_socket.socket = _deny
_socket.create_connection = _deny
_socket.getaddrinfo = _deny
_socket.socketpair = _deny

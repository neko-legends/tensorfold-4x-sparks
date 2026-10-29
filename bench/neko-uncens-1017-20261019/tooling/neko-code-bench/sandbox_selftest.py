#!/usr/bin/env python3
"""Prove the HumanEval execution sandbox behaves: network blocked, CPU runaway killed,
big writes/forks contained, syntax/assert classified."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("hb", "humaneval_bench.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)

tests = {
    "net-urlopen": "import urllib.request\nprint(urllib.request.urlopen('http://example.com', timeout=3).status)\n",
    "net-socket": "import socket\ns = socket.socket()\nprint('made', s)\n",
    "net-dns": "import socket\nprint(socket.getaddrinfo('forge', 8000))\n",
    "timeout-loop": "while True: pass\n",
    "syntax": "def f(:\n  pass\n",
    "assert": "assert 1 == 2\n",
    "bigfile": "open('x', 'wb').write(b'A' * (1024 * 1024 * 64))\n",
    "forkbomb": "import os\nn = 0\nwhile n < 500:\n    try:\n        os.fork()\n        n += 1\n    except OSError:\n        break\nprint('forks', n)\n",
    "ok": "print('fine')\n",
}
for k, v in tests.items():
    r = hb.run_program(v, timeout=20)
    err = r["stderr_tail"][-160:].replace("\n", " | ")
    print("%-12s -> %-24s rc=%s wall=%s  err=%s" % (k, r["category"], r["returncode"], r["wall_s"], err))

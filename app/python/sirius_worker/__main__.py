"""Command line of the SIRIUS compute worker.

    python -m sirius_worker [--host H] [--port P] [--token T] [--device auto|cuda|cpu] [--log-level L]

Listens on host:port (port 0 picks a free one), prints one JSON line
``{"port": N, "pid": ..., "host": ..., "device": ...}`` to stdout once it is
ready -- the launching application reads exactly that -- and logs to stderr.
The token, when given, must be sent with the client's ``hello``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sirius_worker", description="SIRIUS compute worker")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (0.0.0.0 for a cluster node)")
    parser.add_argument("--port", type=int, default=0, help="TCP port; 0 picks a free port")
    parser.add_argument("--token", default=os.environ.get("SIRIUS_TOKEN", ""),
                        help="shared secret the client must present (default: $SIRIUS_TOKEN)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="where models run; auto = cuda when torch sees a GPU")
    parser.add_argument("--log-level", default="INFO", help="stderr log level")
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from .server import WorkerServer, announce
    from .steps import workbench

    try:
        wb = workbench()
        logging.getLogger("sirius_worker").info("step library: %s", getattr(wb, "__source_file__", wb.__file__))
    except ImportError as e:
        logging.getLogger("sirius_worker").error("%s", e)
        return 2

    server = WorkerServer(args.host, args.port, args.token, args.device)
    server.bind()
    announce(server)

    def on_signal(signum, frame):  # noqa: ARG001
        server.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError):
            pass
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

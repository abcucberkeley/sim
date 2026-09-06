"""The compute worker: one TCP listener, one client at a time, one job at a
time, streamed progress, cancellation.

Requests (see protocol.py for the framing):

    hello       {token}                     -> capabilities
    ping        {}                          -> {}
    model_info  {path}                      -> format, input_shape, output_shape, dtype, size_bytes, channels_out
    run         {kind, params} + tensors    -> "progress"* then "result" (+ tensors)
    cancel      {id}                        -> {} (the cancelled run replies with an error "cancelled")
    shutdown    {}                          -> {} and the server exits

Run kinds and their tensors:

    torch_segment  in  "input" (z, y, x) float32
                   out "prob"  (C, z, y, x) float32, result {channels, seconds}
    sim            in  "input" (sections, y, x) or (c, t, sections, y, x)
                   out "output" reconstructed, same rank; result {fits, seconds}
    <numpy kinds>  in  "input" (c, t, z, y, x) [+ "labels" (t, z, y, x) uint32]
                   out "output" (+ "labels", "prob"), result {meta, info, seconds}

The reader loop runs on the connection's thread and the job on a worker
thread, so a cancel request is read while a run is in progress. Every reply
carries the request's id.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from . import __version__
from .protocol import ProtocolError, encode_frame, read_frame
from .steps import workbench

log = logging.getLogger("sirius_worker")

# kinds served through run_step plus the two with their own tensor contracts
_SPECIAL_KINDS = ("torch_segment", "sim")


class _Cancelled(Exception):
    pass


class WorkerServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, token: str = "", device: str = "auto",
                 max_clients: int = 1) -> None:
        self.host = host
        self.port = port
        self.token = token or ""
        self.device = device
        self.max_clients = max_clients
        self._listener: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._job_lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None

    # --- lifecycle ------------------------------------------------------------

    def bind(self) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(4)
        s.settimeout(0.5)
        self._listener = s
        self.port = s.getsockname()[1]
        return self.port

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        assert self._listener is not None
        log.info("listening on %s:%d (device %s)", self.host, self.port, self.resolved_device())
        try:
            while not self._stop.is_set():
                try:
                    conn, addr = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                log.info("client %s:%d connected", *addr[:2])
                try:
                    self._serve_client(conn)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    log.info("client %s:%d disconnected", *addr[:2])
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

    # --- capabilities ----------------------------------------------------------

    def resolved_device(self) -> str:
        return workbench().resolve_device(self.device)

    def capabilities(self) -> Dict[str, Any]:
        wb = workbench()
        methods = ["hello", "ping", "model_info", "run", "cancel", "shutdown"]
        kinds = list(_SPECIAL_KINDS) + [k for k in wb.step_kinds() if k not in _SPECIAL_KINDS]
        methods += [f"run:{k}" for k in kinds]
        cuda = False
        device = "cpu"
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                cuda = True
                idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(idx)
                device = f"cuda:{idx} · {props.name} · {props.total_memory / 2**30:.0f} GB"
        except Exception:  # noqa: BLE001 - torch is optional
            pass
        if not cuda:
            try:
                import sirius  # type: ignore

                if sirius.cuda_available():
                    cuda = True
                    p = sirius.device_properties(sirius.Device.cuda(0))
                    device = f"cuda:0 · {p.name} · {p.total_memory_bytes / 2**30:.0f} GB"
            except Exception:  # noqa: BLE001
                pass
        if self.resolved_device() == "cpu" or not cuda:
            device = f"cpu · {os.cpu_count() or 1} threads"
        return {
            "version": __version__,
            "methods": methods,
            "cuda": cuda and self.resolved_device().startswith("cuda"),
            "device": device,
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "torch": _module_version("torch"),
            "sirius": _module_version("sirius"),
            "workbench": getattr(wb, "__source_file__", getattr(wb, "__file__", "")),
        }

    # --- one connection ----------------------------------------------------------

    def _serve_client(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        send_lock = threading.Lock()
        authenticated = not self.token

        def send(header: Dict[str, Any], tensors=None) -> None:
            data = encode_frame(header, tensors)
            with send_lock:
                conn.sendall(data)

        def reply(rid, result: Dict[str, Any], tensors=None) -> None:
            send({"id": rid, "type": "result", "result": result}, tensors)

        def error(rid, message: str) -> None:
            send({"id": rid, "type": "error", "message": message})

        while not self._stop.is_set():
            try:
                header, tensors = read_frame(conn)
            except ConnectionError:
                break
            except ProtocolError as e:
                log.warning("protocol error: %s", e)
                try:
                    error(None, f"protocol error: {e}")
                except OSError:
                    pass
                break
            except OSError as e:
                log.info("connection error: %s", e)
                break
            rid = header.get("id")
            method = str(header.get("method", ""))
            params = header.get("params") or {}
            if header.get("type", "request") != "request":
                error(rid, f"unexpected frame type '{header.get('type')}'")
                continue
            try:
                if method == "hello":
                    if self.token and str(params.get("token", "")) != self.token:
                        error(rid, "authentication failed: bad token")
                        log.warning("rejected client with a bad token")
                        break
                    authenticated = True
                    reply(rid, self.capabilities())
                elif not authenticated:
                    error(rid, "not authenticated: send 'hello' with the worker token first")
                elif method == "ping":
                    reply(rid, {"time": time.time()})
                elif method == "shutdown":
                    reply(rid, {})
                    self.stop()
                    break
                elif method == "model_info":
                    reply(rid, workbench().model_info(str(params.get("path", "")), self.resolved_device()))
                elif method == "cancel":
                    target = params.get("id", header.get("target"))
                    self._cancel(target)
                    reply(rid, {"cancelled": target})
                elif method == "run":
                    self._start_run(rid, params, tensors, send)
                else:
                    error(rid, f"unknown method '{method}'")
            except Exception as e:  # noqa: BLE001 - every failure is reported to the client
                log.error("%s failed: %s", method, e)
                log.debug("%s", traceback.format_exc())
                try:
                    error(rid, _message(e))
                except OSError:
                    break
        # the connection is gone: cancel whatever is still running
        self._cancel(None)
        job = self._current_job()
        if job is not None:
            job["thread"].join(timeout=30)

    # --- jobs ----------------------------------------------------------------------

    def _current_job(self) -> Optional[Dict[str, Any]]:
        with self._job_lock:
            return self._job

    def _cancel(self, rid) -> None:
        with self._job_lock:
            job = self._job
        if job is None:
            return
        if rid is None or job["id"] == rid:
            job["cancel"].set()
            log.info("cancel requested for %s", job["id"])

    def _start_run(self, rid, params: Dict[str, Any], tensors: Dict[str, np.ndarray], send) -> None:
        """Run a request on its own thread; `send(header, tensors)` is the
        connection's locked sender, shared by progress frames and the reply."""
        with self._job_lock:
            if self._job is not None and self._job["thread"].is_alive():
                send({"id": rid, "type": "error", "message": f"busy: request {self._job['id']} is still running"})
                return
            cancel = threading.Event()
            job: Dict[str, Any] = {"id": rid, "cancel": cancel, "thread": None}
            self._job = job

        def progress(fraction: float, message: str = "") -> None:
            try:
                send({"id": rid, "type": "progress", "fraction": float(fraction), "message": str(message)})
            except OSError:
                pass

        def run() -> None:
            t0 = time.time()
            try:
                result, out = self._execute(rid, params, tensors, cancel, progress)
                if cancel.is_set():
                    send({"id": rid, "type": "error", "message": "cancelled"})
                else:
                    result["seconds"] = time.time() - t0
                    send({"id": rid, "type": "result", "result": result}, out)
            except Exception as e:  # noqa: BLE001 - every failure is reported to the client
                if cancel.is_set() or isinstance(e, _Cancelled) or e.__class__.__name__ == "Cancelled":
                    message = "cancelled"
                else:
                    log.error("run %s failed: %s", params.get("kind"), e)
                    log.debug("%s", traceback.format_exc())
                    message = _message(e)
                try:
                    send({"id": rid, "type": "error", "message": message})
                except OSError:
                    pass
            finally:
                with self._job_lock:
                    if self._job is job:
                        self._job = None

        thread = threading.Thread(target=run, name=f"sirius-run-{rid}", daemon=True)
        job["thread"] = thread
        thread.start()

    def _execute(self, rid, params: Dict[str, Any], tensors: Dict[str, np.ndarray], cancel: threading.Event,
                 progress):
        wb = workbench()
        kind = str(params.get("kind", ""))
        p = params.get("params") or {}
        device = self.resolved_device()

        def cancelled() -> bool:
            return cancel.is_set()

        def check() -> None:
            if cancel.is_set():
                raise _Cancelled()

        if kind == "torch_segment":
            volume = _tensor(tensors, "input", 3)
            model = wb.load_model(str(p.get("model") or p.get("model_path") or ""), device)
            tile = _triple(p.get("tile"), (32, 256, 256))
            ov = p.get("overlap", 32)
            overlap = _triple(ov, (4, 32, 32)) if isinstance(ov, (list, tuple, str)) else (max(1, int(ov) // 8), int(ov), int(ov))
            prob = wb.tiled_inference(volume, model, tile, overlap, device, int(p.get("pad_to", 1) or 1),
                                      str(p.get("activation", "auto")), bool(p.get("normalize", True)),
                                      progress=progress, cancelled=cancelled)
            check()
            return {"channels": int(prob.shape[0]), "device": device}, {"prob": prob}

        if kind == "sim":
            raw = tensors.get("input")
            if raw is None:
                raise ValueError("run sim: missing tensor 'input'")
            rank = raw.ndim
            meta = params.get("meta") or {}
            res = wb.run_step("sim", p, raw, meta, progress=progress, cancelled=cancelled, device=device)
            check()
            out = res.array
            if rank == 3:
                out = out[0, 0]
            return {"meta": _jsonable(res.meta), "info": _jsonable(res.info), "device": device}, {"output": out}

        if kind not in wb.step_kinds() and kind not in wb._KIND_ALIASES:  # noqa: SLF001 - same package family
            supported = ", ".join(list(_SPECIAL_KINDS) + [k for k in wb.step_kinds() if k not in _SPECIAL_KINDS])
            raise ValueError(f"unknown run kind '{kind}'; supported: {supported}")
        arr = tensors.get("input")
        if arr is None:
            raise ValueError(f"run {kind}: missing tensor 'input'")
        labels = tensors.get("labels")
        res = wb.run_step(kind, p, arr, params.get("meta") or None, labels, progress=progress,
                          cancelled=cancelled, device=device)
        check()
        out = {"output": res.array}
        if res.labels is not None:
            out["labels"] = np.ascontiguousarray(res.labels, dtype=np.uint32)
        if res.prob is not None:
            out["prob"] = res.prob
        return {"meta": _jsonable(res.meta), "info": _jsonable(res.info), "device": device}, out


# --- helpers -----------------------------------------------------------------------


def _tensor(tensors: Dict[str, np.ndarray], name: str, ndim: int) -> np.ndarray:
    a = tensors.get(name)
    if a is None:
        raise ValueError(f"missing tensor '{name}'")
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > ndim and a.shape[0] == 1:
        a = a[0]
    if a.ndim != ndim:
        raise ValueError(f"tensor '{name}' must have {ndim} dimensions, got shape {a.shape}")
    return np.ascontiguousarray(a)


def _triple(v, default) -> tuple:
    if v is None:
        return tuple(default)
    if isinstance(v, str):
        parts = [x for x in v.replace("×", ",").replace("x", ",").split(",") if x.strip()]
        v = [int(float(x)) for x in parts]
    if isinstance(v, (int, float)):
        return (int(v),) * 3
    v = [int(x) for x in v]
    return tuple(v + list(default)[len(v):])[:3]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def _message(e: BaseException) -> str:
    text = str(e).strip() or e.__class__.__name__
    return f"{e.__class__.__name__}: {text}" if not text.startswith(e.__class__.__name__) else text


def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def announce(server: WorkerServer, stream=None) -> None:
    """Print the one JSON line the launching application waits for."""
    stream = stream or sys.stdout
    stream.write(json.dumps({"port": server.port, "pid": os.getpid(), "host": server.host,
                             "device": server.resolved_device()}) + "\n")
    stream.flush()

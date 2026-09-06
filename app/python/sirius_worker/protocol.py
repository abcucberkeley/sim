"""Wire protocol between the SIRIUS application and the compute worker.

The C++ side is ``app/core/rpc.hpp``; both ends implement exactly this:

    frame := u32 header_len | header (UTF-8 JSON) | u64 payload_len | payload

Both integers are little endian. The header is an object with

    id       request id (echoed by every reply to it)
    type     "request" | "progress" | "result" | "error"
    method   "hello" | "model_info" | "run" | "cancel" | "ping" | "shutdown"   (requests)
    params   method arguments (object)
    tensors  [{"name", "dtype", "shape", "offset", "nbytes"}]  -- arrays in the payload
    message  progress / error text
    fraction progress 0..1

Tensors are raw little-endian C-order arrays concatenated in the payload at
the given byte offsets. Nothing is pickled: every value crossing the wire is
JSON or a plain array.
"""

from __future__ import annotations

import socket
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json

import numpy as np

__all__ = [
    "DTYPES",
    "FrameReader",
    "ProtocolError",
    "decode_frame",
    "encode_frame",
    "read_frame",
    "recv_exactly",
    "write_frame",
]

HEADER_LEN = struct.Struct("<I")
PAYLOAD_LEN = struct.Struct("<Q")
MAX_HEADER = 64 << 20          # a header larger than this is a corrupt stream
MAX_PAYLOAD = 1 << 40

# protocol dtype name -> numpy little-endian dtype
DTYPES: Dict[str, np.dtype] = {
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "uint8": np.dtype("u1"),
    "int8": np.dtype("i1"),
    "uint16": np.dtype("<u2"),
    "int16": np.dtype("<i2"),
    "uint32": np.dtype("<u4"),
    "int32": np.dtype("<i4"),
    "uint64": np.dtype("<u8"),
    "int64": np.dtype("<i8"),
}
_NAMES = {v: k for k, v in DTYPES.items()}

Tensors = Union[Dict[str, np.ndarray], Sequence[Tuple[str, np.ndarray]], None]


class ProtocolError(RuntimeError):
    pass


def dtype_name(a: np.ndarray) -> str:
    """Protocol name of an array's dtype (bool becomes uint8)."""
    d = a.dtype
    if d == np.bool_:
        return "uint8"
    d = d.newbyteorder("<") if d.byteorder == ">" else d
    key = np.dtype(d.str.replace(">", "<").replace("|", ""))
    for name, nd in DTYPES.items():
        if nd == key or nd.kind == d.kind and nd.itemsize == d.itemsize:
            return name
    raise ProtocolError(f"unsupported tensor dtype {a.dtype}")


def _items(tensors: Tensors) -> List[Tuple[str, np.ndarray]]:
    if not tensors:
        return []
    if isinstance(tensors, dict):
        return list(tensors.items())
    return list(tensors)


def encode_frame(header: Dict[str, Any], tensors: Tensors = None) -> bytes:
    """Serialize one frame. `tensors` are appended to the header's "tensors"
    list and their bytes concatenated into the payload."""
    header = dict(header)
    descriptors: List[Dict[str, Any]] = []
    chunks: List[bytes] = []
    offset = 0
    for name, array in _items(tensors):
        a = np.ascontiguousarray(array)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        name_ = dtype_name(a)
        a = a.astype(DTYPES[name_], copy=False)
        data = a.tobytes(order="C")
        descriptors.append({"name": str(name), "dtype": name_, "shape": [int(s) for s in a.shape],
                            "offset": offset, "nbytes": len(data)})
        chunks.append(data)
        offset += len(data)
    header["tensors"] = descriptors
    hb = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    parts = [HEADER_LEN.pack(len(hb)), hb, PAYLOAD_LEN.pack(offset)]
    parts.extend(chunks)
    return b"".join(parts)


def _decode_tensors(header: Dict[str, Any], payload: memoryview) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for d in header.get("tensors") or []:
        try:
            name = str(d["name"])
            dtype = DTYPES[str(d["dtype"])]
            shape = tuple(int(s) for s in d["shape"])
            offset = int(d["offset"])
            nbytes = int(d["nbytes"])
        except (KeyError, TypeError, ValueError) as e:
            raise ProtocolError(f"malformed tensor descriptor {d!r}: {e}") from e
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize if shape else dtype.itemsize
        if nbytes != expected or offset < 0 or offset + nbytes > len(payload):
            raise ProtocolError(f"tensor '{name}': {nbytes} bytes at {offset} do not fit shape {shape} / payload {len(payload)}")
        arr = np.frombuffer(payload[offset:offset + nbytes], dtype=dtype).reshape(shape)
        out[name] = arr.copy()  # own the memory: the buffer is reused by the reader
    return out


def decode_frame(buffer: bytearray) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """Consume one complete frame from the front of `buffer`; None when it holds
    less than a frame. Raises ProtocolError on a malformed frame."""
    if len(buffer) < HEADER_LEN.size:
        return None
    (hlen,) = HEADER_LEN.unpack_from(buffer, 0)
    if hlen > MAX_HEADER:
        raise ProtocolError(f"header length {hlen} exceeds {MAX_HEADER}")
    pos = HEADER_LEN.size + hlen
    if len(buffer) < pos + PAYLOAD_LEN.size:
        return None
    (plen,) = PAYLOAD_LEN.unpack_from(buffer, pos)
    if plen > MAX_PAYLOAD:
        raise ProtocolError(f"payload length {plen} exceeds {MAX_PAYLOAD}")
    end = pos + PAYLOAD_LEN.size + plen
    if len(buffer) < end:
        return None
    try:
        header = json.loads(bytes(buffer[HEADER_LEN.size:pos]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise ProtocolError("header is not a JSON object")
    payload = memoryview(buffer)[pos + PAYLOAD_LEN.size:end]
    tensors = _decode_tensors(header, payload)
    del payload
    del buffer[:end]
    return header, tensors


class FrameReader:
    """Incremental decoder: feed() bytes as they arrive, take complete frames."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
        self._buf.extend(data)
        frames = []
        while True:
            frame = decode_frame(self._buf)
            if frame is None:
                break
            frames.append(frame)
        return frames

    @property
    def pending(self) -> int:
        return len(self._buf)


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read n bytes or raise ConnectionError when the peer closes."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(sock: socket.socket) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Blocking read of one frame from a socket."""
    (hlen,) = HEADER_LEN.unpack(recv_exactly(sock, HEADER_LEN.size))
    if hlen > MAX_HEADER:
        raise ProtocolError(f"header length {hlen} exceeds {MAX_HEADER}")
    hb = recv_exactly(sock, hlen)
    (plen,) = PAYLOAD_LEN.unpack(recv_exactly(sock, PAYLOAD_LEN.size))
    if plen > MAX_PAYLOAD:
        raise ProtocolError(f"payload length {plen} exceeds {MAX_PAYLOAD}")
    payload = recv_exactly(sock, plen) if plen else b""
    try:
        header = json.loads(hb.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise ProtocolError("header is not a JSON object")
    return header, _decode_tensors(header, memoryview(payload))


def write_frame(sock: socket.socket, header: Dict[str, Any], tensors: Tensors = None) -> None:
    sock.sendall(encode_frame(header, tensors))

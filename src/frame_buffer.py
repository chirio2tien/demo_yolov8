import struct

import numpy as np
from multiprocessing import shared_memory

_HEADER_FMT = "IIII"  # h, w, c, active_slot (0 hoặc 1)
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_NUM_SLOTS = 2


def _slot_offset(frame_bytes: int, slot: int) -> int:
    return _HEADER_SIZE + slot * frame_bytes


class SharedFrameBuffer:
    """Double-buffer shm: capture ghi slot A, display/infer đọc slot B — zero-copy."""

    def __init__(self, name: str, max_h: int = 1440, max_w: int = 2560):
        self.name = name
        self.max_h = max_h
        self.max_w = max_w
        self._frame_bytes = max_h * max_w * 3
        self._size = _HEADER_SIZE + _NUM_SLOTS * self._frame_bytes
        self._shm = shared_memory.SharedMemory(create=True, size=self._size)

    @property
    def shm_name(self) -> str:
        return self._shm.name

    def write(self, frame: np.ndarray):
        h, w, c = frame.shape
        if h > self.max_h or w > self.max_w:
            raise ValueError(f"Frame {w}x{h} vượt buffer {self.max_w}x{self.max_h}")

        _, _, _, active = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        inactive = 1 - active
        offset = _slot_offset(self._frame_bytes, inactive)
        buf = np.ndarray((h, w, c), dtype=np.uint8, buffer=self._shm.buf, offset=offset)
        buf[:] = frame
        struct.pack_into(_HEADER_FMT, self._shm.buf, 0, h, w, c, inactive)

    def read_view(self) -> np.ndarray:
        h, w, c, active = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        if h == 0 or w == 0:
            raise RuntimeError("Chưa có frame trong shared buffer")
        offset = _slot_offset(self._frame_bytes, active)
        return np.ndarray((h, w, c), dtype=np.uint8, buffer=self._shm.buf, offset=offset)

    def read_copy(self) -> np.ndarray:
        # OpenCV cần array contiguous — shm view không dùng trực tiếp được
        return np.ascontiguousarray(self.read_view())

    def close(self):
        self._shm.close()

    def unlink(self):
        self._shm.unlink()


def attach_frame_buffer(shm_name: str, max_h: int = 1440, max_w: int = 2560):
    shm = shared_memory.SharedMemory(name=shm_name)
    return _AttachedBuffer(shm, max_h, max_w)


class _AttachedBuffer:
    def __init__(self, shm: shared_memory.SharedMemory, max_h: int, max_w: int):
        self._shm = shm
        self.max_h = max_h
        self.max_w = max_w
        self._frame_bytes = max_h * max_w * 3

    def write(self, frame: np.ndarray):
        h, w, c = frame.shape
        if h > self.max_h or w > self.max_w:
            raise ValueError(f"Frame {w}x{h} vượt buffer {self.max_w}x{self.max_h}")

        _, _, _, active = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        inactive = 1 - active
        offset = _slot_offset(self._frame_bytes, inactive)
        buf = np.ndarray((h, w, c), dtype=np.uint8, buffer=self._shm.buf, offset=offset)
        buf[:] = frame
        struct.pack_into(_HEADER_FMT, self._shm.buf, 0, h, w, c, inactive)

    def read_view(self) -> np.ndarray:
        h, w, c, active = struct.unpack_from(_HEADER_FMT, self._shm.buf, 0)
        if h == 0 or w == 0:
            raise RuntimeError("Chưa có frame trong shared buffer")
        offset = _slot_offset(self._frame_bytes, active)
        return np.ndarray((h, w, c), dtype=np.uint8, buffer=self._shm.buf, offset=offset)

    def read_copy(self) -> np.ndarray:
        return np.ascontiguousarray(self.read_view())

    def close(self):
        self._shm.close()

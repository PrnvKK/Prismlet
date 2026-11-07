"""Safetensors header reader, threaded reader, layer stream, writer."""

import json
import os
import queue
import struct
import threading

import torch

DT = {"BF16": torch.bfloat16, "F32": torch.float32}


class STFile:
    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        hdr.pop("__metadata__", None)
        self.header_len = n
        self.data_start = 8 + n
        self.t = {k: (v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
                  for k, v in hdr.items()}
        self.nbytes = os.path.getsize(self.path) - self.data_start

    def read_all(self, buf, nthreads=4, block=64 << 20):
        mv = memoryview(buf.numpy())                       # buf: 1-D CPU uint8 tensor
        blocks = [(o, min(block, self.nbytes - o)) for o in range(0, self.nbytes, block)]
        it, lock, errs = iter(blocks), threading.Lock(), []

        def work():
            try:
                with open(self.path, "rb", buffering=0) as f:
                    while True:
                        with lock:
                            b = next(it, None)
                        if b is None:
                            return
                        o, n = b
                        f.seek(self.data_start + o)
                        got = 0
                        while got < n:
                            r = f.readinto(mv[o + got: o + n])
                            if not r:
                                raise IOError("short read")
                            got += r
            except BaseException as e:
                errs.append(e)

        ths = [threading.Thread(target=work) for _ in range(nthreads)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        if errs:
            raise errs[0]

    def views(self, buf):
        out = {}
        for k, (dt, shape, a, b) in self.t.items():
            raw = buf[a:b]
            out[k] = raw.view(*shape) if dt == "F8_E4M3" else raw.view(DT[dt]).view(*shape)
        return out


class LayerStream:
    """Read layer files in ``order`` on a background thread into ``nbuf`` ring buffers."""

    def __init__(self, files, order, nbuf=2, nthreads=4, pin=True):
        self.files, self.order, self.nthreads = files, order, nthreads
        size = max(files[i].nbytes for i in order)
        self.free, self.ready = queue.Queue(), queue.Queue(maxsize=nbuf)
        self.pinned = False
        for _ in range(nbuf):
            try:
                if pin:
                    self.free.put(torch.empty(size, dtype=torch.uint8, pin_memory=True))
                    self.pinned = True
                else:
                    raise RuntimeError
            except Exception:
                self.free.put(torch.empty(size, dtype=torch.uint8))
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            for li in self.order:
                buf = self.free.get()
                self.files[li].read_all(buf, self.nthreads)
                self.ready.put((li, buf))
            self.ready.put(None)
        except BaseException as e:
            self.ready.put(e)

    def __iter__(self):
        while True:
            item = self.ready.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            li, buf = item
            yield li, self.files[li].views(buf)
            self.free.put(buf)


def write_safetensors(path_tmp, tensors):
    """Write a safetensors file atomically.

    ``tensors`` is a list of ``(name, dtype_str, shape, bytes_like)``. The final
    path is ``path_tmp`` with its last four characters removed.
    """
    hdr, off = {}, 0
    sizes = []
    for name, dt, shape, data in tensors:
        nb = memoryview(data).nbytes
        sizes.append(nb)
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "pt"}
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * ((8 - len(js) % 8) % 8)
    with open(path_tmp, "wb") as f:
        f.write(struct.pack("<Q", len(js)))
        f.write(js)
        for (_, _, _, data), nb in zip(tensors, sizes):
            f.write(memoryview(data).cast("B") if memoryview(data).format != "B" else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path_tmp, path_tmp[:-4])

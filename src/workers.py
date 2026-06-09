import os
import time
import queue as thread_queue
from multiprocessing import Queue

import cv2
from rknnlite.api import RKNNLite

from src.detector import RKNNYoloTrackerBuilder
from src.frame_buffer import attach_frame_buffer

_FFMPEG_OPTS = (
    "rtsp_transport;tcp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|analyzeduration;0"
    "|probesize;32"
)


def _notify(q: Queue, item):
    if q.full():
        try:
            q.get_nowait()
        except thread_queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except thread_queue.Full:
        pass


def _maybe_downscale(frame, max_width: int):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def capture_worker(cam_name: str, rtsp_url: str, stream_id: str,
                   shm_info: tuple, infer_in_q: Queue, stop_event,
                   capture_max_width: int = 0, frame_skip: int = 1):
    """Process riêng: decode RTSP, ghi frame vào shared memory."""
    shm_name, shm_max_h, shm_max_w = shm_info
    frame_buf = attach_frame_buffer(shm_name, shm_max_h, shm_max_w)
    frame_skip = max(1, frame_skip)
    retry, max_retries = 0, 10

    try:
        while not stop_event.is_set() and retry < max_retries:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTS
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                retry += 1
                print(f"[{cam_name}] Kết nối thất bại. Thử lại ({retry}/{max_retries})...")
                time.sleep(2)
                continue

            print(f"[{cam_name}] [{time.strftime('%H:%M:%S')}] RTSP kết nối thành công!")
            retry = 0

            while not stop_event.is_set():
                ok = True
                for _ in range(frame_skip - 1):
                    if not cap.grab():
                        ok = False
                        break
                if not ok:
                    print(f"[{cam_name}] Mất kết nối, đang kết nối lại...")
                    retry += 1
                    break

                ret, frame = cap.read()
                if not ret:
                    print(f"[{cam_name}] Mất kết nối, đang kết nối lại...")
                    retry += 1
                    break

                frame = _maybe_downscale(frame, capture_max_width)
                frame_buf.write(frame)
                _notify(infer_in_q, (stream_id, cam_name))
                del frame

            cap.release()
            if retry > 0:
                time.sleep(2)
    finally:
        frame_buf.close()
        print(f"[{cam_name}] Process capture đã dừng.")


def inference_worker(model_path: str, model_input_size: int, core_mask,
                     shm_names: dict, infer_in_q: Queue,
                     infer_out_q: Queue, stop_event, worker_name: str = "inference"):
    """Process riêng: load 1 model trên 1 NPU core, đọc frame từ shared memory."""
    frame_bufs = {
        sid: attach_frame_buffer(name, max_h, max_w)
        for sid, (name, max_h, max_w) in shm_names.items()
    }

    print(f"[{worker_name}] Load {model_path} {model_input_size}x{model_input_size} (core={core_mask})...")
    t0 = time.perf_counter()
    detector = (
        RKNNYoloTrackerBuilder()
        .set_model_path(model_path)
        .set_input_size(model_input_size)
        .set_confidence(0.20)
        .set_iou(0.45)
        .set_core_mask(core_mask)
        .build()
    )
    print(f"[{worker_name}] Model sẵn sàng: {(time.perf_counter() - t0) * 1000:.0f} ms")

    try:
        while not stop_event.is_set():
            try:
                stream_id, cam_name = infer_in_q.get(timeout=1.0)
            except thread_queue.Empty:
                continue

            try:
                frame = frame_bufs[stream_id].read_copy()
            except RuntimeError:
                continue  # capture chưa ghi frame đầu / reconnect RTSP

            detection = detector.process_frame(frame)
            _notify(infer_out_q, (stream_id, cam_name, detection))
    finally:
        for buf in frame_bufs.values():
            buf.close()
        print(f"[{worker_name}] Process inference đã dừng.")

import time
import queue
import threading
from multiprocessing import Process, Queue, Event

from src.pipeline import WebStreamer
from src.visualizer import FrameVisualizer, BoundingBoxRenderer, StatusTextRenderer
from src.frame_buffer import SharedFrameBuffer
from src.model_pool import build_model_slots, cam_to_infer_queue, MAX_CAMS_PER_MODEL
from src.workers import capture_worker, inference_worker

_QUEUE_SIZE = 2
_DISPLAY_INTERVAL = 1.0 / 25


def _build_visualizer() -> FrameVisualizer:
    return (
        FrameVisualizer()
        .add_renderer(BoundingBoxRenderer(font_scale=0.7, thickness=2))
        .add_renderer(StatusTextRenderer())
    )


def _infer_cache_loop(infer_out_q: Queue, detection_cache: dict,
                      cache_lock: threading.Lock, stop_event: Event):
    infer_fps = {}

    while not stop_event.is_set():
        try:
            stream_id, cam_name, detection = infer_out_q.get(timeout=1.0)
        except queue.Empty:
            continue

        state = infer_fps.setdefault(stream_id, {'cam_name': cam_name, 'count': 0, 'times': []})
        state['count'] += 1
        now = time.time()
        state['times'].append(now)
        if len(state['times']) > 30:
            state['times'].pop(0)
        times = state['times']
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

        with cache_lock:
            detection_cache[stream_id] = {
                'detection': detection,
                'infer_fps': fps,
                'frame_count': state['count'],
            }

        if state['count'] % 30 == 0:
            print(f"[{cam_name}] Infer FPS: {fps:.1f}")


def _display_loop(stream_id: str, cam_name: str, frame_buffer: SharedFrameBuffer,
                  streamer: WebStreamer, visualizer: FrameVisualizer,
                  detection_cache: dict, cache_lock: threading.Lock,
                  stop_event: Event):
    display_fps = {'count': 0, 'times': []}

    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            frame = frame_buffer.read_copy()  # copy 1 lần — OpenCV không vẽ trên shm view
        except RuntimeError:
            time.sleep(0.01)
            continue

        with cache_lock:
            cached = detection_cache.get(stream_id)

        detection = cached['detection'] if cached else None
        infer_fps = cached['infer_fps'] if cached else 0.0
        infer_count = cached['frame_count'] if cached else 0

        display_fps['count'] += 1
        now = time.time()
        display_fps['times'].append(now)
        if len(display_fps['times']) > 30:
            display_fps['times'].pop(0)
        times = display_fps['times']
        show_fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

        context = {
            'results':        detection,
            'num_detections': len(detection) if detection else 0,
            'fps':            infer_fps,
            'infer_fps':      infer_fps,
            'current_fps':    show_fps,
            'frame_count':    infer_count,
        }

        frame = visualizer.draw(frame, context)
        streamer.update(frame)
        del frame

        if display_fps['count'] % 75 == 0:
            print(f"[{cam_name}] Display FPS: {show_fps:.1f}")

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, _DISPLAY_INTERVAL - elapsed))


def main():
    MODEL_PATH = "yolov8n_416.rknn"
    MODEL_INPUT_SIZE = 416
    # (tên cam, stream_id, rtsp_url, web_port)
    CAMERAS = [
        ("CAM 1", "cam1",
         "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0", 5002),
        ("CAM 2", "cam2",
         "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0", 5003),
         ("CAM 3", "cam3",
         "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0", 5004),
    ]

    stop_event = Event()
    infer_out_q = Queue(maxsize=_QUEUE_SIZE)
    detection_cache = {}
    cache_lock = threading.Lock()

    slots = build_model_slots(CAMERAS, _QUEUE_SIZE)
    cam_infer_q = cam_to_infer_queue(slots)

    frame_buffers = {
        stream_id: SharedFrameBuffer(name=stream_id)
        for _, stream_id, _, _ in CAMERAS
    }
    shm_names = {sid: buf.shm_name for sid, buf in frame_buffers.items()}

    streamers = {}
    visualizers = {}
    for cam_name, stream_id, _, port in CAMERAS:
        streamers[stream_id] = WebStreamer(port=port, title=cam_name)
        visualizers[stream_id] = _build_visualizer()

    threading.Thread(
        target=_infer_cache_loop,
        args=(infer_out_q, detection_cache, cache_lock, stop_event),
        daemon=True,
        name="InferCache",
    ).start()

    for cam_name, stream_id, _, _ in CAMERAS:
        threading.Thread(
            target=_display_loop,
            args=(stream_id, cam_name, frame_buffers[stream_id],
                  streamers[stream_id], visualizers[stream_id],
                  detection_cache, cache_lock, stop_event),
            daemon=True,
            name=f"{cam_name}-Display",
        ).start()

    processes = []
    for slot in slots:
        worker_shm = {sid: shm_names[sid] for sid in slot['stream_ids']}
        cam_names = ', '.join(entry[0] for entry in slot['cameras'])
        print(
            f"[main] {slot['label']} | {MODEL_PATH} | core={slot['npu_core']} "
            f"| cams=[{cam_names}]"
        )
        processes.append(Process(
            target=inference_worker,
            args=(MODEL_PATH, MODEL_INPUT_SIZE, slot['npu_core'],
                  worker_shm, slot['infer_in_q'], infer_out_q, stop_event,
                  slot['label']),
            name=slot['label'],
            daemon=True,
        ))

    for cam_name, stream_id, rtsp_url, _ in CAMERAS:
        processes.append(Process(
            target=capture_worker,
            args=(cam_name, rtsp_url, stream_id,
                  shm_names[stream_id], cam_infer_q[stream_id], stop_event),
            name=f"{cam_name}-Capture",
            daemon=True,
        ))

    print(
        f"[main] Model pool: {len(slots)} instance(s), "
        f"{len(CAMERAS)} cam, max {MAX_CAMS_PER_MODEL} cam/model"
    )
    for proc in processes:
        proc.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nNgười dùng bấm Ctrl+C. Đang dừng...")
        stop_event.set()
        for proc in processes:
            proc.join(timeout=5)
        for s in streamers.values():
            s.stop()
        for buf in frame_buffers.values():
            buf.close()
            buf.unlink()


if __name__ == "__main__":
    main()

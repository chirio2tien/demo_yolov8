import time
import threading
from rknnlite.api import RKNNLite
from src.detector import RKNNYoloTrackerBuilder
from src.visualizer import FrameVisualizer, BoundingBoxRenderer, StatusTextRenderer
from src.pipeline import CameraCapturePipeline


def build_pipeline(cam_name: str, rtsp_url: str, model_path: str,
                   web_port: int, npu_core) -> CameraCapturePipeline:
    #Khởi tạo detector + visualizer rồi đóng gói vào pipeline 3-luồng.
    print(f"[{cam_name}] Đang khởi tạo Model trên lõi NPU {npu_core}...")
    t0 = time.perf_counter()

    detector = (
        RKNNYoloTrackerBuilder()
        .set_model_path(model_path)
        .set_confidence(0.20)
        .set_iou(0.45)
        .set_core_mask(npu_core)
        .build()
    )
    print(f"[{cam_name}] Init Model: {(time.perf_counter() - t0) * 1000:.2f} ms")

    visualizer = (
        FrameVisualizer()
        .add_renderer(BoundingBoxRenderer(font_scale=0.7, thickness=2))
        .add_renderer(StatusTextRenderer())
    )

    return CameraCapturePipeline(
        cam_name=cam_name,
        rtsp_url=rtsp_url,
        detector=detector,
        visualizer=visualizer,
        web_port=web_port,
    )


def _build_parallel(configs: list) -> list:
    #Khởi tạo nhiều pipeline song song 
    results = [None] * len(configs)

    def _init(i, cfg):
        results[i] = build_pipeline(*cfg)

    threads = [threading.Thread(target=_init, args=(i, cfg), daemon=True)
               for i, cfg in enumerate(configs)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[main] Tổng thời gian init song song: {(time.perf_counter()-t0)*1000:.0f} ms")
    return results


def main():
    MODEL_PATH = "yolov8n_416.rknn"
    cam1_url = "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0"
    cam2_url = "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0"

    pipes = _build_parallel([
        ("CAM 1", cam1_url, MODEL_PATH, 5002, RKNNLite.NPU_CORE_0),
        ("CAM 2", cam2_url, MODEL_PATH, 5003, RKNNLite.NPU_CORE_1),
    ])

    for pipe in pipes:
        pipe.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nNgười dùng bấm Ctrl+C. Đang dừng...")
        for pipe in pipes:
            pipe.stop()


if __name__ == "__main__":
    main()

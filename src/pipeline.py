import cv2
import time
import gc
import queue
import threading
import logging
import os
from flask import Flask, Response, render_template_string

logging.basicConfig(level=logging.WARNING)

_GC_INTERVAL = 50


_STREAM_MAX_WIDTH = 1280
_STREAM_JPEG_QUALITY = 70


def _drop_old_and_put(q: queue.Queue, item):
   
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


class WebStreamer:

    def __init__(self, port: int = 5000):
        self.port = port
        self.latest_frame = None
        self._lock = threading.Lock()
        self.is_running = True

        self._app = Flask(__name__)

        @self._app.route('/')
        def index():
            html = """
            <html>
                <head>
                    <title>Camera AI Radxa</title>
                    <style>
                        body { margin: 0; padding: 0; background-color: #000; overflow: hidden; }
                        img  { width: 100vw; height: 100vh; object-fit: contain; border: none; }
                    </style>
                </head>
                <body><img src="/video_feed" /></body>
            </html>
            """
            return render_template_string(html)

        @self._app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        print(f" Web Server đang chạy tại: http://localhost:{self.port}")

    def _run_server(self):
        self._app.run(
            host='0.0.0.0',
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )

    def _generate(self):
      
        last_frame = None
        while self.is_running:
            with self._lock:
                frame = self.latest_frame

            if frame is None or frame is last_frame:
                time.sleep(0.005)
                continue

            last_frame = frame  

            h, w = frame.shape[:2]
            if w > _STREAM_MAX_WIDTH:
                scale = _STREAM_MAX_WIDTH / w
                stream_frame = cv2.resize(
                    frame, (_STREAM_MAX_WIDTH, int(h * scale)),
                    interpolation=cv2.INTER_AREA
                )
            else:
                stream_frame = frame
            ret, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_JPEG_QUALITY])
            if stream_frame is not frame:
                del stream_frame
            frame = None  

            if not ret:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + buffer.tobytes()
                + b'\r\n'
            )
          
    def update(self, frame):
        with self._lock:
            self.latest_frame = frame

    def stop(self):
        self.is_running = False


class CameraCapturePipeline:
  

    def __init__(self, cam_name: str, rtsp_url: str, detector, visualizer, web_port: int):
        self.cam_name  = cam_name
        self.rtsp_url  = rtsp_url
        self.detector  = detector
        self.visualizer = visualizer

        self.frame_queue  = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)

        self._stop = threading.Event()
        self.streamer = WebStreamer(port=web_port)

    def start(self):
        for target, name in [
            (self._capture_loop,   f"{self.cam_name}-Capture"),
            (self._inference_loop, f"{self.cam_name}-Inference"),
            (self._output_loop,    f"{self.cam_name}-Output"),
        ]:
            threading.Thread(target=target, daemon=True, name=name).start()
        print(f"[{self.cam_name}] Đã khởi động 3 luồng daemon.")

    def stop(self):
        self._stop.set()
        self.streamer.stop()


    def _capture_loop(self):

        retry, max_retries = 0, 10

        # FFmpeg: tắt buffer nội bộ, transport TCP ổn định hơn UDP
        _FFMPEG_OPTS = (
            "rtsp_transport;tcp"
            "|fflags;nobuffer"
            "|flags;low_delay"
            "|analyzeduration;0"
            "|probesize;32"
        )

        while not self._stop.is_set() and retry < max_retries:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _FFMPEG_OPTS
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                retry += 1
                print(f"[{self.cam_name}] Kết nối thất bại. Thử lại ({retry}/{max_retries})...")
                time.sleep(2)
                continue

            print(f"[{self.cam_name}] RTSP kết nối thành công!")
            retry = 0

            while not self._stop.is_set():
                ret, frame = cap.read()

                if not ret:
                    print(f"[{self.cam_name}] Mất kết nối, đang kết nối lại...")
                    retry += 1
                    break

                _drop_old_and_put(self.frame_queue, frame)

                del frame

            cap.release()
            if retry > 0:
                time.sleep(2)

        print(f"[{self.cam_name}] Luồng Capture đã dừng.")



    def _inference_loop(self):
       
        frame_count = 0
        fps_times   = []  

        while not self._stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_count += 1

            detection = self.detector.process_frame(frame)

            now = time.time()
            fps_times.append(now)
            if len(fps_times) > 30:
                fps_times.pop(0)
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0.0

            context = {
                'results':        detection,
                'num_detections': len(detection) if detection else 0,
                'fps':            fps,
                'current_fps':    fps,
                'frame_count':    frame_count,
            }

            _drop_old_and_put(self.result_queue, (frame, context))

            del frame

            if frame_count % 30 == 0:
                print(f"[{self.cam_name}] Pipeline FPS: {fps:.1f}")

            if frame_count % _GC_INTERVAL == 0:
                gc.collect()

        print(f"[{self.cam_name}] Luồng Inference đã dừng.")



    def _output_loop(self):
       
        while not self._stop.is_set():
            try:
                frame, context = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame = self.visualizer.draw(frame, context)

            self.streamer.update(frame)

            del frame

        print(f"[{self.cam_name}] Luồng Output đã dừng.")

import cv2
import time
import numpy as np
import os
from pathlib import Path
from abc import ABC, abstractmethod
from flask import Flask, Response, render_template_string
import threading
import logging

# Thiết lập bộ lọc Log để tránh làm rác Terminal
logging.basicConfig(level=logging.WARNING)


class PipelineStep(ABC):
    @abstractmethod
    def process(self, frame, context):
        """
        Nhận vào frame và context (dữ liệu chia sẻ), 
        xử lý, và trả về frame mới.
        """
        pass
        
    def cleanup(self):
        """Dọn dẹp tài nguyên khi kết thúc (nếu có)"""
        pass


class WebStreamStep(PipelineStep):
    """Trạm Web: Phát Video trực tiếp lên Trình duyệt Web và OBS (Bản nét căng)"""
    def __init__(self, port=5000):
        self.port = port
        self.app = Flask(__name__)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_running = False

        # TỐI ƯU 1: Xóa bỏ thẻ Meta Refresh gây lỗi quay web.
        # Xóa các viền thừa, ép video full màn hình cực mượt để OBS bắt nét.
        @self.app.route('/')
        def index():
            html = """
            <html>
                <head>
                    <title>Camera AI Radxa</title>
                    <style>
                        body { margin: 0; padding: 0; background-color: #000; overflow: hidden; }
                        img { width: 100vw; height: 100vh; object-fit: contain; border: none; }
                    </style>
                </head>
                <body>
                    <img src="/video_feed" />
                </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        try:
            self.web_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.web_thread.start()
            self.is_running = True
            print(f" Web Server đang chạy tại: http://localhost:{self.port}")
        except Exception as e:
            print(f"Error starting web server: {e}")
            self.is_running = False

    def _run_server(self):
        try:
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False,
                threaded=True
            )
        except Exception as e:
            print(f" Web server error: {e}")
            self.is_running = False

    def generate(self):
        """Liên tục nén ảnh JPEG và đẩy luồng dữ liệu mạng lên Client"""
        while self.is_running:
            try:
                with self.lock:
                    if self.latest_frame is None:
                        time.sleep(0.01)
                        continue
                    
                    stream_frame = cv2.resize(
                        self.latest_frame,
                        (1920, 1080),
                        interpolation=cv2.INTER_LINEAR
                        )
                    ret, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.01)
            except Exception as e:
                time.sleep(0.1)

    def process(self, frame, context):
        if self.is_running:
            try:
                with self.lock:
                    self.latest_frame = frame.copy()
            except Exception as e:
                pass
        return frame
    
    def cleanup(self):
        self.is_running = False

class AIDetectorStep(PipelineStep):
    """Bước phát hiện AI: Đẩy ma trận ảnh vào lõi xử lý NPU"""
    def __init__(self, detector):
        self.detector = detector
        self.frame_count = 0

    def process(self, frame, context):
        self.frame_count += 1
        try:
            # Nhận trực tiếp đối tượng DetectionResult hiệu năng cao từ lõi xử lý mới
            detection_result = self.detector.process_frame(frame)
            
            # Lưu trực tiếp vào context, loại bỏ hoàn toàn bộ Adapter mô phỏng cũ
            context['results'] = detection_result
            context['num_detections'] = len(detection_result) if detection_result is not None else 0
        except Exception as e:
            print(f" Detector error at frame {self.frame_count}: {e}")
            context['results'] = None
            context['num_detections'] = 0
        
        return frame


class VisualizationStep(PipelineStep):
    #Vẽ khung Bounding Box và Dashboard thông số lên Frame
    def __init__(self, visualizer):
        self.visualizer = visualizer

    def process(self, frame, context):
        try:
            return self.visualizer.draw(frame, context)
        except Exception as e:
            print(f" Visualization error: {e}")
            return frame


class TelemetryStep(PipelineStep):
    #Tính FPS thời gian thực
    def __init__(self, monitor=None):
        self.monitor = monitor
        self.frame_times = []

    def process(self, frame, context):
        try:
            start_time = context.get('start_time', time.time())
            frame_time = time.time() - start_time
            
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30: 
                self.frame_times.pop(0)
                
            current_fps = 1 / np.mean(self.frame_times) if self.frame_times else 0
            
            # Đồng bộ các biến FPS cho module hiển thị trực quan đọc dữ liệu
            context['fps'] = current_fps
            context['current_fps'] = current_fps
            context['status_text'] = f"Frame: {context.get('frame_count', 0)} | FPS: {current_fps:.1f}"
            
            if self.monitor:
                try:
                    metrics = self.monitor.get_metrics(context.get('frame_count', 0), current_fps, context.get('num_detections', 0))
                    self.monitor.log_metrics(metrics)
                    if context.get('frame_count', 0) % 30 == 0: 
                        self.monitor.print_metrics(metrics)
                except Exception as e:
                    print(f" Monitor error: {e}")
        except Exception as e:
            print(f" Telemetry error: {e}")
            context['fps'] = 0
            context['current_fps'] = 0
            
        return frame


class DisplayAndSaveStep(PipelineStep):
    #Bước xuất dữ liệu ra màn hình cục bộ 
    def __init__(self, display=True, output_path=None, fps=100, size=(640, 480)):
        self.display = display
        self.video_writer = None
        self.has_display = display and os.environ.get('DISPLAY') is not None
        self.frame_count = 0
        
        if output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, size)
                if not self.video_writer.isOpened():
                    print(f" Failed to open video writer: {output_path}")
                    self.video_writer = None
                else:
                    print(f" Recording to: {output_path}")
            except Exception as e:
                print(f" Error initializing video writer: {e}")
                self.video_writer = None

    def process(self, frame, context):
        self.frame_count += 1
        
        if self.has_display:
            try:
                cv2.imshow('Human Tracking Pipeline', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    context['stop_requested'] = True
            except Exception as e:
                if self.frame_count == 1:
                    print(f" Display error: {e}")
                self.has_display = False
                
        if self.video_writer:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f" Error writing frame {self.frame_count}: {e}")
            
        return frame
        
    def cleanup(self):
        if self.video_writer:
            try: self.video_writer.release()
            except: pass
        try: cv2.destroyAllWindows()
        except: pass


class VideoPipeline:
    # lý luồng dữ liệu Camera 
    def __init__(self):
        self.steps = []

    def add_step(self, step: PipelineStep):
        self.steps.append(step)
        return self

    def run(self, video_path, max_retries=3):
        is_rtsp = str(video_path).startswith("rtsp://")
        if not is_rtsp and not Path(video_path).exists(): 
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")
        
        retry_count = 0
        
        while retry_count < (max_retries if is_rtsp else 1):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if is_rtsp:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Giảm bộ đệm xuống tối thiểu để triệt tiêu hiện tượng lag hình
                if retry_count == 0:
                    print(f" Connecting to RTSP stream: {video_path}")
            
            if not cap.isOpened():
                if is_rtsp:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f" Connection failed. Retrying... ({retry_count}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        raise RuntimeError(f"Failed to connect to RTSP after {max_retries} attempts: {video_path}")
                else:
                    raise RuntimeError(f"Failed to open video: {video_path}")
            
            if is_rtsp and retry_count == 0:
                print(" Connected successfully!\n")
            
            frame_count = 0
            context = {
                'stop_requested': False,
                'is_live': is_rtsp,
                'reconnect_attempts': retry_count
            }
            
            try:
                while cap.isOpened() and not context['stop_requested']:
                    start_time = time.time()
                    cap.grab()
                    ret, frame = cap.retrieve()
                    
                    if not ret:
                        if is_rtsp:
                            print("\n Lost connection to RTSP stream. Attempting to reconnect...")
                            retry_count += 1
                            break
                        else:
                            break
                    
                    frame_count += 1
                    retry_count = 0  
                    
                    context['frame_count'] = frame_count
                    context['start_time'] = start_time
                    
                    try:
                        for step in self.steps:
                            frame = step.process(frame, context)
                    except Exception as e:
                        print(f" Error in pipeline step: {e}")
                        continue
                    
            except KeyboardInterrupt:
                print("\n Pipeline interrupted by user.")
                context['stop_requested'] = True
            except Exception as e:
                print(f" Error in pipeline loop: {e}")
            finally:
                cap.release()
                
                if context['stop_requested']:
                    break
                
                if is_rtsp and retry_count < max_retries:
                    time.sleep(2)
                else:
                    break
        
        print("\n Cleaning up resources...")
        for step in self.steps:
            try: step.cleanup()
            except Exception as e: print(f" Error cleaning up step: {e}")
        
        print(" Pipeline stopped cleanly.")
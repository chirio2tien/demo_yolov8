import cv2
import time
import threading
import numpy as np
from rknnlite.api import RKNNLite

_DEBUG_TIMING = True
def _suppress_contained_boxes(xyxy_list, conf_list, contain_ratio=0.7):
    n = len(xyxy_list)
    if n <= 1:
        return xyxy_list, conf_list

    boxes = np.asarray(xyxy_list, dtype=np.float32) 
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    is_smaller = areas[:, None] <= areas[None, :] * contain_ratio
    ratio      = inter / (areas[:, None] + 1e-9)
    contained  = np.any((ratio >= contain_ratio) & is_smaller, axis=1)

    idx = [i for i in range(n) if not contained[i]]
    return [xyxy_list[i] for i in idx], [conf_list[i] for i in idx]


class DetectionResult:
    def __init__(self, xyxy, confidence, class_ids=None):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_ids = class_ids

    def __len__(self):
        return len(self.xyxy) if self.xyxy is not None else 0

    def __getitem__(self, idx):
        class DummyTensor:
            def __init__(self, data): self.data = np.array(data)
            def cpu(self): return self
            def numpy(self): return self.data
            def astype(self, t): return self.data.astype(t)

        class FakeBoxes:
            def __init__(self, xyxy, conf):
                self.xyxy = DummyTensor(xyxy)
                self.conf = DummyTensor(conf)
                self.id = None
            def __len__(self): return len(self.xyxy.data)

        if self.xyxy is None or len(self.xyxy) == 0:
            return type('FakeResult', (object,), {'boxes': None})()
        return type('FakeResult', (object,), {'boxes': FakeBoxes(self.xyxy, self.confidence)})()


class RKNNYoloTracker:
    def __init__(self, builder):
        self.conf_thresh    = builder.confidence
        self.iou_thresh     = builder.iou
        self.human_class_id = builder.human_class_id
        self.core_mask      = builder.core_mask
        self.async_mode     = builder.async_mode
        self.input_size     = builder.input_size
        self._local = threading.local()  # per-thread stats + letterbox canvas
        self._lock  = threading.Lock()

        t_rknn = time.perf_counter()
        self.rknn = RKNNLite()
        print(f"[{time.strftime('%H:%M:%S')}] RKNNLite() constructor: {(time.perf_counter() - t_rknn)*1000:.0f} ms")
        self._initialize_model(builder.model_path)

    def _initialize_model(self, model_path: str):
        t0 = time.perf_counter()
        ts0 = time.strftime('%H:%M:%S')
        print(f"[{ts0}] Đang tải RKNN model từ {model_path}...")
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError("Lỗi khi load model RKNN!")
        t_load = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] Load model xong: {(t_load - t0)*1000:.0f} ms")

        print(f"[{time.strftime('%H:%M:%S')}] Đang khởi tạo NPU (Core Mask: {self.core_mask})...")
        if self.rknn.init_runtime(core_mask=self.core_mask,
                                   async_mode=self.async_mode) != 0:
            raise RuntimeError("Lỗi khởi tạo NPU Runtime!")
        t_init = time.perf_counter()
        print(
            f"[{time.strftime('%H:%M:%S')}] Init NPU xong: {(t_init - t_load)*1000:.0f} ms"
            f" | Tổng khởi tạo model: {(t_init - t0)*1000:.0f} ms"
        )

        dummy = np.zeros((1, self.input_size, self.input_size, 3), dtype=np.uint8)
        t_warm = time.perf_counter()
        self.rknn.inference(inputs=[dummy])
        print(f"[{time.strftime('%H:%M:%S')}] Warm-up inference: {(time.perf_counter() - t_warm)*1000:.0f} ms")

    def _get_local(self):
        loc = self._local
        if not hasattr(loc, 'frame'):
            loc.frame = 0
            loc.sum_pre = loc.sum_inf = loc.sum_out = 0.0
            loc.cam_name = threading.current_thread().name
        return loc

    def process_frame(self, frame):
        loc = self._get_local()
        loc.frame += 1
        img_h, img_w = frame.shape[:2]

        if loc.frame == 1:
            print(f"[{loc.cam_name}] Độ phân giải nguồn camera: {img_w}x{img_h}")

        start_input = time.perf_counter()
        img_input, ratio, dw, dh = self._letterbox(frame)
        img_expanded = np.expand_dims(img_input, axis=0)
        end_input = time.perf_counter()
        loc.sum_pre += end_input - start_input

        start_predict = time.perf_counter()
        with self._lock:
            outputs = self.rknn.inference(inputs=[img_expanded])
        end_predict = time.perf_counter()
        loc.sum_inf += end_predict - start_predict

        start_output = time.perf_counter()
        results = self._decode_and_filter(outputs, img_w, img_h, ratio, dw, dh)
        end_output = time.perf_counter()
        loc.sum_out += end_output - start_output

        if _DEBUG_TIMING and loc.frame % 30 == 0:
            avg_pre = loc.sum_pre / 30 * 1000
            avg_inf = loc.sum_inf / 30 * 1000
            avg_out = loc.sum_out / 30 * 1000
            total   = avg_pre + avg_inf + avg_out
            print(
                f"\n--- [{loc.cam_name}] 30 FRAME (frame {loc.frame}) ---"
                f"\n1. Input     : {avg_pre:.2f} ms"
                f"\n2. Inference : {avg_inf:.2f} ms"
                f"\n3. Output    : {avg_out:.2f} ms"
                f"\n>> TỔNG      : {total:.2f} ms (~{1000/total:.1f} FPS)"
            )
            loc.sum_pre = loc.sum_inf = loc.sum_out = 0.0

        return results

    def _decode_and_filter(self, outputs, img_w, img_h, ratio, dw, dh):
        if outputs is None:
            return DetectionResult(None, None)

    
        box_tensors   = [x for x in outputs if len(x.shape) == 4 and x.shape[1] == 64]
        score_tensors = [x for x in outputs if len(x.shape) == 4 and x.shape[1] != 64]

        all_boxes, all_scores = [], []
        num_classes = max(x.shape[1] for x in score_tensors) if score_tensors else 80

        for box_t in box_tensors:
            H, W   = box_t.shape[2], box_t.shape[3]
            stride = self.input_size / H  

            cls_cands = [x for x in outputs
                         if x.shape[2] == H and x.shape[3] == W and x.shape[1] == num_classes]
            if not cls_cands:
                continue

            # Chỉ lấy class người — tiết kiệm 80× phép tính 
            cls_t = cls_cands[0][0, self.human_class_id:self.human_class_id + 1, :, :].transpose(1, 2, 0)

            if np.max(cls_t) > 1.0 or np.min(cls_t) < 0.0:
                cls_t = np.clip(cls_t, -20, 20)
                cls_t = 1.0 / (1.0 + np.exp(-cls_t))

            score_mask = cls_t[..., 0] > self.conf_thresh
            if not np.any(score_mask):
                continue

            valid_cls  = cls_t[score_mask]
            grid_y, grid_x = np.mgrid[0:H, 0:W]
            valid_grid = np.stack((grid_x, grid_y), axis=-1)[score_mask]

            b_t        = box_t[0].transpose(1, 2, 0).reshape(H, W, 4, 16)
            valid_b_t  = b_t[score_mask]

            valid_b_t  = np.exp(valid_b_t - np.max(valid_b_t, axis=-1, keepdims=True))
            valid_b_t /= np.sum(valid_b_t, axis=-1, keepdims=True)
            dfl_out    = np.sum(valid_b_t * np.arange(16, dtype=np.float32), axis=-1)

            lt, rb = dfl_out[..., :2], dfl_out[..., 2:]
            x1y1   = valid_grid - lt
            x2y2   = valid_grid + rb
            cx_cy  = (x1y1 + x2y2) / 2 * stride
            w_h    = (x2y2 - x1y1) * stride
            xywh   = np.concatenate((cx_cy, w_h), axis=-1)

            all_boxes.append(xywh)
            all_scores.append(valid_cls[..., 0])

        if not all_boxes:
            return DetectionResult(None, None)

        boxes  = np.concatenate(all_boxes,  axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # xywh → xyxy trong ảnh letterbox, rồi scale về tọa độ gốc (vectorized)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = np.clip((cx - w / 2 - dw) / ratio, 0, img_w)
        y1 = np.clip((cy - h / 2 - dh) / ratio, 0, img_h)
        x2 = np.clip((cx + w / 2 - dw) / ratio, 0, img_w)
        y2 = np.clip((cy + h / 2 - dh) / ratio, 0, img_h)

        # NMS cần xywh (top-left + wh)
        nms_input = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices   = cv2.dnn.NMSBoxes(
            nms_input.tolist(), scores.tolist(),
            self.conf_thresh, self.iou_thresh
        )

        if indices is None or len(indices) == 0:
            return DetectionResult(None, None)

        idx = np.array(indices).flatten()
        nms_xyxy = np.stack([x1[idx], y1[idx], x2[idx], y2[idx]], axis=1).astype(int).tolist()
        nms_conf = scores[idx].tolist()

        nms_xyxy, nms_conf = _suppress_contained_boxes(nms_xyxy, nms_conf)
        if not nms_xyxy:
            return DetectionResult(None, None)

        return DetectionResult(
            np.array(nms_xyxy),
            np.array(nms_conf),
            np.full(len(nms_xyxy), self.human_class_id)
        )

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r  = min(self.input_size / h, self.input_size / w)
        nw = int(w * r)
        nh = int(h * r)
        dw = (self.input_size - nw) // 2
        dh = (self.input_size - nh) // 2

        loc = self._get_local()
        if getattr(loc, 'canvas_src', None) != (h, w):
            loc.canvas     = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
            loc.canvas_src = (h, w)

        loc.canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
        return loc.canvas, r, dw, dh

    def __del__(self):
        if hasattr(self, 'rknn'):
            try:
                self.rknn.release()
            except Exception:
                pass


class RKNNYoloTrackerBuilder:
    def __init__(self):
        self.model_path     = 'yolov8n.rknn'
        self.confidence     = 0.20
        self.iou            = 0.55
        self.human_class_id = 0
        self.core_mask      = RKNNLite.NPU_CORE_0_1_2
        self.async_mode     = False
        self.input_size     = 640

    def set_model_path(self, path):
        self.model_path = path
        return self

    def set_async_mode(self, enabled):
        self.async_mode = enabled
        return self

    def set_confidence(self, conf):
        self.confidence = conf
        return self

    def set_iou(self, iou):
        self.iou = iou
        return self

    def set_core_mask(self, core_mask):
        self.core_mask = core_mask
        return self

    def set_input_size(self, size: int):
        self.input_size = size
        return self

    def build(self) -> RKNNYoloTracker:
        return RKNNYoloTracker(self)

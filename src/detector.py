import cv2
import numpy as np
from rknnlite.api import RKNNLite


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

        # Giả lập cấu trúc trả về giống hệt YOLOv8 PyTorch
        if self.xyxy is None or len(self.xyxy) == 0:
            return type('FakeResult', (object,), {'boxes': None})()
            
        return type('FakeResult', (object,), {'boxes': FakeBoxes(self.xyxy, self.confidence)})()


class RKNNYoloTracker:
    def __init__(self, builder):
        self.conf_thresh = builder.confidence
        self.iou_thresh = builder.iou
        self.human_class_id = builder.human_class_id
        self.core_mask = builder.core_mask
        
        self.rknn = RKNNLite()
        self._initialize_model(builder.model_path)
    
    def _initialize_model(self, model_path: str):
        print(f"Đang tải RKNN model từ {model_path}...")
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError("Lỗi khi load model RKNN!")
            
        print(f"Đang khởi tạo NPU (Core Mask: {self.core_mask})...")
        if self.rknn.init_runtime(core_mask=self.core_mask) != 0:
            raise RuntimeError("Lỗi khởi tạo NPU Runtime!")
        print("Khởi tạo NPU thành công!")

    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        
        img_input, ratio, dw, dh = self._letterbox(frame)

        img_expanded = np.expand_dims(img_input, axis=0)

        outputs = self.rknn.inference(inputs=[img_expanded])

        return self._decode_and_filter(
            outputs,
            img_w,
            img_h,
            ratio,
            dw,
            dh
        )

    def _decode_and_filter(
        self,
        outputs,
        img_w,
        img_h,
        ratio,
        dw,
        dh
    ):
        box_tensors = [x for x in outputs if len(x.shape) == 4 and x.shape[1] == 64]
        score_tensors = [x for x in outputs if len(x.shape) == 4 and x.shape[1] != 64]
        
        all_boxes, all_scores = [], []
        num_classes = max([x.shape[1] for x in score_tensors]) if score_tensors else 80

        for box_t in box_tensors:
            H, W = box_t.shape[2], box_t.shape[3]
            stride = 640 / H
            
            cls_cands = [x for x in outputs if x.shape[2] == H and x.shape[3] == W and x.shape[1] == num_classes]
            if not cls_cands: continue
            
            # TỐI ƯU 1: Cắt ngang ma trận, chỉ lấy điểm của Class Người (Tiết kiệm 80x phép tính)
            cls_t = cls_cands[0][0, self.human_class_id:self.human_class_id+1, :, :].transpose(1, 2, 0)
            
            # Giải quyết Anchor Grid Flood (Double Sigmoid)
            if np.max(cls_t) > 1.0 or np.min(cls_t) < 0.0:
                cls_t = np.clip(cls_t, -20, 20)
                cls_t = 1 / (1 + np.exp(-cls_t))
                
            # TỐI ƯU 2: Lọc rác trước khi giải mã tọa độ (Tiết kiệm tài nguyên DFL)
            score_mask = cls_t[..., 0] > self.conf_thresh
            if not np.any(score_mask):
                continue
                
            valid_cls = cls_t[score_mask]
            
            grid_y, grid_x = np.mgrid[0:H, 0:W]
            valid_grid = np.stack((grid_x, grid_y), axis=-1)[score_mask]
            
            b_t = box_t[0].transpose(1, 2, 0).reshape(H, W, 4, 16)
            valid_b_t = b_t[score_mask]
            
            # DFL Box Decoding (Chỉ chạy trên các hộp có khả năng là người)
            valid_b_t = np.exp(valid_b_t - np.max(valid_b_t, axis=-1, keepdims=True))
            valid_b_t = valid_b_t / np.sum(valid_b_t, axis=-1, keepdims=True)
            
            weights = np.arange(16, dtype=np.float32)
            dfl_out = np.sum(valid_b_t * weights, axis=-1)
            
            lt, rb = dfl_out[..., :2], dfl_out[..., 2:]
            x1y1 = valid_grid - lt
            x2y2 = valid_grid + rb
            
            cx_cy = (x1y1 + x2y2) / 2 * stride
            w_h = (x2y2 - x1y1) * stride
            xywh = np.concatenate((cx_cy, w_h), axis=-1)
            
            all_boxes.append(xywh)
            all_scores.append(valid_cls[..., 0])

        if not all_boxes:
            return DetectionResult(None, None)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # 4. Scale về kích thước gốc
        x_scale, y_scale = img_w / 640.0, img_h / 640.0
        final_boxes = []
        for cx, cy, w, h in boxes:

            x1 = cx - w / 2
            y1 = cy - h / 2

            x2 = cx + w / 2
            y2 = cy + h / 2

            x1 = (x1 - dw) / ratio
            y1 = (y1 - dh) / ratio

            x2 = (x2 - dw) / ratio
            y2 = (y2 - dh) / ratio

            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))

            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))

            final_boxes.append([
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            ])
            
        indices = cv2.dnn.NMSBoxes(final_boxes, scores.tolist(), self.conf_thresh, self.iou_thresh)
        
        if len(indices) == 0:
            return DetectionResult(None, None)
            
        nms_xyxy, nms_conf = [], []
        for i in indices.flatten():
            x, y, w, h = final_boxes[i]
            nms_xyxy.append([x, y, x + w, y + h])
            nms_conf.append(scores[i])
            
        return DetectionResult(np.array(nms_xyxy), np.array(nms_conf), np.full(len(nms_xyxy), self.human_class_id))
    def _letterbox(self, img, new_shape=(640, 640)):
        h, w = img.shape[:2]

        r = min(new_shape[0] / h, new_shape[1] / w)

        nw = int(w * r)
        nh = int(h * r)

        resized = cv2.resize(img, (nw, nh))

        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

        dw = (640 - nw) // 2
        dh = (640 - nh) // 2

        canvas[dh:dh+nh, dw:dw+nw] = resized

        return canvas, r, dw, dh
        
    def __del__(self):
        if hasattr(self, 'rknn'):
            try:
                self.rknn.release()
            except: pass


class RKNNYoloTrackerBuilder:
    def __init__(self):
        self.model_path = 'yolov8n.rknn'
        self.confidence = 0.20
        self.iou = 0.55
        self.human_class_id = 0
        self.core_mask = RKNNLite.NPU_CORE_0_1_2

    def set_model_path(self, path):
        self.model_path = path
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

    def build(self) -> RKNNYoloTracker:
        return RKNNYoloTracker(self)
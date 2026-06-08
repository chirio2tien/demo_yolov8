import cv2
import json
import time
import numpy as np
from abc import ABC, abstractmethod


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, frame, context_data):
        pass



class BoundingBoxRenderer(BaseRenderer):

    def __init__(
        self,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.6,
        thickness=2
    ):
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

        np.random.seed(42)
        self.colors = np.random.randint(
            50,
            255,
            (100, 3)
        ).tolist()

    def render(self, frame, context_data):

        detection = context_data.get("results")

        if (
            detection is None
            or detection.xyxy is None
            or len(detection.xyxy) == 0
        ):
            return frame

        

        try:
            for idx in range(len(detection.xyxy)):

                x1, y1, x2, y2 = map(
                    int,
                    detection.xyxy[idx]
                )

                conf = float(
                    detection.confidence[idx]
                )

                track_id = (
                    int(detection.class_ids[idx])
                    if detection.class_ids is not None
                    else 0
                )

                color = tuple(
                    int(c)
                    for c in self.colors[track_id % 100]
                )

                # Clamp tránh box âm
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(0, x2)
                y2 = max(0, y2)

                # Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    self.thickness
                )

                label = f"Human {conf:.1%}"

                (tw, th), _ = cv2.getTextSize(
                    label,
                    self.font,
                    self.font_scale,
                    2
                )

                top_y = max(0, y1 - th - 10)

                cv2.rectangle(
                    frame,
                    (x1, top_y),
                    (x1 + tw + 8, y1),
                    color,
                    -1
                )

                cv2.putText(
                    frame,
                    label,
                    (x1 + 4, y1 - 4),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        except Exception:
            pass

        return frame


class StatusTextRenderer(BaseRenderer):

    def __init__(
        self,
        position=(20, 40),
        text_color=(255, 255, 255),
        bg_color=(20, 20, 20)
    ):
        self.position = position
        self.text_color = text_color
        self.bg_color = bg_color

    def render(self, frame, context_data):

        detection = context_data.get("results")

        num_targets = (
            len(detection.xyxy)
            if (
                detection is not None
                and detection.xyxy is not None
            )
            else 0
        )

        show_fps  = context_data.get("current_fps", 0.0)
        infer_fps = context_data.get("infer_fps")

        if infer_fps is not None:
            text = f"Target: {num_targets} | Show: {show_fps:.0f} | Infer: {infer_fps:.0f}"
        else:
            text = f"Target: {num_targets} | FPS: {show_fps:.1f}"

        x, y = self.position

        (tw, th), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

        cv2.rectangle(
            frame,
            (x - 10, y - th - 12),
            (x + tw + 10, y + 10),
            self.bg_color,
            -1
        )

        cv2.rectangle(
            frame,
            (x - 10, y - th - 12),
            (x + tw + 10, y + 10),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.text_color,
            2,
            cv2.LINE_AA
        )

        return frame

class FrameVisualizer:

    def __init__(self):
        self._renderers = []

    def add_renderer(self, renderer: BaseRenderer):
        self._renderers.append(renderer)
        return self

    def draw(self, frame, context_data):

        for renderer in self._renderers:
            frame = renderer.render(
                frame,
                context_data
            )

        return frame
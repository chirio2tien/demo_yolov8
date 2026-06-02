from src.detector import RKNNYoloTrackerBuilder
from src.visualizer import (
    FrameVisualizer,
    BoundingBoxRenderer,
    StatusTextRenderer
)

from src.pipeline import (
    VideoPipeline,
    AIDetectorStep,
    VisualizationStep,
    DisplayAndSaveStep,
    WebStreamStep,
    TelemetryStep
)

from rknnlite.api import RKNNLite


def main():

    RTSP_URL = "rtsp://192.168.105.18:8554/b8aa7044-6780-798b-d0d6-0e617d223e69_0"
    MODEL_PATH = "yolov8n.rknn"
    OUTPUT_PATH = None
    WEB_PORT = 5000
    try:

        detector = (
            RKNNYoloTrackerBuilder()
            .set_model_path(MODEL_PATH)
            .set_confidence(0.12)
            .set_iou(0.65)
            .set_core_mask(RKNNLite.NPU_CORE_AUTO)
            .build()
        )

        visualizer = (
            FrameVisualizer()
            .add_renderer(BoundingBoxRenderer(font_scale=0.7,thickness=2))
            .add_renderer(StatusTextRenderer())
        )

        pipeline = (
            VideoPipeline()
            .add_step(AIDetectorStep(detector))
            .add_step(TelemetryStep())
            .add_step(VisualizationStep(visualizer))
            .add_step(WebStreamStep(port=WEB_PORT))
            .add_step(DisplayAndSaveStep(
                display=False,
                output_path=OUTPUT_PATH,
                fps=30,
                size=(1920, 1080)
            ))
        )

        print("=" * 60)
        print("YOLOv8n RKNN Human Detection")
        print(f"RTSP : {RTSP_URL}")
        print(f"MODEL: {MODEL_PATH}")
        print(f"WEB  : http://localhost:{WEB_PORT}")
        print("=" * 60)

        pipeline.run(
            RTSP_URL,
            max_retries=5
        )

    except KeyboardInterrupt:
        print("\nStopped by user")

    except Exception as e:
        print(f"\nCritical Error: {e}")

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
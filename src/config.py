# --- Chế độ vận hành ---
ENABLE_WEB_STREAM = False  # True: Flask + display | False: detect-only (test / production)

# --- Capture (giữ chất lượng 6 cam) ---
CAPTURE_MAX_WIDTH = 1280   # 0 = full 2560×1440
CAPTURE_FRAME_SKIP = 3      # 2 ≈ 12 FPS infer/cam

# --- Model pool (6 cam: 3 model) ---
MAX_CAMS_PER_MODEL = 2


def shm_dimensions(capture_max_width: int) -> tuple[int, int]:
    """Kích thước shm (max_h, max_w) — phải khớp giữa main và process con."""
    if capture_max_width > 0:
        max_h = int(1440 * capture_max_width / 2560) + 16
        return max_h, capture_max_width
    return 1440, 2560

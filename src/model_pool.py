from multiprocessing import Queue

from rknnlite.api import RKNNLite

# 416 ~15ms/frame → ~65 infer/s. Mục tiêu ~15 FPS infer/cam → tối đa ~4 cam/model.
# Đặt 2 để mỗi cam infer ổn định; cam thứ 3+ tự load model mới.
MAX_CAMS_PER_MODEL = 2

NPU_CORES = [
    RKNNLite.NPU_CORE_0,
    RKNNLite.NPU_CORE_1,
    RKNNLite.NPU_CORE_2,
]


def build_model_slots(cameras: list, queue_size: int) -> list:
    """
    Gom camera vào từng slot model.
    Đủ chỗ trong slot hiện tại → không tạo model mới.
    Vượt MAX_CAMS_PER_MODEL → tự spawn slot (model instance) tiếp theo.
    """
    slots = []
    for i in range(0, len(cameras), MAX_CAMS_PER_MODEL):
        chunk = cameras[i:i + MAX_CAMS_PER_MODEL]
        slot_id = len(slots)
        slots.append({
            'slot_id':    slot_id,
            'cameras':    chunk,
            'stream_ids': [entry[1] for entry in chunk],
            'infer_in_q': Queue(maxsize=queue_size),
        })

    if len(slots) == 1:
        slots[0]['npu_core'] = RKNNLite.NPU_CORE_0_1_2
        slots[0]['label'] = 'infer-all'
    else:
        if len(slots) > len(NPU_CORES):
            raise ValueError(
                f"Cần {len(slots)} model instance nhưng RK3588 chỉ có {len(NPU_CORES)} NPU core. "
                f"Tăng MAX_CAMS_PER_MODEL hoặc giảm số camera."
            )
        for slot in slots:
            idx = slot['slot_id']
            slot['npu_core'] = NPU_CORES[idx]
            slot['label'] = f"infer-slot{idx}"

    return slots


def cam_to_infer_queue(slots: list) -> dict:
    """Map stream_id → infer_in_q của slot model được gán."""
    mapping = {}
    for slot in slots:
        for entry in slot['cameras']:
            mapping[entry[1]] = slot['infer_in_q']
    return mapping

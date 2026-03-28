from yunchang.ring import (
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_qkvpacked_func,
)

RING_IMPL_DICT = {
    "zigzag": zigzag_ring_flash_attn_func,
}

RING_IMPL_QKVPACKED_DICT = {
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
}

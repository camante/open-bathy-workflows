import numpy as np

from river_masking import apply_estuary_first_channel_domain


def test_apply_estuary_first_channel_domain_removes_estuary_pixels_before_scaffold_generation():
    channel = np.array([[1, 1, 1], [0, 1, 1]], dtype=np.uint8)
    estuary = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    domain = apply_estuary_first_channel_domain(channel, estuary)
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(domain, expected)

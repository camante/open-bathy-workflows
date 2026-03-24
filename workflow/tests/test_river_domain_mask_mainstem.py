import numpy as np

from river_domain_mask import _derive_output_mainstem_mask


def test_output_mainstem_mask_is_confined_to_retained_channel():
    corridor_main = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    channel = np.array(
        [
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    out = _derive_output_mainstem_mask(corridor_main, channel)

    assert np.all(out <= channel)
    assert int(out.sum()) == 4
    assert not out[0, 1]
    assert not out[1, 3]
    assert not out[2, 3]

import numpy as np

from fridge.windowing import RingBuffer, window_indices


def test_window_indices_count():
    sample_rate = 16000
    num_samples = sample_rate * 180
    window_size = sample_rate * 2
    hop_size = int(sample_rate * 0.5)
    indices = window_indices(num_samples, window_size, hop_size)
    assert len(indices) == 357
    assert indices[0] == (0, window_size)
    assert indices[1] == (hop_size, hop_size + window_size)


def test_ring_buffer_basic():
    rb = RingBuffer(4)
    rb.append(np.array([1.0, 2.0], dtype=float))
    assert not rb.filled
    rb.append(np.array([3.0, 4.0], dtype=float))
    assert rb.filled
    assert rb.get().tolist() == [1.0, 2.0, 3.0, 4.0]

    rb.append(np.array([5.0, 6.0], dtype=float))
    assert rb.get().tolist() == [3.0, 4.0, 5.0, 6.0]

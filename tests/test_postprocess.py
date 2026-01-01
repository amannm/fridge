from fridge.postprocess import PostProcessor


def test_postprocess_hysteresis():
    post = PostProcessor(
        ema_alpha=0.5,
        on_threshold=0.7,
        on_frames=3,
        off_threshold=0.3,
        off_frames=6,
    )

    state = None
    for _ in range(3):
        _, state = post.update(1.0)
    assert state is True

    for _ in range(7):
        _, state = post.update(0.0)
    assert state is False

from perception.stacks import choose_level


def test_level_needs_a_good_fit_and_a_clear_margin():
    assert choose_level([0.62, 0.88, 0.70], min_overlap=0.6) == (1, "")      # clearly 2 high
    assert choose_level([0.84, 0.86, 0.60], min_overlap=0.6)[0] is None       # 1 or 2 high: too close
    assert choose_level([0.40, 0.50, 0.45], min_overlap=0.6)[0] is None       # nothing fits

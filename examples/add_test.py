def check(candidate):
    assert candidate(2, 3) == 5
    assert candidate(-1, 1) == 0
    assert candidate(0, 0) == 0
    assert candidate(100, 200) == 300
    assert candidate(-5, -7) == -12

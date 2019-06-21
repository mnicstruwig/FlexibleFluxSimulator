from unified_model.governing_equations import _is_assembly_above_max_height, _get_assembly_max_position


def test_is_assembly_above_max_height_true():
    """Test that the `_is_assembly_above_max_height` function returns true when the
    magnet assembly is above the maximum height

    """
    test_tube_pos = 0
    test_mag_pos = 0.1
    test_assembly_height = 0.025
    test_max_height = 0.113

    assert(_is_assembly_above_max_height(tube_pos=test_tube_pos,
                                         mag_pos=test_mag_pos,
                                         assembly_height=test_assembly_height,
                                         max_height=test_max_height))


def test_get_assembly_max_position():
    """Test that the `_get_assembly_max_position` function returns the correct
    magnet assembly position

    """
    test_tube_pos = 0.1
    test_assembly_height = 0.010
    test_max_height = 0.113

    expected_result = 0.203
    actual_result = _get_assembly_max_position(tube_pos=test_tube_pos,
                                               assembly_height=test_assembly_height,
                                               max_height=test_max_height)
    assert(expected_result == round(actual_result, 4))

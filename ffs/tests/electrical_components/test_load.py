import numpy as np
from unified_model.electrical_components.load import SimpleLoad


def test_simple_load_get_current_closed_circuit():
    """Test the SimpleLoad `get_current` method for the case where the circuit is
    closed.

    """
    test_load = SimpleLoad(R=50)
    test_coil_resistance = 50

    expected_result = 0.5 / 50  # I=V_load/R
    actual_result = test_load.get_current(emf=1, coil_resistance=test_coil_resistance)

    assert expected_result == actual_result


def test_simple_load_get_current_no_load():
    """Test the SimpleLoad `get_current` method for the case where there is no load
    attached.

    """
    test_load = SimpleLoad(R=np.inf)
    test_coil_resistance = 50

    expected_result = 0
    actual_result = test_load.get_current(emf=1, coil_resistance=test_coil_resistance)

    assert expected_result == actual_result


def test_simple_load_get_current_open_circuit_coil():
    """Test the SimpleLoad `get_current` method for the case where the coil is open
    circuit.

    """
    test_load = SimpleLoad(R=50)
    test_coil_resistance = np.inf

    expected_result = 0
    actual_result = test_load.get_current(emf=1, coil_resistance=test_coil_resistance)

    assert expected_result == actual_result

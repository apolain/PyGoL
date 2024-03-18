import os

import numpy as np
import pytest
import yaml

from pygol import game_of_life

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_game_of_life_initialization_with_size():
    test_size = (10, 10)
    gol = game_of_life.GameOfLife(size=test_size)
    assert gol.size == test_size, pytest.fail(
        "The game has not been set to the correct size"
    )
    assert gol.initial_grid.shape == test_size, pytest.fail(
        "The wrong grid has been initialized"
    )
    assert gol.periodic, pytest.fail(
        "The 'periodic' argument has been incorrectly initialised"
    )
    assert gol.born == [3] and gol.survive == [2, 3], pytest.fail(
        "The default configuration for the Game of Life is not B2/S23"
    )


def test_game_of_life_initialization_with_grid():
    initial_grid = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]])
    gol = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=False)
    assert gol.size == initial_grid.shape, pytest.fail(
        "The game has not been set to the correct size"
    )
    assert np.array_equal(gol.initial_grid, initial_grid) and np.array_equal(
        gol.grid, initial_grid
    ), pytest.fail("The wrong grid has been initialized")
    assert not gol.periodic, pytest.fail(
        "The 'periodic' argument has been incorrectly initialised"
    )


def test_game_of_life_initialization_with_yaml():
    #  Create a YAML file containing an initial grid
    yaml_file_path = os.path.join(TESTS_DIR, "test_grid.yaml")
    yaml_data = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file)

    gol = game_of_life.GameOfLife(file_path=yaml_file_path)

    assert gol.size == np.array(yaml_data).shape, pytest.fail(
        "The game has not been set to the correct size"
    )
    assert np.array_equal(gol.initial_grid, np.array(yaml_data)) and np.array_equal(
        gol.grid, np.array(yaml_data)
    ), pytest.fail("The wrong grid has been initialized")
    assert gol.periodic, pytest.fail(
        "The 'periodic' argument has been incorrectly initialised"
    )
    os.remove(yaml_file_path)


def test_invalid_configuration():
    invalid_config = "B3/S2X"
    with pytest.raises(ValueError):
        game_of_life.GameOfLife(configuration=invalid_config)


def test_game_step_periodic():
    initial_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    game = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=True)

    game.step()

    expected_grid = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    assert np.array_equal(game.grid, expected_grid), pytest.fail(
        "The step method is altered"
    )


def test_game_step_non_periodic():
    initial_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    game = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=False)

    game.step()

    expected_grid = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert np.array_equal(game.grid, expected_grid), pytest.fail(
        "The step method is altered"
    )


def test_game_reset():
    initial_grid = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    game = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=False)

    game.step()
    game.reset()

    assert np.array_equal(game.grid, initial_grid), pytest.fail(
        "Resetting the game does not work"
    )


def test_count_number_case_periodic():
    initial_grid = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]])
    game = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=True)

    count_center = game._count_number_case(1, 1)
    assert count_center == 3, pytest.fail("The number of cells counted is incorrect ")

    count_border = game._count_number_case(0, 0)
    assert count_border == 4, pytest.fail("The number of cells counted is incorrect ")

    count_border_midle = game._count_number_case(1, 0)
    assert count_border_midle == 4, pytest.fail(
        "The number of cells counted is incorrect "
    )


def test_count_number_case_non_periodic():
    initial_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    game = game_of_life.GameOfLife(initial_grid=initial_grid, periodic=False)

    count_center = game._count_number_case(1, 1)
    assert count_center == 2, pytest.fail("The number of cells counted is incorrect ")

    count_border = game._count_number_case(0, 0)
    assert count_border == 2, pytest.fail("The number of cells counted is incorrect ")

    count_border_midle = game._count_number_case(1, 0)
    assert count_border_midle == 3, pytest.fail(
        "The number of cells counted is incorrect "
    )


@pytest.mark.parametrize(
    "config_string, expected_result",
    [
        ("B3/S23", True),
        ("B3/S23/E", False),
        ("B1357/S1357", True),
        ("B3/S23/D", False),
        ("B2/S2", True),
        ("B12/O45", False),
        ("B1/S1", True),
        ("C1/S12", False),
        ("B4678/S35678", True),
    ],
)
def test_check_configuration(config_string, expected_result):
    result = game_of_life.GameOfLife.check_configuration(config_string)
    assert result == expected_result, pytest.fail(
        "At least one configuration has been incorrectly checked"
    )

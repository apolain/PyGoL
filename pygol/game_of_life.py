import re
from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
import yaml


class GameOfLife:
    """Class representing Conway's game of life"""

    def __init__(
        self,
        size: Optional[Tuple] = None,
        initial_grid: Optional[np.ndarray] = None,
        file_path: Optional[str] = None,
        configuration: str = "B3/S23",
        periodic: Optional[bool] = True,
    ) -> None:
        """Class constructor

        Parameters
        ----------
        size : Optional[Tuple], optional
            Initial grid size. If the parameter is supplied alone, the initial grid is
            random, by default None
        initial_grid : Optional[np.ndarray], optional
            Initial grid in np.ndarray format of dimension 2, by default None
        file_path : Optional[str], optional
            Path to a YAML file containing an initial grid. If, in addition to
            file_path, the size argument is passed as a parameter, then the initial grid
            of the YAML file will be contained in a grid of size. The dimensions given
            in size must be greater than or equal to those of the initial grid contained
            in the YAML file, by default None
        configuration : str, optional
            The configuration of the game of life to be used, by default "B3/S23"
        periodic : Optional[bool], optional
            Determines behavior at the edge of the grid, by default True
        """
        if not self.check_configuration(configuration):
            raise ValueError("Invalid game of life configuration")

        if size is None and initial_grid is None and file_path is None:
            raise ValueError(
                "At least one of the arguments (size, inital_grid, file_path) must be passed as a parameter"
            )

        if size is not None and not (isinstance(size, Tuple) and len(size) == 2):
            raise ValueError("The size argument must be a tuple of dimension 2")

        if initial_grid is not None:
            self.initial_grid = initial_grid

        elif file_path is not None:
            self.initial_grid = self.read_yaml(file_path, size=size)

        else:
            self.initial_grid = self.get_random_init_config(size)

        self.size = self.initial_grid.shape

        self.reset()
        self.periodic = periodic

        self.born, self.survive = self.parse_configuration(configuration)

    def step(self) -> None:
        """Function that moves the game forward one step"""
        new_grid = np.zeros((self.size[0], self.size[1]))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                nb_voisins = self._count_number_case(i, j)
                if self.grid[i, j] == 0 and nb_voisins in self.born:
                    new_grid[i, j] = 1
                elif self.grid[i, j] == 1 and nb_voisins in self.survive:
                    new_grid[i, j] = 1

                else:
                    new_grid[i, j] = 0

        self.grid = new_grid

    def reset(self) -> None:
        """Resets the game of life to the initial grid"""
        self.grid = self.initial_grid

    def _count_number_case(self, i: int, j: int) -> int:
        """Count the number of living cells around a grid cell

        Parameters
        ----------
        i : int
            Line index
        j : int
            Row index

        Returns
        -------
        int
            Number of living cells
        """
        live_neighbors = 0
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                if self.periodic:
                    live_neighbors += self.grid[x % self.size[0], y % self.size[1]]
                else:
                    if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
                        live_neighbors += self.grid[x, y]
        live_neighbors -= self.grid[i, j]
        return live_neighbors

    def get_random_init_config(
        self, size: Tuple[int], p: Optional[Number] = 1 / 5
    ) -> np.ndarray:
        """Generates a random initial configuration for the grid

        Parameters
        ----------
        size : Tuple[int]
            Grid size to be generated
        p : Optional[Number], optional
            Probability of a cell being alive, by default 1/5

        Returns
        -------
        np.ndarray
            Random initial matrix
        """
        return np.random.binomial(1, p, size=size)

    @staticmethod
    def check_configuration(config_string: str) -> bool:
        """Checks the validity of the configuration passed in parameter

        Parameters
        ----------
        string : str
            Configuration

        Returns
        -------
        bool
            Boolean indicating whether the configuration is valid
        """
        pattern = r"^B\d+\/S\d+$"
        return bool(re.match(pattern, config_string))

    @staticmethod
    def parse_configuration(config_string: str) -> Tuple[int, int]:
        """Parses the configuration of the set of life passed in parameter

        Parameters
        ----------
        config_string : str
            Configuration

        Returns
        -------
        Tuple[int, int]
            A tuple containing the number of neighbors for birth and survival
        """

        def split_number(number) -> List[int]:
            return [int(chiffre) for chiffre in str(number)]

        regex_b = r"B(\d+)"
        regex_s = r"S(\d+)"

        figures_b = []
        figures_s = []

        matches_b = re.findall(regex_b, config_string)
        matches_s = re.findall(regex_s, config_string)

        if matches_b:
            figures_b.extend(split_number(matches_b[0]))
        if matches_s:
            figures_s.extend(split_number(matches_s[0]))

        return figures_b, figures_s

    def to_yaml(self, file_path: str, smaller_possible: Optional[bool] = True) -> None:
        """Extracts current grid status in YAML format

        Parameters
        ----------
        file_path : str
            Path in which to save the grid
        smaller_possible : Optional[bool], optional
            Determines whether the saved figure should be as small as possible, by
            default True
        """
        res_matrix = self.grid

        if smaller_possible:
            non_zero_indices = np.nonzero(self.grid)
            min_row, max_row = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            min_col, max_col = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])

            res_matrix = self.grid[min_row : max_row + 1, min_col : max_col + 1]

        with open(file_path, "w") as file:
            yaml.dump(res_matrix.tolist(), file)

    @staticmethod
    def read_yaml(file_path: str, size=None) -> np.ndarray:
        """Method for reading the YAML file for the initial grid

        Parameters
        ----------
        file_path : str
            Path to the YAML file
        size : _type_, optional
            Size of the matrix into which the read file is integrated, by default None

        Returns
        -------
        np.ndarray
            Returns the matrix containing the initial grid

        Raises
        ------
        ValueError
            Returns an error if the matrix read cannot be integrated into a matrix of
            size "size"
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        data_rows, data_cols = len(data), len(data[0])
        if size is None:
            rows, cols = np.maximum(data_rows, data_cols), np.maximum(
                data_rows, data_cols
            )
        else:
            rows, cols = size

        if rows < data_rows or cols < data_cols:
            raise ValueError(
                "Number of rows or columns specified too small to contain the YAML read"
            )

        result = np.zeros((rows, cols), dtype=int)

        start_row = (rows - data_rows) // 2
        start_col = (cols - data_cols) // 2

        result[start_row : start_row + data_rows, start_col : start_col + data_cols] = (
            np.array(data)
        )

        return result

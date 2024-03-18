import multiprocessing
from numbers import Number
from typing import Optional

import numpy as np
import pygame
import pygame.locals

from . import game_of_life


class PygameGOLDisplayer:
    """Class displaying the game of life via Pygame"""

    def __init__(
        self, gol: game_of_life.GameOfLife, height: Number = 720, width: Number = 1280
    ) -> None:
        """Class constructor

        Parameters
        ----------
        gol : game_of_life.GameOfLife
            Instance of the game of life to display
        height : Number, optional
            Window height, by default 720
        width : Number, optional
            Window width, by default 1280
        """
        self.gol = gol
        self.process = None
        self.running = False

        self.cell_size = np.minimum(height / self.gol.size[0], width / self.gol.size[1])
        self.height, self.width = (
            self.gol.size[0] * self.cell_size,
            self.gol.size[1] * self.cell_size,
        )

    def run(self) -> None:
        """Method for starting the game display"""
        self.running = True
        self.process = multiprocessing.Process(target=self._display_game)
        self.process.start()

    def stop(self) -> None:
        """Method for stopping the display"""
        if self.process:
            self.process.terminate()

        self.running = False

    def _init_pygame(self) -> None:
        """Pygame window initialisation method"""
        pygame.init()
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Game of life")

    def _handle_events(self) -> None:
        """Event management method (here, only closing the Pygame window)"""
        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                self.running = False

    def _draw_grid(self) -> None:
        """Method for drawing the grid"""
        self.window.fill((0, 0, 0))
        for i, row in enumerate(self.gol.grid):
            for j, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(
                        self.window,
                        (255, 255, 255),
                        (
                            j * self.cell_size,
                            i * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )

    def _display_game(self, wait_time: Optional[Number] = 100) -> None:
        """Method for displaying the game

        Parameters
        ----------
        wait_time : Optional[Number], optional
            Waiting time between frames, by default 100
        """
        self._init_pygame()

        while self.running:
            self._handle_events()

            self._draw_grid()

            self.gol.step()
            pygame.display.flip()
            pygame.time.wait(wait_time)

        pygame.quit()

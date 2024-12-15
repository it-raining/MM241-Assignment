import random

import gymnasium as gym
import matplotlib as mpl
import numpy as np
import pygame
from gymnasium import spaces
from matplotlib import colormaps

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class CuttingStockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(
        self,
        render_mode=None,
        min_w=50,
        min_h=50,
        max_w=100,
        max_h=100,
        num_stocks=100,
        max_product_type=25,
        max_product_per_type=20,
        seed=42,
    ):
        self.seed = seed
        self.min_w = min_w
        self.min_h = min_h
        self.max_w = max_w
        self.max_h = max_h
        self.num_stocks = num_stocks
        self.max_product_type = max_product_type
        self.max_product_per_type = max_product_per_type
        self.cutted_stocks = np.full((num_stocks,), fill_value=0, dtype=int)

        # Set seed explicitly
        set_seed(seed)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "stocks": spaces.Tuple(
                    [spaces.MultiDiscrete(np.full((max_w, max_h), max_product_type + 2, dtype=int))] * num_stocks
                ),
                "products": spaces.Sequence(
                    spaces.Dict(
                        {
                            "size": spaces.MultiDiscrete(
                                np.array([max_w, max_h]), start=np.array([1, 1])
                            ),
                            "quantity": spaces.Discrete(
                                max_product_per_type + 1, start=0
                            ),
                        }
                    )
                ),
            }
        )

        # Define action space
        self.action_space = spaces.Dict(
            {
                "stock_idx": spaces.Discrete(num_stocks),
                "size": spaces.Box(
                    low=np.array([1, 1]),
                    high=np.array([max_w, max_h]),
                    shape=(2,),
                    dtype=int,
                ),
                "position": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([max_w - 1, max_h - 1]),
                    shape=(2,),
                    dtype=int,
                ),
            }
        )

        # Initialize empty stocks and products
        self._stocks = []
        self._products = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Rendering variables
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"stocks": self._stocks, "products": self._products}

    def _get_info(self):
        filled_ratio = np.mean(self.cutted_stocks).item()
        trim_loss = []

        for sid, stock in enumerate(self._stocks):
            if self.cutted_stocks[sid] == 0:
                continue
            tl = (stock == -1).sum() / (stock != -2).sum()
            trim_loss.append(tl)

        trim_loss = np.mean(trim_loss).item() if trim_loss else 1

        return {"filled_ratio": filled_ratio, "trim_loss": trim_loss}

    def reset(self, seed=42, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        set_seed(seed)

        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)

        # Hardcoded stocks with quantities
        stock_configurations = {
            (60, 90): 10,
            (70, 50): 20,
            (80, 60): 15,
            (95, 95): 4,
        }

        self._stocks = []
        for (width, height), quantity in stock_configurations.items():
            for _ in range(quantity):
                stock = np.pad(
                    np.full((width, height), -1, dtype=int),
                    ((0, self.max_w - width), (0, self.max_h - height)),
                    constant_values=-2
                )
                self._stocks.append(stock)
        self._stocks = tuple(self._stocks)

        # Hardcoded products
        self._products = [
            {"size": np.array([20, 10]), "quantity": 100},
            {"size": np.array([25, 25]), "quantity": 30},
            {"size": np.array([10, 60]), "quantity": 20},
            {"size": np.array([70, 70]), "quantity": 2},
        ]
        self._products = tuple(self._products)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        # Check if the product is in the product list
        product_idx = None
        for i, product in enumerate(self._products):
            if np.array_equal(product["size"], size) or np.array_equal(
                product["size"], size[::-1]
            ):
                if product["quantity"] == 0:
                    continue

                product_idx = i  # Product index starts from 0
                break

        if product_idx is not None:
            if 0 <= stock_idx < self.num_stocks:
                stock = self._stocks[stock_idx]
                # Check if the product fits in the stock
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                if (
                    x >= 0
                    and y >= 0
                    and x + width <= stock_width
                    and y + height <= stock_height
                ):
                    # Check if the position is empty
                    if np.all(stock[x : x + width, y : y + height] == -1):
                        self.cutted_stocks[stock_idx] = 1
                        stock[x : x + width, y : y + height] = product_idx
                        self._products[product_idx]["quantity"] -= 1

        # An episode is done iff all product quantities are 0
        terminated = all([product["quantity"] == 0 for product in self._products])
        reward = 1 if terminated else 0  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    def _get_window_size(self):
        width = int(np.ceil(np.sqrt(self.num_stocks)))
        height = int(np.ceil(self.num_stocks / width))
        return width * self.max_w, height * self.max_h
    def _render_frame(self):
        window_size = self._get_window_size()
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Cutting Stock Environment")
            self.window = pygame.display.set_mode((window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_size)
        canvas.fill((0, 0, 0))
        pix_square_size = 1  # The size of a single grid square in pixels

        # Create a colormap for the products
        cmap = colormaps.get_cmap("hsv")
        norms = mpl.colors.Normalize(vmin=0, vmax=self.max_product_type - 1)
        list_colors = [cmap(norms(i)) for i in range(self.max_product_type)]

        # First we draw the stocks with the products
        for i, stock in enumerate(self._stocks):
            # Compute the actual stock width and height
            # Outside of the stock, we have empty cells (-2)
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))

            # Fill the stocks wuth grey color
            pygame.draw.rect(
                canvas,
                (128, 128, 128),
                pygame.Rect(
                    (i % (window_size[0] // self.max_w) * self.max_w) * pix_square_size,
                    (i // (window_size[0] // self.max_w) * self.max_h)
                    * pix_square_size,
                    stock_width * pix_square_size,
                    stock_height * pix_square_size,
                ),
            )

            for x in range(stock.shape[0]):
                for y in range(stock.shape[1]):
                    if stock[x, y] > -1:
                        color = list_colors[stock[x, y]][:3]
                        color = (
                            int(color[0] * 255),
                            int(color[1] * 255),
                            int(color[2] * 255),
                        )
                        pygame.draw.rect(
                            canvas,
                            color,
                            pygame.Rect(
                                (i % (window_size[0] // self.max_w) * self.max_w + x)
                                * pix_square_size,
                                (i // (window_size[0] // self.max_w) * self.max_h + y)
                                * pix_square_size,
                                pix_square_size,
                                pix_square_size,
                            ),
                        )

        # Finally, add horizontal and vertical gridlines
        for i in range(window_size[0] // self.max_w):
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (i * self.max_w * pix_square_size, 0),
                (i * self.max_w * pix_square_size, window_size[1]),
            )
        for i in range(window_size[1] // self.max_h):
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (0, i * self.max_h * pix_square_size),
                (window_size[0], i * self.max_h * pix_square_size),
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.font.quit()

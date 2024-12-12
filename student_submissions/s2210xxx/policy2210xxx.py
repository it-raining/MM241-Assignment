import numpy as np
from policy import Policy


class Policy2210xxx(Policy):
    policy = 0
    stock_sort = None
    observation_old = None
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2
    # def _sort_stocks(self, stocks):
    #     stock_info = []
    #     for idx, stock in enumerate(stocks):
    #         stock_w, stock_h = self._get_stock_size_(stock)
    #         area = stock_w * stock_height
    #         perimeter = 2 * (stock_w + stock_height)
    #         stock_info.append((idx, stock, area, perimeter))

    #     # Sort by decreasing area and increasing perimeter
    #     sorted_stock_info = sorted(stock_info, key=lambda x: (-x[2], x[3]))
    #     return [(info[0], info[1]) for info in sorted_stock_info]
    def get_action(self, observation, info):
        # if self.observation_old != observation:
        #     self.stock_sort = self._sort_stocks(observation["stocks"])
        # Student code here
        match self.policy:
            case 1:
                list_prods = observation["products"]

                prod_size = [0, 0]
                stock_idx = -1
                pos_x, pos_y = None, None
                best_pos = None
                # Pick a product that has quality > 0
                

                        # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    for prod in list_prods:
                        if prod["quantity"] > 0:
                            prod_size = prod["size"]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size

                            # Try placing the product in the bottom-left-most position
                            if stock_w >= prod_w and stock_h >= prod_h:
                                for y in range(stock_h - prod_h + 1):
                                    for x in range(stock_w - prod_w + 1):
                                        if self._can_place_(stock, (x, y), prod_size):
                                            if best_pos is None:
                                                best_pos = (x, y)
                                                break
                                    if best_pos is not None:
                                        break
                            # Check rotated orientation
                            if stock_w >= prod_h and stock_h >= prod_w:
                                for y in range(stock_h - prod_w + 1):
                                    for x in range(stock_w - prod_h + 1):
                                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                                            if best_pos is None or y < best_pos[1] or x < best_pos[0]:
                                                best_pos = (x, y)
                                                prod_size = prod_size[::-1]
                                                break
                                    if best_pos is not None:
                                        break
                            if best_pos is not None:
                                pos_x, pos_y = best_pos
                                stock_idx = i
                                break

                    if pos_x is not None and pos_y is not None:
                        break

                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            # case 2
    # Student code here
    # You can add more functions if needed

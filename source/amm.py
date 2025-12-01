class UniswapAMM:
    def __init__(self, token_x, token_y, reserve_x, reserve_y):
        self.token_x = token_x  # es. "ETH"
        self.token_y = token_y  # es. "USDC"
        self.x = reserve_x      # es. 100 ETH
        self.y = reserve_y      # es. 100_000 USDC

    def get_price_x_to_y(self):
        return self.y / self.x

    def get_price_y_to_x(self):
        return self.x / self.y

    def swap_x_for_y(self, dx):
        # Senza fee
        dy = self.y - (self.x * self.y) / (self.x + dx)
        self.x += dx
        self.y -= dy
        return dy

    def swap_y_for_x(self, dy):
        dx = self.x - (self.x * self.y) / (self.y + dy)
        self.y += dy
        self.x -= dx
        return dx

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


    def swap_x_for_y_with_fee(self, dx, fee=0.003):
        price_before = self.get_price_x_to_y()
        dx_fee = dx * (1 - fee)
        dy = self.y - (self.x * self.y) / (self.x + dx_fee)
        self.x += dx_fee
        self.y -= dy
        price_after = self.get_price_x_to_y()
        effective_price = dy / dx
        slippage = ((effective_price - price_before) / price_before) * 100

        print(f"[SWAP] {dx} {self.token_x} → {dy:.2f} {self.token_y}")
        print(f"  • Effective price: {effective_price:.2f} {self.token_y}/{self.token_x}")
        print(f"  • Price before: {price_before:.2f}, after: {price_after:.2f}")
        print(f"  • Slippage: {slippage:.2f}%")
        print(f"  • Reserves → {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")

        return dy


    def swap_y_for_x_with_fee(self, dy, fee=0.003):
        price_before = self.get_price_y_to_x()
        dy_fee = dy * (1 - fee)
        dx = self.x - (self.x * self.y) / (self.y + dy_fee)
        self.y += dy_fee
        self.x -= dx
        price_after = self.get_price_y_to_x()
        effective_price = dx / dy
        slippage = ((effective_price - price_before) / price_before) * 100

        print(f"[SWAP] {dy} {self.token_y} → {dx:.4f} {self.token_x}")
        print(f"  • Effective price: {effective_price:.4f} {self.token_x}/{self.token_y}")
        print(f"  • Price before: {price_before:.4f}, after: {price_after:.4f}")
        print(f"  • Slippage: {slippage:.2f}%")
        print(f"  • Reserves → {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")

        return dx

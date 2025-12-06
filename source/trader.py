class Trader:
    def __init__(self, name, token, amount):
        self.name = name
        self.token = token
        self.amount = amount

    def act(self, amm):
        if self.token == amm.token_x:
            print(f"\n[Trader {self.name}] swap {self.amount} {amm.token_x} → {amm.token_y}")
            amm.swap_x_for_y_with_fee(self.amount)
        else:
            print(f"\n[Trader {self.name}] swap {self.amount} {amm.token_y} → {amm.token_x}")
            amm.swap_y_for_x_with_fee(self.amount)

    def __str__(self):
        return f"{self.name} – {self.amount:.2f} {self.token}"
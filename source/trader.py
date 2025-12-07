import random
class Trader:
    def __init__(self, name, eth_amount=0.0, usdc_amount=0.0):
        self.name = name
        self.wallet = {
            "ETH": eth_amount,
            "USDC": usdc_amount
        }

    def __str__(self):
        eth_str = f"{self.wallet['ETH']:.4f}".rstrip("0").rstrip(".")
        usdc_str = f"{self.wallet['USDC']:.2f}"
        return f"{self.name} – {eth_str} ETH | {usdc_str} USDC"

    def can_afford(self, token, amount):
        return self.wallet[token] >= amount

    def deduct(self, token, amount):
        self.wallet[token] -= amount

    def add(self, token, amount):
        self.wallet[token] += amount

    def act(self, amm, strategy="random"):
            """
            Il trader decide in modo casuale se fare uno swap e in quale direzione
            """
            action = random.choice(["buy", "sell", "hold"])

            if action == "hold":
                print(f"[{self.name}] Passa il turno.")
                return

            if action == "buy" and self.wallet["USDC"] >= 50:
                usdc_in = random.uniform(50, min(self.wallet["USDC"], 300))
                eth_out = amm.swap_y_for_x_with_fee(usdc_in)
                self.deduct("USDC", usdc_in)
                self.add("ETH", eth_out)
                print(f"[{self.name}] Compra ETH: spende {usdc_in:.2f} USDC → ottiene {eth_out:.4f} ETH")

            elif action == "sell" and self.wallet["ETH"] >= 0.05:
                eth_in = random.uniform(0.05, min(self.wallet["ETH"], 0.5))
                usdc_out = amm.swap_x_for_y_with_fee(eth_in)
                self.deduct("ETH", eth_in)
                self.add("USDC", usdc_out)
                print(f"[{self.name}] Vende ETH: scambia {eth_in:.4f} ETH → ottiene {usdc_out:.2f} USDC")
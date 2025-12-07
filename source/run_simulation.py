import random
from amm import UniswapAMM
from trader import Trader
from agents import Arbitrageur, PanicLP

class Simulation:
    def __init__(self):
        self.amm = UniswapAMM("ETH", "USDC", 100, 200000)
        self.market_price = 2000
        self.traders = [
            Trader("Alice", eth_amount=5, usdc_amount=1000),
            Trader("Bob", eth_amount=2, usdc_amount=3000)
        ]
        self.arb = Arbitrageur("ArbBot")
        self.lp = PanicLP("BigWhale", 0.5, self.amm)
        self.price_log = []

    def print_status(self):
        print(f"\n🌐 Market price: {self.market_price:.2f} USDC/ETH")
        print(f"🧮 AMM price:     {self.amm.get_price_x_to_y():.2f} USDC/ETH")
        print(f"🏦 Pool: {self.amm.x:.2f} ETH | {self.amm.y:.2f} USDC")
        print("👥 Traders:")
        for t in self.traders:
            print(f"  - {t.name}: {t.wallet['ETH']:.2f} ETH | {t.wallet['USDC']:.2f} USDC")

    def apply_market_shock(self, percent):
        self.market_price *= (1 + percent)
        print(f"\n⚡ MERCATO SHOCKATO! Nuovo prezzo: {self.market_price:.2f} USDC/ETH")

    def step(self, step_num):
        print(f"\n=== TICK {step_num} ===")

        # Volatilità casuale (puoi disattivarla)
        change_pct = random.uniform(-0.02, 0.02)
        self.market_price *= (1 + change_pct)
        volatility = abs(change_pct)

        # Arbitraggio
        self.arb.act(self.amm, self.market_price)

        # Trader casuali
        for t in self.traders:
            t.act(self.amm, strategy="random")

        # LP controlla volatilità
        self.lp.check_stress(self.amm, volatility)

        self.price_log.append({
            "amm": self.amm.get_price_x_to_y(),
            "market": self.market_price
        })

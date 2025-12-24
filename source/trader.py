# Il file implementa un trader retail con comportamente casuale, che può:
# - Comprare ETH con USDC
# - Vendere ETH per USDC
# - Restare fermo ("hold")

import random

class Trader:
    def __init__(self, name, eth_amount=0.0, usdc_amount=0.0, 
                 min_trade_usdc=50, max_trade_usdc=300,
                 min_trade_eth=0.05, max_trade_eth=0.5,
                 hold_probability=0.3):
        """
        Args:
            name: Trader's name
            eth_amount: Initial ETH
            usdc_amount: Initial USDC
            min_trade_usdc: Minimum USDC per trade
            max_trade_usdc: Maximum USDC per trade
            min_trade_eth: Minimum ETH per trade
            max_trade_eth: Maximum ETH per trade
            hold_probability: Probability of not trading (0-1)
        """
        self.name = name
        self.wallet = {
            "ETH": float(eth_amount),
            "USDC": float(usdc_amount)
        }
        self.initial_wallet = self.wallet.copy()
        
        # Trading parameters
        self.min_trade_usdc = min_trade_usdc
        self.max_trade_usdc = max_trade_usdc
        self.min_trade_eth = min_trade_eth
        self.max_trade_eth = max_trade_eth
        self.hold_probability = hold_probability
        
        # Statistics
        self.trades_executed = 0
        self.trades_buy = 0
        self.trades_sell = 0
        self.total_volume_eth = 0.0
        self.total_volume_usdc = 0.0
        self.failed_trades = 0

    def __str__(self):
        eth_str = f"{self.wallet['ETH']:.4f}".rstrip("0").rstrip(".")
        usdc_str = f"{self.wallet['USDC']:.2f}"
        return f"{self.name} – {eth_str} ETH | {usdc_str} USDC"

    def can_afford(self, token, amount):
        """Checks if the trader has enough funds"""
        return self.wallet.get(token, 0) >= amount

    def deduct(self, token, amount):
        """Removes tokens from the wallet"""
        if not self.can_afford(token, amount):
            raise ValueError(f"{self.name} doesn't have enough {token}")
        self.wallet[token] -= amount

    def add(self, token, amount):
        """Adds tokens to the wallet"""
        self.wallet[token] += amount

    def get_portfolio_value(self, eth_price):
        """Calculates total portfolio value in USDC"""
        return self.wallet["ETH"] * eth_price + self.wallet["USDC"]

    def get_pnl(self, eth_price):
        """Calculates profit/loss relative to initial value"""
        current_value = self.get_portfolio_value(eth_price)
        initial_value = self.initial_wallet["ETH"] * eth_price + self.initial_wallet["USDC"]
        return current_value - initial_value

    def act(self, amm, strategy="random", verbose=True):
        """
        Executes a trading action.
        
        Args:
            amm: The AMM to trade on
            strategy: Trading strategy ("random", "momentum", "contrarian")
            verbose: If True, prints logs
        """
        # Chance to do nothing
        if random.random() < self.hold_probability:
            if verbose:
                print(f"💤 [{self.name}] Hold")
            return

        # Determine action based on strategy
        if strategy == "random":
            action = random.choice(["buy", "sell"])
        elif strategy == "momentum":
            # Buy if price is rising (simplified)
            action = "buy" if random.random() < 0.6 else "sell"
        elif strategy == "contrarian":
            # Sell if price is rising (simplified)
            action = "sell" if random.random() < 0.6 else "buy"
        else:
            action = random.choice(["buy", "sell"])

        try:
            if action == "buy":
                self._try_buy(amm, verbose)
            elif action == "sell":
                self._try_sell(amm, verbose)
        except Exception as e:
            self.failed_trades += 1
            if verbose:
                print(f"[{self.name}] Trade failed: {e}")

    def _try_buy(self, amm, verbose):
        """Attempts to buy ETH with USDC"""
        if not self.can_afford("USDC", self.min_trade_usdc):
            if verbose:
                print(f"[{self.name}] Insufficient USDC to buy")
            return

        # Calculate how much USDC to spend
        max_spend = min(self.wallet["USDC"], self.max_trade_usdc)
        usdc_in = random.uniform(self.min_trade_usdc, max_spend)

        # Execute swap
        eth_out = amm.swap_y_for_x_with_fee(usdc_in, verbose=False)
        
        # Update wallet
        self.deduct("USDC", usdc_in)
        self.add("ETH", eth_out)
        
        # Update statistics
        self.trades_executed += 1
        self.trades_buy += 1
        self.total_volume_usdc += usdc_in
        self.total_volume_eth += eth_out
        
        if verbose:
            print(f"🟢 [{self.name}] BUY: {usdc_in:.2f} USDC → {eth_out:.4f} ETH")

    def _try_sell(self, amm, verbose):
        """Attempts to sell ETH for USDC"""
        if not self.can_afford("ETH", self.min_trade_eth):
            if verbose:
                print(f"[{self.name}] Insufficient ETH to sell")
            return

        # Calculate how much ETH to sell
        max_sell = min(self.wallet["ETH"], self.max_trade_eth)
        eth_in = random.uniform(self.min_trade_eth, max_sell)

        # Execute swap
        usdc_out = amm.swap_x_for_y_with_fee(eth_in, verbose=False)
        
        # Update wallet
        self.deduct("ETH", eth_in)
        self.add("USDC", usdc_out)
        
        # Update statistics
        self.trades_executed += 1
        self.trades_sell += 1
        self.total_volume_eth += eth_in
        self.total_volume_usdc += usdc_out
        
        if verbose:
            print(f"🔴 [{self.name}] SELL: {eth_in:.4f} ETH → {usdc_out:.2f} USDC")

    def get_stats(self):
        """Returns trader statistics"""
        return {
            "name": self.name,
            "wallet": self.wallet.copy(),
            "initial_wallet": self.initial_wallet.copy(),
            "trades_executed": self.trades_executed,
            "trades_buy": self.trades_buy,
            "trades_sell": self.trades_sell,
            "total_volume_eth": self.total_volume_eth,
            "total_volume_usdc": self.total_volume_usdc,
            "failed_trades": self.failed_trades
        }

    def reset(self):
        """Resets the trader to initial state"""
        self.wallet = self.initial_wallet.copy()
        self.trades_executed = 0
        self.trades_buy = 0
        self.trades_sell = 0
        self.total_volume_eth = 0.0
        self.total_volume_usdc = 0.0
        self.failed_trades = 0


# Additional class: More sophisticated trader
class SmartTrader(Trader):
    """Trader that considers slippage and price impact before trading"""
    
    def __init__(self, name, eth_amount=0.0, usdc_amount=0.0, max_slippage=0.02):
        super().__init__(name, eth_amount, usdc_amount)
        self.max_slippage = max_slippage  # 2% max tolerated slippage
    
    def _try_buy(self, amm, verbose):
        """Buys only if slippage is acceptable (FIXED VERSION)"""
        if not self.can_afford("USDC", self.min_trade_usdc):
            return
        
        max_spend = min(self.wallet["USDC"], self.max_trade_usdc)
        usdc_in = random.uniform(self.min_trade_usdc, max_spend)
                
        # 1. Calcola quanto ETH riceverai realmente
        actual_eth_out = amm.calculate_swap_output(usdc_in, amm.y, amm.x)
        
        # 2. Calcola il prezzo effettivo dello swap
        effective_price = usdc_in / actual_eth_out  # USDC per 1 ETH effettivo
        
        # 3. Calcola il prezzo spot di mercato (riferimento)
        spot_price = amm.get_price_y_to_x()  # USDC per 1 ETH prima dello swap
        
        # 4. Slippage = differenza tra prezzo effettivo e prezzo spot
        slippage = abs(effective_price - spot_price) / spot_price
        
        if slippage > self.max_slippage:
            if verbose:
                print(f"[{self.name}] Slippage too high ({slippage:.2%}), skip buy")
            return
    
        # Proceed with the trade
        super()._try_buy(amm, verbose)

    def _try_sell(self, amm, verbose):
        """Sells only if slippage is acceptable (FIXED VERSION)"""
        if not self.can_afford("ETH", self.min_trade_eth):
            return
        
        max_sell = min(self.wallet["ETH"], self.max_trade_eth)
        eth_in = random.uniform(self.min_trade_eth, max_sell)
                
        # 1. Calcola quanto USDC riceverai realmente
        actual_usdc_out = amm.calculate_swap_output(eth_in, amm.x, amm.y)
        
        # 2. Calcola il prezzo effettivo dello swap
        effective_price = actual_usdc_out / eth_in  # USDC per 1 ETH effettivo
        
        # 3. Calcola il prezzo spot di mercato
        spot_price = amm.get_price_x_to_y()  # USDC per 1 ETH prima dello swap
        
        # 4. Slippage = differenza tra prezzo effettivo e prezzo spot
        slippage = abs(effective_price - spot_price) / spot_price
        
        if slippage > self.max_slippage:
            if verbose:
                print(f"[{self.name}] Slippage too high ({slippage:.2%}), skip sell")
            return
        
        # Proceed with the trade
        super()._try_sell(amm, verbose)
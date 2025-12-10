# Il file implementa un Constant Product Market Maker (CPMM) tipo Uniswap V2 con:
# - Gestione liquidità (add/remove)
# - Swap con fee (0.3% default standard)
# - Calcolo prezzi e slippage
# - LP tokens basati su media geometrica

import math

class UniswapAMM:
    def __init__(self, token_x, token_y, reserve_x, reserve_y, fee=0.003):
        """
        Initializes a Constant Product AMM (x * y = k).
        
        Args:
            token_x: Name of token X (e.g. "ETH")
            token_y: Name of token Y (e.g. "USDC")
            reserve_x: Initial reserve of X
            reserve_y: Initial reserve of Y
            fee: Swap fee (default 0.3%)
        """
        if reserve_x <= 0 or reserve_y <= 0:
            raise ValueError("Initial reserves must be positive")
        if not 0 <= fee < 1:
            raise ValueError("Fee must be between 0 and 1")
            
        self.token_x = token_x
        self.token_y = token_y
        self.x = float(reserve_x)
        self.y = float(reserve_y)
        self.fee = fee
        
        # LP tokens = sqrt(x * y) like Uniswap V2
        self.total_supply = math.sqrt(reserve_x * reserve_y)
        
        # Tracking for metrics
        self.total_fee_x = 0.0
        self.total_fee_y = 0.0
        self.swap_count = 0
        self.k_history = [self.x * self.y]

    def get_k(self):
        """Returns the constant product k"""
        return self.x * self.y

    def get_price_x_to_y(self):
        """Price of X in terms of Y (how many Y for 1 X)"""
        if self.x <= 0:
            raise ValueError("Reserve X is zero or negative")
        return self.y / self.x

    def get_price_y_to_x(self):
        """Price of Y in terms of X (how many X for 1 Y)"""
        if self.y <= 0:
            raise ValueError("Reserve Y is zero or negative")
        return self.x / self.y

    def add_liquidity(self, amount_x, amount_y):
        """
        Adds liquidity to the pool.
        Returns the number of LP tokens minted.
        """
        if amount_x <= 0 or amount_y <= 0:
            raise ValueError("Amounts must be positive")
        
        if self.total_supply == 0:
            # First liquidity: mint sqrt(x*y)
            liquidity = math.sqrt(amount_x * amount_y)
            if liquidity <= 0:
                raise ValueError("Initial liquidity too low")
        else:
            # Maintain current ratio
            ratio_x = amount_x / self.x
            ratio_y = amount_y / self.y
            
            # Use the smaller ratio to avoid price manipulation
            liquidity = min(
                amount_x * self.total_supply / self.x,
                amount_y * self.total_supply / self.y
            )
            
            # Warn if ratios don't match (user is losing value)
            if abs(ratio_x - ratio_y) > 0.01:  # 1% tolerance
                print(f"[AMM] Warning: Non-optimal ratio! "
                      f"Ratio X: {ratio_x:.4f}, Ratio Y: {ratio_y:.4f}")
        
        self.x += amount_x
        self.y += amount_y
        self.total_supply += liquidity
        self.k_history.append(self.get_k())
        
        return liquidity

    def remove_liquidity(self, liquidity):
        """
        Removes liquidity by burning LP tokens.
        Returns (amount_x, amount_y).
        """
        if liquidity <= 0:
            raise ValueError("Liquidity to remove must be positive")
        if liquidity > self.total_supply:
            raise ValueError(f"Insufficient liquidity: {liquidity} > {self.total_supply}")
        if self.total_supply == 0:
            return 0, 0
        
        # Calculate proportional shares
        amount_x = (liquidity * self.x) / self.total_supply
        amount_y = (liquidity * self.y) / self.total_supply
        
        # Verify we're not completely emptying the pool
        if amount_x >= self.x or amount_y >= self.y:
            raise ValueError("Cannot remove all liquidity")
        
        self.x -= amount_x
        self.y -= amount_y
        self.total_supply -= liquidity
        self.k_history.append(self.get_k())
        
        return amount_x, amount_y

    def calculate_swap_output(self, amount_in, reserve_in, reserve_out, fee=None):
        """
        Calculates swap output using the CPMM formula.
        Formula: dy = y - (x*y)/(x + dx*(1-fee))
        """
        if fee is None:
            fee = self.fee
            
        if amount_in <= 0:
            raise ValueError("Input amount must be positive")
        if reserve_in <= 0 or reserve_out <= 0:
            raise ValueError("Reserves must be positive")
        
        # Apply fee to input
        amount_in_with_fee = amount_in * (1 - fee)
        
        # CPMM formula: amount_out = reserve_out - (reserve_in * reserve_out) / (reserve_in + amount_in_with_fee)
        amount_out = reserve_out - (reserve_in * reserve_out) / (reserve_in + amount_in_with_fee)
        
        # Verify we're not emptying the pool
        if amount_out >= reserve_out:
            raise ValueError("Swap too large: would empty the pool")
        
        return amount_out

    def swap_x_for_y_with_fee(self, dx, fee=None, verbose=True):
        """
        Swaps X for Y.
        Returns the amount of Y received.
        """
        if fee is None:
            fee = self.fee
            
        price_before = self.get_price_x_to_y()
        k_before = self.get_k()
        
        # Calculate output
        dy = self.calculate_swap_output(dx, self.x, self.y, fee)
        
        # Update reserves
        self.x += dx  # ALL dx enters the pool
        self.y -= dy
        
        # Fee remains in the pool as added value to k
        fee_amount = dx * fee
        self.total_fee_x += fee_amount
        
        price_after = self.get_price_x_to_y()
        effective_price = dy / dx
        
        # Slippage: difference between initial price and effective price
        slippage = abs(effective_price - price_before) / price_before
        
        # Price impact: how much the pool price changes
        price_impact = abs(price_after - price_before) / price_before
        
        self.swap_count += 1
        self.k_history.append(self.get_k())
        
        if verbose:
            print(f"[SWAP #{self.swap_count}] {dx:.4f} {self.token_x} → {dy:.2f} {self.token_y}")
            print(f" - Price before: {price_before:.2f} → after: {price_after:.2f} {self.token_y}/{self.token_x}")
            print(f" - Effective price: {effective_price:.2f} {self.token_y}/{self.token_x}")
            print(f" - Slippage: {slippage:.2%} | Price impact: {price_impact:.2%}")
            print(f" - Fee collected: {fee_amount:.4f} {self.token_x}")
            print(f" - Reserves: {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")
            print(f" - k: {k_before:.2f} → {self.get_k():.2f} (Δ: {self.get_k() - k_before:.2f})")
        
        return dy

    def swap_y_for_x_with_fee(self, dy, fee=None, verbose=True):
        """
        Swaps Y for X.
        Returns the amount of X received.
        """
        if fee is None:
            fee = self.fee
            
        price_before = self.get_price_x_to_y()
        k_before = self.get_k()
        
        # Calculate output
        dx = self.calculate_swap_output(dy, self.y, self.x, fee)
        
        # Update reserves
        self.y += dy  # ALL dy enters the pool
        self.x -= dx
        
        # Fee remains in the pool
        fee_amount = dy * fee
        self.total_fee_y += fee_amount
        
        price_after = self.get_price_x_to_y()
        effective_price = dy / dx
        
        slippage = abs(effective_price - price_before) / price_before
        price_impact = abs(price_after - price_before) / price_before
        
        self.swap_count += 1
        self.k_history.append(self.get_k())
        
        if verbose:
            print(f"[SWAP #{self.swap_count}] {dy:.2f} {self.token_y} → {dx:.4f} {self.token_x}")
            print(f"  • Price before: {price_before:.4f} → after: {price_after:.4f} {self.token_x}/{self.token_y}")
            print(f"  • Effective price: {effective_price:.4f} {self.token_x}/{self.token_y}")
            print(f"  • Slippage: {slippage:.2%} | Price impact: {price_impact:.2%}")
            print(f"  • Fee collected: {fee_amount:.2f} {self.token_y}")
            print(f"  • Reserves: {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")
            print(f"  • k: {k_before:.2f} → {self.get_k():.2f} (Δ: {self.get_k() - k_before:.2f})")
        
        return dx

    def get_stats(self):
        """Returns pool statistics"""
        return {
            "reserves_x": self.x,
            "reserves_y": self.y,
            "price_x_to_y": self.get_price_x_to_y(),
            "price_y_to_x": self.get_price_y_to_x(),
            "total_supply": self.total_supply,
            "k": self.get_k(),
            "total_fee_x": self.total_fee_x,
            "total_fee_y": self.total_fee_y,
            "swap_count": self.swap_count,
            "fee_percentage": self.fee * 100
        }

    def __repr__(self):
        return (f"UniswapAMM({self.token_x}/{self.token_y}: "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"price={self.get_price_x_to_y():.2f}, "
                f"k={self.get_k():.2f})")

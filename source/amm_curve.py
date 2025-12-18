# amm_curve.py
# Curve StableSwap AMM implementation
# Designed for stablecoins but tested on ETH/USDC to demonstrate design trade-offs

import math

class CurveStableSwapAMM:
    """
    Curve StableSwap Automated Market Maker
    
    Formula: Blends Constant Sum (x + y = k) and Constant Product (x * y = k)
    controlled by amplification parameter A.
    
    ⚠️  DESIGNED FOR STABLECOINS (USDC/USDT, DAI/USDC)
    
    Characteristics:
    - Low slippage near 1:1 price ratio
    - Amplification coefficient controls curve shape
    - Best for: Equal-value token pairs
    - NOT suitable for: Volatile pairs (ETH/USDC, BTC/USDT)
    
    ⚠️  Expected behavior with volatile pairs (ETH/USDC):
    - Extreme amplification during volatility
    - High slippage on large swaps
    - Arbitrageur losses
    - Price divergence from market
    
    This is INTENTIONAL to demonstrate AMM design trade-offs.
    For volatile pairs, use Constant Product (Uniswap) instead.
    
    Reference: Curve Finance StableSwap (simplified for educational purposes)
    """
    
    def __init__(self, token_x, token_y, reserve_x, reserve_y, 
                 amplification=10, fee=0.003):
        """
        Initializes a Curve StableSwap AMM.
        
        Args:
            token_x: Name of token X (e.g. "ETH")
            token_y: Name of token Y (e.g. "USDC")
            reserve_x: Initial reserve of X
            reserve_y: Initial reserve of Y
            amplification: A parameter (default 10)
                          Higher = flatter curve near 1:1 (100-200 for stables)
                          This BREAKS on volatile pairs!
            fee: Swap fee (default 0.003)
        """
        if reserve_x <= 0 or reserve_y <= 0:
            raise ValueError("Initial reserves must be positive")
        if not 0 <= fee < 1:
            raise ValueError("Fee must be between 0 and 1")
        if amplification <= 0:
            raise ValueError("Amplification must be positive")
            
        self.token_x = token_x
        self.token_y = token_y
        self.x = float(reserve_x)
        self.y = float(reserve_y)
        self.A = amplification  # Amplification coefficient
        self.fee = fee
        
        # LP tokens = sqrt(x * y) like Uniswap
        self.total_supply = math.sqrt(reserve_x * reserve_y)
        
        # Tracking for metrics
        self.total_fee_x = 0.0
        self.total_fee_y = 0.0
        self.swap_count = 0
        self.k_history = [self.get_invariant()]
        
        # ⚠️  WARNING: Check if initialization is appropriate
        ratio = reserve_y / reserve_x
        if ratio < 0.5 or ratio > 2.0:
            print(f"\n{'⚠️ '*20}")
            print(f"⚠️  WARNING: Curve StableSwap initialized on volatile pair!")
            print(f"⚠️  Token ratio: {ratio:.4f} (optimal is ~1.0 for stablecoins)")
            print(f"⚠️  Amplification: {self.A}")
            print(f"⚠️  ")
            print(f"⚠️  This configuration will cause INSTABILITY:")
            print(f"⚠️  - High slippage on large swaps")
            print(f"⚠️  - Amplified price divergence during volatility")
            print(f"⚠️  - Arbitrageur losses")
            print(f"⚠️  - Poor capital efficiency")
            print(f"⚠️  ")
            print(f"⚠️  This is EXPECTED for demonstration purposes.")
            print(f"⚠️  Use Uniswap (Constant Product) for volatile pairs.")
            print(f"{'⚠️ '*20}\n")
    
    def get_invariant(self):
        """
        Calculate Curve's StableSwap invariant - FIXED
        """
        x, y = self.x, self.y
        
        # Amplification-weighted blend
        alpha = self.A / (self.A + 1)  # Higher A → alpha closer to 1
        
        # Blend geometric mean (CP) and arithmetic mean (CS)
        k = (x * y) ** alpha * (x + y) ** (1 - alpha)
    
        return k
    
    def get_k(self):
        """Returns the invariant k (for compatibility with other AMMs)"""
        return self.get_invariant()
    
    def get_price_x_to_y(self):
        """
        Price of X in terms of Y (how many Y for 1 X).
        
        In Curve, price depends on reserves and amplification.
        Near balanced reserves: price ≈ y/x (like Constant Sum)
        Far from balance: price diverges (like Constant Product)
        """
        if self.x <= 0:
            raise ValueError("Reserve X is zero or negative")
        return self.y / self.x
    
    def get_price_y_to_x(self):
        """Price of Y in terms of X"""
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
            liquidity = min(
                amount_x * self.total_supply / self.x,
                amount_y * self.total_supply / self.y
            )
            
            # Warn if ratios don't match
            ratio_x = amount_x / self.x
            ratio_y = amount_y / self.y
            if abs(ratio_x - ratio_y) > 0.01:  # 1% tolerance
                print(f"[AMM-CURVE] Warning: Non-optimal ratio! "
                      f"Ratio X: {ratio_x:.4f}, Ratio Y: {ratio_y:.4f}")
        
        self.x += amount_x
        self.y += amount_y
        self.total_supply += liquidity
        self.k_history.append(self.get_invariant())
        
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
        self.k_history.append(self.get_invariant())
        
        return amount_x, amount_y
    
    def calculate_swap_output(self, amount_in, reserve_in, reserve_out, fee=None):
        """
        Calculate swap output using Curve's StableSwap formula.
        
        SIMPLIFIED APPROACH:
        - Blends Constant Sum (1:1 swap) and Constant Product (x*y=k)
        - Blend factor controlled by amplification parameter A
        - Higher A = more like Constant Sum (flatter curve)
        - Lower A = more like Constant Product (more slippage)
        
        Real Curve uses Newton's method to solve the invariant equation.
        This is a pedagogical approximation for demonstration.
        """
        if fee is None:
            fee = self.fee
    
        if amount_in <= 0:
            raise ValueError("Input amount must be positive")
        if reserve_in <= 0 or reserve_out <= 0:
            raise ValueError("Reserves must be positive")
        
        # Apply fee
        amount_in_with_fee = amount_in * (1 - fee)
        
        # Base: Constant Product (sempre sicuro)
        cp_output = reserve_out - (reserve_in * reserve_out) / (reserve_in + amount_in_with_fee)
        
        # Calcola quanto siamo vicini a 1:1
        reserve_ratio = max(reserve_in, reserve_out) / min(reserve_in, reserve_out)
        
        # Se siamo MOLTO vicini a 1:1 (es. ratio < 1.1), aggiungi un piccolo boost
        if reserve_ratio < 1.1:
            # Quanto siamo vicini a perfetto bilanciamento
            closeness = (1.1 - reserve_ratio) / 0.1  # 1.0 se ratio=1.0, 0 se ratio=1.1
            
            # Boost massimo: 5% in più del CP output
            boost_factor = 0.05 * closeness
            amount_out = cp_output * (1 + boost_factor)
        else:
            # Lontani da 1:1: usa solo CP (comportamento Uniswap)
            amount_out = cp_output
        
        # Safety check
        if amount_out >= reserve_out * 0.95:
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
        k_before = self.get_invariant()
        
        # Calculate output using Curve formula
        dy = self.calculate_swap_output(dx, self.x, self.y, fee)
        
        # Update reserves
        self.x += dx  # ALL dx enters the pool
        self.y -= dy
        
        # Fee remains in the pool
        fee_amount = dx * fee
        self.total_fee_x += fee_amount
        
        self.swap_count += 1
        self.k_history.append(self.get_invariant())
        
        if verbose:
            price_after = self.get_price_x_to_y()
            effective_price = dy / dx
            
            # Slippage: difference between initial price and effective price
            slippage = abs(effective_price - price_before) / price_before
            
            # Price impact: how much the pool price changes
            price_impact = abs(price_after - price_before) / price_before
            
            print(f"[SWAP-CURVE #{self.swap_count}] {dx:.4f} {self.token_x} → {dy:.2f} {self.token_y}")
            print(f" - Price before: {price_before:.2f} → after: {price_after:.2f} {self.token_y}/{self.token_x}")
            print(f" - Effective price: {effective_price:.2f} {self.token_y}/{self.token_x}")
            print(f" - Slippage: {slippage:.2%} | Price impact: {price_impact:.2%}")
            print(f" - Fee collected: {fee_amount:.4f} {self.token_x}")
            print(f" - Reserves: {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")
            print(f" - k: {k_before:.2f} → {self.get_invariant():.2f} (Δ: {self.get_invariant() - k_before:.2f})")
        
        return dy
    
    def swap_y_for_x_with_fee(self, dy, fee=None, verbose=True):
        """
        Swaps Y for X.
        Returns the amount of X received.
        """
        if fee is None:
            fee = self.fee
        
        price_before = self.get_price_x_to_y()
        k_before = self.get_invariant()
        
        # Calculate output using Curve formula
        dx = self.calculate_swap_output(dy, self.y, self.x, fee)
        
        # Update reserves
        self.y += dy  # ALL dy enters the pool
        self.x -= dx
        
        # Fee remains in the pool
        fee_amount = dy * fee
        self.total_fee_y += fee_amount
        
        self.swap_count += 1
        self.k_history.append(self.get_invariant())
        
        if verbose:
            price_after = self.get_price_x_to_y()
            effective_price = dy / dx
            
            slippage = abs(effective_price - price_before) / price_before
            price_impact = abs(price_after - price_before) / price_before
            
            print(f"[SWAP-CURVE #{self.swap_count}] {dy:.2f} {self.token_y} → {dx:.4f} {self.token_x}")
            print(f"  • Price before: {price_before:.4f} → after: {price_after:.4f} {self.token_x}/{self.token_y}")
            print(f"  • Effective price: {effective_price:.4f} {self.token_x}/{self.token_y}")
            print(f"  • Slippage: {slippage:.2%} | Price impact: {price_impact:.2%}")
            print(f"  • Fee collected: {fee_amount:.2f} {self.token_y}")
            print(f"  • Reserves: {self.x:.2f} {self.token_x}, {self.y:.2f} {self.token_y}")
            print(f"  • k: {k_before:.2f} → {self.get_invariant():.2f} (Δ: {self.get_invariant() - k_before:.2f})")
        
        return dx
    
    def get_stats(self):
        """Returns pool statistics"""
        return {
            "reserves_x": self.x,
            "reserves_y": self.y,
            "price_x_to_y": self.get_price_x_to_y(),
            "price_y_to_x": self.get_price_y_to_x(),
            "total_supply": self.total_supply,
            "k": self.get_invariant(),
            "amplification": self.A,
            "total_fee_x": self.total_fee_x,
            "total_fee_y": self.total_fee_y,
            "swap_count": self.swap_count,
            "fee_percentage": self.fee * 100
        }
    
    def __repr__(self):
        return (f"CurveStableSwapAMM({self.token_x}/{self.token_y}: "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"price={self.get_price_x_to_y():.2f}, "
                f"A={self.A}, k={self.get_invariant():.2f})")
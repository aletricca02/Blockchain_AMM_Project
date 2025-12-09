# Il file implementta due tipi di agenti specializzati che interafiscono con l'AMM.
# Arbitrageur: Sfrutta le differenze di prezzo tra AMM e mercato esterno.
# PanicLP: Liquidity Provider che ritira fondi durante alta volatilità.
# WhaleTrader: Trader che esegue operazioni di grande volume (utile per scenari caotici).

import math

class Arbitrageur:
    def __init__(self, name, capital_x=1000, capital_y=1000, min_profit_threshold=0.005):
        """
        Args:
            name: Arbitrageur's name
            capital_x: Initial capital in token X (e.g. ETH)
            capital_y: Initial capital in token Y (e.g. USDC)
            min_profit_threshold: Minimum profit threshold (default 0.5%)
        """
        self.name = name
        self.capital_x = capital_x
        self.capital_y = capital_y
        self.initial_capital_x = capital_x
        self.initial_capital_y = capital_y
        self.min_profit_threshold = min_profit_threshold
        self.trades_executed = 0
        self.total_profit_x = 0
        self.total_profit_y = 0

    def act(self, amm, market_price_x_to_y):
        """
        Executes arbitrage between AMM and external market.
        """
        if market_price_x_to_y <= 0:
            print(f"[ARB] {self.name}: Invalid market price ({market_price_x_to_y})")
            return
            
        amm_price = amm.get_price_x_to_y()
        
        if amm_price <= 0:
            print(f"[ARB] {self.name}: Invalid AMM price ({amm_price})")
            return
        
        # Calculate percentage spread
        spread = abs(amm_price - market_price_x_to_y) / amm_price
        
        if spread < self.min_profit_threshold:
            return  # Spread too small
            
        k = amm.x * amm.y
        
        try:
            if amm_price < market_price_x_to_y:
                # X is undervalued in the AMM -> BUY X
                target_y = math.sqrt(k * market_price_x_to_y)
                amount_y_in = target_y - amm.y
                
                if amount_y_in > 0 and amount_y_in <= self.capital_y:
                    print(f"[ARB] {self.name} | Spread: {spread:.2%} | AMM: {amm_price:.2f} < MKT: {market_price_x_to_y:.2f}")
                    
                    # Calculate how much X we will receive
                    x_before = amm.x
                    amm.swap_y_for_x_with_fee(amount_y_in)
                    x_received = x_before - amm.x
                    
                    # Update portfolio
                    self.capital_y -= amount_y_in
                    self.capital_x += x_received
                    self.trades_executed += 1
                                        
                elif amount_y_in > self.capital_y:
                    print(f"[ARB] {self.name}: Insufficient Y capital ({self.capital_y:.2f} < {amount_y_in:.2f})")
                    
            else:
                # X is overvalued in the AMM -> SELL X
                target_x = math.sqrt(k / market_price_x_to_y)
                amount_x_in = target_x - amm.x
                
                if amount_x_in > 0 and amount_x_in <= self.capital_x:
                    print(f"[ARB] {self.name} | Spread: {spread:.2%} | AMM: {amm_price:.2f} > MKT: {market_price_x_to_y:.2f}")
                    
                    # Calculate how much Y we will receive
                    y_before = amm.y
                    amm.swap_x_for_y_with_fee(amount_x_in)
                    y_received = y_before - amm.y
                    
                    # Update portfolio
                    self.capital_x -= amount_x_in
                    self.capital_y += y_received
                    self.trades_executed += 1
                                        
                elif amount_x_in > self.capital_x:
                    print(f"[ARB] {self.name}: Insufficient X capital ({self.capital_x:.4f} < {amount_x_in:.4f})")
                    
        except Exception as e:
            print(f"[ARB] {self.name}: Error during arbitrage: {e}")
    
    def get_profit(self, current_market_price):
        """Calculates total profit in terms of fiat value"""
        current_value = self.capital_x * current_market_price + self.capital_y
        initial_value = self.initial_capital_x * current_market_price + self.initial_capital_y
        return current_value - initial_value
    
    def get_stats(self):
        """Returns arbitrageur's statistics"""
        return {
            "name": self.name,
            "trades": self.trades_executed,
            "capital_x": self.capital_x,
            "capital_y": self.capital_y,
            "initial_capital_x": self.initial_capital_x,
            "initial_capital_y": self.initial_capital_y
        }


class PanicLP:
    def __init__(self, name, initial_share_percent, amm, panic_threshold=0.08, 
                 gradual_exit=False, exit_percent=1.0):
        """
        Args:
            name: LP's name
            initial_share_percent: Initial % of pool owned (0-1)
            amm: Reference to the AMM
            panic_threshold: Volatility threshold for panic (default 8%)
            gradual_exit: If True, exits gradually instead of all at once
            exit_percent: % of liquidity to remove when panicking (default 100%)
        """
        self.name = name
        self.lp_tokens = amm.total_supply * initial_share_percent
        self.initial_lp_tokens = self.lp_tokens
        self.has_exited = False
        self.panic_threshold = panic_threshold
        self.gradual_exit = gradual_exit
        self.exit_percent = exit_percent
        self.panic_events = 0

    def check_stress(self, amm, volatility_index):
        """
        Checks market stress and reacts accordingly.
        
        Args:
            amm: The AMM from which to remove liquidity
            volatility_index: Volatility index (0-1, where 0.08 = 8%)
        """
        if self.has_exited or self.lp_tokens <= 0:
            return
            
        if volatility_index > self.panic_threshold:
            self.panic_events += 1
            
            if self.gradual_exit:
                # Gradual exit: withdraws a % of remaining tokens
                tokens_to_remove = self.lp_tokens * self.exit_percent
            else:
                # Complete exit
                tokens_to_remove = self.lp_tokens
            
            try:
                print(f"[PANIC] {self.name} | Volatility: {volatility_index:.2%} > {self.panic_threshold:.2%}")
                print(f"Removes {tokens_to_remove/self.initial_lp_tokens:.1%} of initial liquidity")
                
                dx, dy = amm.remove_liquidity(tokens_to_remove)
                
                self.lp_tokens -= tokens_to_remove
                
                print(f"Withdrawn: {dx:.4f} X + {dy:.2f} Y")
                
                if self.lp_tokens < 0.01:  # Negligible threshold
                    self.has_exited = True
                    print(f"{self.name} has completely exited the pool")
                    
            except Exception as e:
                print(f"[PANIC] {self.name}: Error during withdrawal: {e}")
    
    def get_stats(self):
        """Returns LP's statistics"""
        return {
            "name": self.name,
            "lp_tokens": self.lp_tokens,
            "initial_lp_tokens": self.initial_lp_tokens,
            "has_exited": self.has_exited,
            "panic_events": self.panic_events,
            "remaining_share": self.lp_tokens / self.initial_lp_tokens if self.initial_lp_tokens > 0 else 0
        }


class WhaleTrader:
    """Trader that executes large volume operations (useful for chaotic scenarios)"""
    def __init__(self, name, capital_x=20, capital_y=40000):
        self.name = name
        self.capital_x = capital_x
        self.capital_y = capital_y
    
    def dump(self, amm, percent=0.1):
        """Sells a large amount of X (dump)"""
        amount = self.capital_x * percent
        if amount > 0:
            print(f"[WHALE] {self.name} dumps {amount:.2f} X!")
            try:
                amm.swap_x_for_y_with_fee(amount)
                self.capital_x -= amount
            except Exception as e:
                print(f"Dump failed: {e}")
    
    def pump(self, amm, percent=0.1):
        """Buys a large amount of X (pump)"""
        amount = self.capital_y * percent
        if amount > 0:
            print(f"[WHALE] {self.name} pumps with {amount:.2f} Y!")
            try:
                amm.swap_y_for_x_with_fee(amount)
                self.capital_y -= amount
            except Exception as e:
                print(f"Pump failed: {e}")
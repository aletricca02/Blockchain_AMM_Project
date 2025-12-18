# agents.py
# Agenti aggiornati per gestire correttamente il parametro 'verbose'

import math

class Arbitrageur:
    def __init__(self, name, capital_x=1000, capital_y=1000, min_profit_threshold=0.005):
        self.name = name
        self.capital_x = capital_x
        self.capital_y = capital_y
        self.initial_capital_x = capital_x
        self.initial_capital_y = capital_y
        self.min_profit_threshold = min_profit_threshold
        self.trades_executed = 0

    # FIX: Aggiunto parametro verbose=True
    def act(self, amm, market_price_x_to_y, verbose=True):
        if market_price_x_to_y <= 0: return

        amm_price = amm.get_price_x_to_y()
        if amm_price <= 0: return

        spread = abs(amm_price - market_price_x_to_y) / amm_price
        if spread < self.min_profit_threshold:
            return

        k = amm.get_k() # Usa il metodo get_k per compatibilità

        try:
            if amm_price < market_price_x_to_y:
                # BUY X (Price AMM < Market)
                # Stima target usando formula Constant Product (semplificata per tutti)
                # x_new = sqrt(k / market_price)
                # dy = y - y_new
                
                # Per semplicità nei vari AMM, proviamo a chiudere parte del gap
                # Logica semplificata: arbitraggio iterativo
                # Compra X finché conviene
                
                # Calcolo approssimativo per CPMM
                target_y = math.sqrt(k * market_price_x_to_y)
                amount_y_in = target_y - amm.y
                
                # Se siamo in Constant Sum o Curve, questa stima potrebbe essere imprecisa
                # ma l'importante è la direzione
                if amount_y_in <= 0: amount_y_in = 1.0 # Tentativo minimo

                if amount_y_in > self.capital_y:
                    amount_y_in = self.capital_y # Usa tutto quello che ha

                if amount_y_in > 0:
                    if verbose:
                        print(f"[ARB] {self.name} | Spread: {spread:.2%} | AMM: {amm_price:.2f} < MKT: {market_price_x_to_y:.2f}")
                    
                    x_before = amm.x
                    # Passiamo verbose all'AMM
                    amm.swap_y_for_x_with_fee(amount_y_in, verbose=verbose)
                    x_received = x_before - amm.x
                    
                    self.capital_y -= amount_y_in
                    self.capital_x += x_received
                    self.trades_executed += 1

            else:
                # SELL X (Price AMM > Market)
                target_x = math.sqrt(k / market_price_x_to_y)
                amount_x_in = target_x - amm.x

                if amount_x_in <= 0: amount_x_in = 0.1 # Tentativo minimo

                if amount_x_in > self.capital_x:
                    amount_x_in = self.capital_x

                if amount_x_in > 0:
                    if verbose:
                        print(f"[ARB] {self.name} | Spread: {spread:.2%} | AMM: {amm_price:.2f} > MKT: {market_price_x_to_y:.2f}")
                    
                    y_before = amm.y
                    # Passiamo verbose all'AMM
                    amm.swap_x_for_y_with_fee(amount_x_in, verbose=verbose)
                    y_received = y_before - amm.y
                    
                    self.capital_x -= amount_x_in
                    self.capital_y += y_received
                    self.trades_executed += 1

        except Exception as e:
            if verbose: print(f"[ARB] Error: {e}")

    def get_profit(self, current_market_price):
        current_val = self.capital_x * current_market_price + self.capital_y
        initial_val = self.initial_capital_x * current_market_price + self.initial_capital_y
        return current_val - initial_val


class PanicLP:
    def __init__(self, name, initial_share_percent, amm, panic_threshold=0.08):
        self.name = name
        self.lp_tokens = amm.total_supply * initial_share_percent
        self.initial_lp_tokens = self.lp_tokens
        self.has_exited = False
        self.panic_threshold = panic_threshold
        self.panic_events = 0

    # FIX: Aggiunto parametro verbose=True
    def check_stress(self, amm, volatility_index, verbose=True):
        if self.has_exited or self.lp_tokens <= 0:
            return

        if volatility_index > self.panic_threshold:
            self.panic_events += 1
            
            # Panic: rimuovi tutto
            tokens_to_remove = self.lp_tokens
            
            try:
                if verbose:
                    print(f"\n{'!'*40}")
                    print(f"[PANIC] {self.name} | Volatility: {volatility_index:.1%} > {self.panic_threshold:.1%}")
                    print(f"Selling all liquidity to save capital!")
                    print(f"{'!'*40}")

                dx, dy = amm.remove_liquidity(tokens_to_remove)
                self.lp_tokens -= tokens_to_remove
                
                if verbose:
                    print(f" -> Withdrawn: {dx:.2f} X + {dy:.2f} Y")
                    print(f" -> {self.name} has completely exited the pool.\n")

                self.has_exited = True
                    
            except Exception as e:
                if verbose: print(f"[PANIC] Error: {e}")

    def get_stats(self):
        return {
            "name": self.name,
            "has_exited": self.has_exited,
            "panic_events": self.panic_events
        }


class WhaleTrader:
    def __init__(self, name, capital_x=20, capital_y=40000):
        self.name = name
        self.capital_x = capital_x
        self.capital_y = capital_y
    
    # FIX: Aggiunto parametro verbose=True
    def dump(self, amm, percent=0.1, verbose=True):
        amount = self.capital_x * percent
        if amount > 0:
            if verbose: print(f"[WHALE] {self.name} dumps {amount:.2f} X!")
            try:
                amm.swap_x_for_y_with_fee(amount, verbose=verbose)
                self.capital_x -= amount
            except Exception as e:
                if verbose: print(f"Dump failed: {e}")
    
    # FIX: Aggiunto parametro verbose=True
    def pump(self, amm, percent=0.1, verbose=True):
        amount = self.capital_y * percent
        if amount > 0:
            if verbose: print(f"[WHALE] {self.name} pumps with {amount:.2f} Y!")
            try:
                amm.swap_y_for_x_with_fee(amount, verbose=verbose)
                self.capital_y -= amount
            except Exception as e:
                if verbose: print(f"Pump failed: {e}")
import math

class Arbitrageur:
    def __init__(self, name):
        self.name = name

    def act(self, amm, market_price_x_to_y):
        """
        Guarda il prezzo esterno e fa swap sull'AMM per allinearlo e fare profitto.
        """
        amm_price = amm.get_price_x_to_y()
        
        # Soglia minima per coprire le fee (es. 0.5% differenza)
        if abs(amm_price - market_price_x_to_y) / amm_price < 0.005:
            return # Gap troppo piccolo, non conviene
            
        k = amm.x * amm.y
        
        if amm_price < market_price_x_to_y:
            # X è troppo economico nell'AMM -> COMPRA X (Vende Y)
            # Target: vogliamo che y_new / x_new = market_price
            # Formula derivata da CPMM: new_y = sqrt(k * market_price)
            target_y = math.sqrt(k * market_price_x_to_y)
            amount_y_in = target_y - amm.y
            
            if amount_y_in > 0:
                print(f"🤖 [ARB] {self.name} vede spread! AMM: {amm_price:.2f} < MKT: {market_price_x_to_y:.2f}")
                amm.swap_y_for_x_with_fee(amount_y_in)
                
        else:
            # X è troppo costoso nell'AMM -> VENDE X (Compra Y)
            # Target: new_x = sqrt(k / market_price)
            target_x = math.sqrt(k / market_price_x_to_y)
            amount_x_in = target_x - amm.x
            
            if amount_x_in > 0:
                print(f"🤖 [ARB] {self.name} vede spread! AMM: {amm_price:.2f} > MKT: {market_price_x_to_y:.2f}")
                amm.swap_x_for_y_with_fee(amount_x_in)

class PanicLP:
    def __init__(self, name, initial_share_percent, amm):
        self.name = name
        # L'LP possiede una % iniziale della pool
        self.lp_tokens = amm.total_supply * initial_share_percent
        self.has_exited = False

    def check_stress(self, amm, volatility_index):
        """
        Se la volatilità supera una soglia, l'LP ritira tutto.
        """
        if not self.has_exited and volatility_index > 0.08: # Soglia panico (8% variazione)
            print(f"😱 [PANIC] {self.name} ha paura! Rimuove liquidità.")
            dx, dy = amm.remove_liquidity(self.lp_tokens)
            print(f"   -> Ritirati {dx:.2f} ETH e {dy:.2f} USDC")
            self.lp_tokens = 0
            self.has_exited = True
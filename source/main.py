from amm import UniswapAMM
from trader import Trader

traders = []
amm = None

def setup_pool():
    global amm
    print("\n🔧 Setup Pool")
    eth = float(input("  Quanti ETH vuoi nel pool iniziale? "))
    usdc = float(input("  Quanti USDC vuoi nel pool iniziale? "))
    amm = UniswapAMM("ETH", "USDC", eth, usdc)
    print(f"✅ Pool creato: {eth} ETH, {usdc} USDC\n")

def create_trader():
    name = input("Nome del trader: ")
    token = input("Token iniziale (ETH o USDC): ").upper()
    amount = float(input(f"Quantità iniziale di {token}: "))
    trader = Trader(name, token, amount)
    traders.append(trader)
    print(f"✅ Trader {name} creato con {amount} {token}\n")

def list_traders():
    print("\n👥 Trader disponibili:")
    for i, t in enumerate(traders):
        print(f"  [{i}] {t.name} – {t.amount:.2f} {t.token}")
    print()

def execute_swap():
    list_traders()
    choice = int(input("Scegli il trader (ID): "))
    t = traders[choice]
    amount = float(input(f"Quanto {t.token} vuole scambiare? "))
    t.amount -= amount
    t.amount = max(t.amount, 0)

    # Esegui swap
    if t.token == "ETH":
        dy = amm.swap_x_for_y_with_fee(amount)
        print(f"🪙 Trader {t.name} ha ricevuto {dy:.2f} USDC\n")
    else:
        dx = amm.swap_y_for_x_with_fee(amount)
        print(f"🪙 Trader {t.name} ha ricevuto {dx:.4f} ETH\n")

def show_pool_status():
    print(f"\n📊 Pool status:")
    print(f"  Reserves: {amm.x:.2f} ETH, {amm.y:.2f} USDC")
    print(f"  Price: 1 ETH = {amm.get_price_x_to_y():.2f} USDC\n")

def main():
    while True:
        print("======== MENU ========")
        print("1. Setup pool")
        print("2. Crea nuovo trader")
        print("3. Lista trader")
        print("4. Stato del pool")
        print("5. Esegui uno swap")
        print("0. Esci")
        choice = input("Scegli un’opzione: ")

        if choice == "1":
            setup_pool()
        elif choice == "2":
            create_trader()
        elif choice == "3":
            list_traders()
        elif choice == "4":
            show_pool_status()
        elif choice == "5":
            execute_swap()
        elif choice == "0":
            print("👋 Esci dal simulatore.")
            break
        else:
            print("❌ Scelta non valida.\n")

if __name__ == "__main__":
    main()

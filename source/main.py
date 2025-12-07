from amm import UniswapAMM
from trader import Trader
from run_simulation import Simulation

sim = Simulation()
step_count = 0

while True:
    print("\n======== SIM MENU ========")
    print("1. Avanza 1 step")
    print("2. Avanza 10 step")
    print("3. Applica shock -30%")
    print("4. Applica pump +25%")
    print("5. Stato attuale")
    print("6. Mostra grafico finale")
    print("0. Esci")
    choice = input("Scelta: ")

    if choice == "1":
        step_count += 1
        sim.step(step_count)
    elif choice == "2":
        for _ in range(10):
            step_count += 1
            sim.step(step_count)
    elif choice == "3":
        sim.apply_market_shock(-0.3)
    elif choice == "4":
        sim.apply_market_shock(+0.25)
    elif choice == "5":
        sim.print_status()
    elif choice == "6":
        sim.plot()
    elif choice == "0":
        break
    else:
        print("❌ Opzione non valida.")

# Il file implementa il menù interattivo per controllare la simulazione degli AMM.

from amm import UniswapAMM
from trader import Trader
from run_simulation import Simulation

from run_simulation import Simulation
import sys

def main():
    sim = Simulation()
    
    while True:
        print("\n" + "="*30)
        print("      AMM SIMULATION MENU")
        print("="*30)
        print("1. Go forward 1 step")
        print("2. Go forward N steps")
        print("3. Apply market shock")
        print("4. Current status")
        print("5. Show graphs")
        print("6. Save results")
        print("0. Exit")
        print("="*30)
        
        try:
            choice = input("Choice: ").strip()
            
            if choice == "1":
                sim.step()
                print("Step completed.")
                
            elif choice == "2":
                try:
                    n = int(input("How many steps? "))
                    if n <= 0:
                        print("Enter a positive number")
                        continue
                    for i in range(n):
                        sim.step(verbose=(i == n-1))  # only last one verbose
                    print(f"✓ {n} steps completed")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == "3":
                try:
                    shock_pct = float(input("Shock % (es. -30 o +25): "))
                    shock = shock_pct / 100
                    confirm = input(f"Do you confirm {shock_pct:+.1f}% shock? (y/n): ")
                    if confirm.lower() == 's':
                        sim.apply_market_shock(shock)
                        print(f"{shock_pct:+.1f}% shock applied")
                except ValueError:
                    print("Invalid value")
                    
            elif choice == "4":
                sim.print_status()
                
            elif choice == "5":
                sim.plot()
                
            elif choice == "6":
                filename = input("File name (default: results.csv): ").strip()
                if not filename:
                    filename = "results.csv"
                sim.save_results(filename)
                print(f"Results saved in {filename}")
                
            elif choice == "0":
                confirm = input("Do you really want to exit? (y/n): ")
                if confirm.lower() == 'y':
                    print("Simulation finished")
                    break
                    
            else:
                print("Invalid option")
                
        except KeyboardInterrupt:
            print("\n\nInterruption detected")
            confirm = input("Do you want to save before exiting? (y/n): ")
            if confirm.lower() == 'y':
                sim.save_results("interrupted_results.csv")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            print("The simulation continues...")

if __name__ == "__main__":
    main()

# experiments.py
# Automated experiment runner for comparing AMM models under various scenarios.
# Uses MASTER_SEED for reproducibility and generates comparative analysis.

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
from run_simulation import Simulation

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ExperimentRunner:
    """
    Orchestrates automated experiments across different AMM types and scenarios.
    
    Ensures reproducibility through MASTER_SEED and generates comparative reports.
    """
    
    def __init__(self, master_seed=42, output_dir="experiments_output"):
        """
        Initialize experiment runner.
        
        Args:
            master_seed: Master seed for reproducibility
            output_dir: Directory to save results
        """
        self.master_seed = master_seed
        self.output_dir = output_dir
        self.results = []  # Per i risultati aggregati (con repetitions)
        self.individual_runs = []  # Per i singoli run (opzionale)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"{'='*60}")
        print(f"EXPERIMENT RUNNER INITIALIZED")
        print(f"{'='*60}")
        print(f"Master Seed: {master_seed}")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*60}\n")
    
    def set_seed(self, seed):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)

    def run_baseline_experiment(self, amm_type, steps=500, seed_offset=0):
        """
        Run baseline experiment with normal volatility.
        
        Args:
            amm_type: "uniswap" or "constant_sum"
            steps: Number of simulation steps
            seed_offset: Offset to add to master seed
            
        Returns:
            dict: Experiment results
        """
        self.set_seed(self.master_seed + seed_offset)
        
        print(f"\n{'='*60}")
        print(f"BASELINE EXPERIMENT: {amm_type.upper()}")
        print(f"Steps: {steps} | Seed: {self.master_seed + seed_offset}")
        print(f"{'='*60}")
        
        sim = Simulation(amm_type=amm_type)
        
        # Run simulation
        for i in range(steps):
            sim.step(verbose=False)
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{steps} steps")
        
        # Collect results
        stats = sim.get_stats()
        
        result = {
            "experiment": "baseline",
            "amm_type": amm_type,
            "steps": steps,
            "seed": self.master_seed + seed_offset,
            **stats
        }
        
        self.results.append(result)
        
        print(f"✓ Baseline experiment completed for {amm_type}")
        return result
    
    def run_flash_crash_experiment(self, amm_type, crash_step=250, 
                                   crash_magnitude=-0.5, total_steps=500, 
                                   seed_offset=100):
        """
        Run experiment with flash crash event.
        
        Args:
            amm_type: "uniswap" or "constant_sum"
            crash_step: Step at which crash occurs
            crash_magnitude: Crash percentage (e.g., -0.5 = -50%)
            total_steps: Total simulation steps
            seed_offset: Offset to add to master seed
            
        Returns:
            dict: Experiment results
        """
        self.set_seed(self.master_seed + seed_offset)
        
        print(f"\n{'='*60}")
        print(f"FLASH CRASH EXPERIMENT: {amm_type.upper()}")
        print(f"Crash at step {crash_step}: {crash_magnitude:.0%}")
        print(f"{'='*60}")
        
        sim = Simulation(amm_type=amm_type)
        
        # Run until crash
        for i in range(crash_step):
            sim.step(verbose=False)
        
        print(f"⚡ APPLYING FLASH CRASH: {crash_magnitude:.0%}")
        sim.apply_market_shock(crash_magnitude)
        
        # Continue simulation
        for i in range(crash_step, total_steps):
            sim.step(verbose=False)
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total_steps} steps")
        
        stats = sim.get_stats()
        
        result = {
            "experiment": "flash_crash",
            "amm_type": amm_type,
            "steps": total_steps,
            "crash_step": crash_step,
            "crash_magnitude": crash_magnitude,
            "seed": self.master_seed + seed_offset,
            **stats
        }
        
        self.results.append(result)
        
        print(f"✓ Flash crash experiment completed for {amm_type}")
        return result
    
    def run_whale_manipulation_experiment(self, amm_type, dump_step=200, 
                                         pump_step=400, total_steps=600, 
                                         seed_offset=200):
        """
        Run experiment with whale pump and dump.
        
        Args:
            amm_type: "uniswap" or "constant_sum"
            dump_step: Step at which dump occurs
            pump_step: Step at which pump occurs
            total_steps: Total simulation steps
            seed_offset: Offset to add to master seed
            
        Returns:
            dict: Experiment results
        """
        self.set_seed(self.master_seed + seed_offset)
        
        print(f"\n{'='*60}")
        print(f"WHALE MANIPULATION EXPERIMENT: {amm_type.upper()}")
        print(f"Dump at step {dump_step} | Pump at step {pump_step}")
        print(f"{'='*60}")
        
        sim = Simulation(amm_type=amm_type)
        
        # Run until dump
        for i in range(dump_step):
            sim.step(verbose=False)
        
        print(f"🐋 WHALE DUMP")
        sim.whale_dump(percent=0.5)
        
        # Run until pump
        for i in range(dump_step, pump_step):
            sim.step(verbose=False)
        
        print(f"🐋 WHALE PUMP")
        sim.whale_pump(percent=0.5)
        
        # Continue simulation
        for i in range(pump_step, total_steps):
            sim.step(verbose=False)
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total_steps} steps")
        
        stats = sim.get_stats()
        
        result = {
            "experiment": "whale_manipulation",
            "amm_type": amm_type,
            "steps": total_steps,
            "dump_step": dump_step,
            "pump_step": pump_step,
            "seed": self.master_seed + seed_offset,
            **stats
        }
        
        self.results.append(result)
        
        print(f"✓ Whale manipulation experiment completed for {amm_type}")
        return result
    
    def run_high_volatility_experiment(self, amm_type, steps=500, 
                                      volatility_multiplier=5.0, 
                                      seed_offset=300):
        """
        Run experiment with high market volatility.
        
        Args:
            amm_type: "uniswap" or "constant_sum"
            steps: Number of simulation steps
            volatility_multiplier: Multiply base volatility by this factor
            seed_offset: Offset to add to master seed
            
        Returns:
            dict: Experiment results
        """
        self.set_seed(self.master_seed + seed_offset)
        
        print(f"\n{'='*60}")
        print(f"HIGH VOLATILITY EXPERIMENT: {amm_type.upper()}")
        print(f"Volatility multiplier: {volatility_multiplier}x")
        print(f"{'='*60}")
        
        sim = Simulation(amm_type=amm_type)
        
        # Temporarily increase volatility
        # Note: This modifies the step() method's volatility range
        original_step = sim.step
        
        def high_volatility_step(verbose=False):
            """Modified step with higher volatility"""
            sim.step_count += 1
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"TICK {sim.step_count}")
                print(f"{'='*60}")
            
            # Higher volatility: ±10% instead of ±2%
            base_vol = 0.02
            change_pct = random.uniform(-base_vol * volatility_multiplier, 
                                       base_vol * volatility_multiplier)
            sim.market_price *= (1 + change_pct)
            volatility = abs(change_pct)
            
            if verbose:
                print(f"Market movement: {change_pct:+.2%} → New price: {sim.market_price:.2f} USDC/ETH")
            
            # Arbitrageur attempts to close price gaps
            sim.arb.act(sim.amm, sim.market_price)
            
            # Retail traders
            for t in sim.traders:
                t.act(sim.amm, strategy="random", verbose=verbose)
            
            # LP monitors volatility
            sim.lp.check_stress(sim.amm, volatility)
            
            # Log data
            sim.price_log.append({
                "step": sim.step_count,
                "amm_price": sim.amm.get_price_x_to_y(),
                "market_price": sim.market_price,
                "price_gap": abs(sim.amm.get_price_x_to_y() - sim.market_price),
                "reserves_x": sim.amm.x,
                "reserves_y": sim.amm.y,
                "k": sim.amm.get_k(),
                "volatility": volatility
            })
            
            if verbose:
                print(f"{'='*60}")
        
        # Replace step method temporarily
        sim.step = high_volatility_step
        
        # Run simulation
        for i in range(steps):
            sim.step(verbose=False)
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{steps} steps")
        
        stats = sim.get_stats()
        
        result = {
            "experiment": "high_volatility",
            "amm_type": amm_type,
            "steps": steps,
            "volatility_multiplier": volatility_multiplier,
            "seed": self.master_seed + seed_offset,
            **stats
        }
        
        self.results.append(result)
        
        print(f"✓ High volatility experiment completed for {amm_type}")
        return result
    
    def run_experiment_with_repetitions(self, experiment_func, amm_type, 
                                    num_runs=10, **kwargs):
        """
        Run an experiment multiple times and collect statistics.
        
        Args:
            experiment_func: Function to run (e.g., self.run_baseline_experiment)
            amm_type: "uniswap" or "constant_sum"
            num_runs: Number of repetitions
            **kwargs: Parameters to pass to experiment_func
            
        Returns:
            dict: Aggregated results with mean, std, min, max
        """
        print(f"\n{'='*60}")
        print(f"RUNNING {num_runs} REPETITIONS: {amm_type.upper()}")
        print(f"{'='*60}")
        
        all_runs = []
        
        # IMPORTANTE: Salva temporaneamente self.results
        temp_results = self.results.copy()
        self.results = []  # Svuota per raccogliere solo i run correnti
        
        for run_id in range(num_runs):
            print(f"\n[Run {run_id+1}/{num_runs}]")
            
            # Use different seed for each run
            seed_offset = kwargs.get('seed_offset', 0) + run_id * 1000
            kwargs['seed_offset'] = seed_offset
            
            result = experiment_func(amm_type, **kwargs)
            all_runs.append(result)
        
        # Salva tutti i run individuali (opzionale)
        self.individual_runs.extend(all_runs)
        
        # Ripristina self.results
        self.results = temp_results
        
        # Aggregate results
        df_runs = pd.DataFrame(all_runs)
        
        # Calculate statistics
        aggregated = {
            "experiment": all_runs[0]["experiment"],
            "amm_type": amm_type,
            "num_runs": num_runs,
            "steps": all_runs[0]["steps"],
            "seed_base": self.master_seed
        }
        
        # Metrics to aggregate
        metrics = ['avg_price_gap', 'max_price_gap', 'price_stability', 
                'k_growth_percent', 'arb_profit', 'arb_trades', 
                'lp_panic_events', 'total_swaps', 'avg_volatility', 
                'max_volatility']
        
        for metric in metrics:
            if metric in df_runs.columns:
                aggregated[f"{metric}_mean"] = df_runs[metric].mean()
                aggregated[f"{metric}_std"] = df_runs[metric].std()
                aggregated[f"{metric}_min"] = df_runs[metric].min()
                aggregated[f"{metric}_max"] = df_runs[metric].max()
        
        # Boolean aggregation for lp_exited (percentage of runs where LP exited)
        if 'lp_exited' in df_runs.columns:
            aggregated['lp_exit_rate'] = df_runs['lp_exited'].mean()
        
        self.results.append(aggregated)
        
        print(f"\n✓ Completed {num_runs} runs for {amm_type}")
        print(f"   Mean price gap: {aggregated.get('avg_price_gap_mean', 0):.4f} ± {aggregated.get('avg_price_gap_std', 0):.4f}")
        print(f"   Mean arb profit: {aggregated.get('arb_profit_mean', 0):.2f} ± {aggregated.get('arb_profit_std', 0):.2f}")
        
        return aggregated


    def run_all_experiments_with_repetitions(self, num_runs=10):
        """Run complete experimental suite with multiple repetitions"""
        print(f"\n{'#'*60}")
        print(f"RUNNING FULL EXPERIMENTAL SUITE ({num_runs} runs each)")
        print(f"{'#'*60}\n")
        
        experiments = [
            ("baseline", self.run_baseline_experiment, {}),
            ("flash_crash", self.run_flash_crash_experiment, 
            {"crash_magnitude": -0.5}),
            ("whale_manipulation", self.run_whale_manipulation_experiment, {}),
            ("high_volatility", self.run_high_volatility_experiment, 
            {"volatility_multiplier": 5.0})
        ]
        
        for amm_type in ["uniswap", "constant_sum"]:
            print(f"\n{'*'*60}")
            print(f"TESTING {amm_type.upper()} AMM")
            print(f"{'*'*60}")
            
            for exp_name, exp_func, params in experiments:
                self.run_experiment_with_repetitions(
                    exp_func, 
                    amm_type, 
                    num_runs=num_runs,
                    **params
                )
        
        print(f"\n{'#'*60}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"Total runs: {len(self.results)} aggregated results")
        print(f"({'×'.join([str(num_runs), '4 scenarios', '2 AMMs'])} = {num_runs*4*2} individual runs)")
        print(f"{'#'*60}\n")
    
    def run_all_experiments(self):
        """Run complete experimental suite for both AMM types"""
        print(f"\n{'#'*60}")
        print(f"RUNNING FULL EXPERIMENTAL SUITE")
        print(f"{'#'*60}\n")
        
        experiments = [
            ("baseline", {}),
            ("flash_crash", {"crash_magnitude": -0.5}),
            ("whale_manipulation", {}),
            ("high_volatility", {"volatility_multiplier": 5.0})
        ]
        
        for amm_type in ["uniswap", "constant_sum"]:
            print(f"\n{'*'*60}")
            print(f"TESTING {amm_type.upper()} AMM")
            print(f"{'*'*60}")
            
            for exp_name, params in experiments:
                if exp_name == "baseline":
                    self.run_baseline_experiment(amm_type, **params)
                elif exp_name == "flash_crash":
                    self.run_flash_crash_experiment(amm_type, **params)
                elif exp_name == "whale_manipulation":
                    self.run_whale_manipulation_experiment(amm_type, **params)
                elif exp_name == "high_volatility":
                    self.run_high_volatility_experiment(amm_type, **params)
        
        print(f"\n{'#'*60}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"Total experiments run: {len(self.results)}")
        print(f"{'#'*60}\n")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to analyze. Run experiments first!")
            return
        
        df = pd.DataFrame(self.results)
        
        print(f"\n{'='*60}")
        print(f"COMPARATIVE ANALYSIS REPORT")
        print(f"{'='*60}\n")
        
        # Group by experiment type
        for exp_type in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp_type]
            
            print(f"\n{'─'*60}")
            print(f"EXPERIMENT: {exp_type.upper()}")
            print(f"Number of runs per AMM: {exp_df['num_runs'].iloc[0]}")
            print(f"{'─'*60}")
            
            comparison = exp_df.groupby('amm_type').agg({
                'avg_price_gap_mean': 'first',
                'avg_price_gap_std': 'first',
                'max_price_gap_mean': 'first',
                'price_stability_mean': 'first',
                'k_growth_percent_mean': 'first',
                'arb_profit_mean': 'first',
                'arb_profit_std': 'first',
                'arb_trades_mean': 'first',
                'lp_panic_events_mean': 'first',
                'lp_exit_rate': 'first'
            }).round(4)
            
            # Rename columns for clarity
            comparison.columns = [
                'Avg Price Gap (mean)',
                'Avg Price Gap (std)',
                'Max Price Gap',
                'Price Stability',
                'K Growth %',
                'Arb Profit (mean)',
                'Arb Profit (std)',
                'Arb Trades',
                'LP Panic Events',
                'LP Exit Rate'
            ]
            
            print(comparison)
            print()
        
        # Save AGGREGATED results to CSV
        csv_path = os.path.join(self.output_dir, "experiment_results_aggregated.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Aggregated results saved to: {csv_path}")
        
        # Save INDIVIDUAL runs to CSV (optional, for detailed analysis)
        if self.individual_runs:
            df_individual = pd.DataFrame(self.individual_runs)
            csv_individual_path = os.path.join(self.output_dir, "experiment_results_individual.csv")
            df_individual.to_csv(csv_individual_path, index=False)
            print(f"✓ Individual run results saved to: {csv_individual_path}")
        
        # Save to JSON for detailed analysis
        json_path = os.path.join(self.output_dir, "experiment_results_aggregated.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Detailed aggregated results saved to: {json_path}")
    
    def plot_comparative_analysis(self):
        """Generate comparative visualizations with error bars"""
        if not self.results:
            print("No results to plot. Run experiments first!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('AMM Comparative Analysis (mean ± std across runs)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Average Price Gap
        ax1 = axes[0, 0]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='avg_price_gap_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='avg_price_gap_std')
        pivot_mean.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax1.set_title('Average Price Gap', fontweight='bold')
        ax1.set_ylabel('USDC')
        ax1.set_xlabel('')
        ax1.legend(title='AMM Type')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Price Stability (lower is better)
        ax2 = axes[0, 1]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='price_stability_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='price_stability_std')
        pivot_mean.plot(kind='bar', ax=ax2, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax2.set_title('Price Stability (lower = better)', fontweight='bold')
        ax2.set_ylabel('Std Dev')
        ax2.set_xlabel('')
        ax2.legend(title='AMM Type')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. K Growth (fee accumulation)
        ax3 = axes[0, 2]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='k_growth_percent_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='k_growth_percent_std')
        pivot_mean.plot(kind='bar', ax=ax3, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax3.set_title('K Growth % (fee accumulation)', fontweight='bold')
        ax3.set_ylabel('Percent')
        ax3.set_xlabel('')
        ax3.legend(title='AMM Type')
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 4. Arbitrageur Profit
        ax4 = axes[1, 0]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='arb_profit_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='arb_profit_std')
        pivot_mean.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax4.set_title('Arbitrageur Profit', fontweight='bold')
        ax4.set_ylabel('USDC')
        ax4.set_xlabel('')
        ax4.legend(title='AMM Type')
        ax4.grid(axis='y', alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 5. LP Panic Events
        ax5 = axes[1, 1]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='lp_panic_events_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='lp_panic_events_std')
        pivot_mean.plot(kind='bar', ax=ax5, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax5.set_title('LP Panic Events', fontweight='bold')
        ax5.set_ylabel('Count')
        ax5.set_xlabel('')
        ax5.legend(title='AMM Type')
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Max Volatility
        ax6 = axes[1, 2]
        pivot_mean = df.pivot(index='experiment', columns='amm_type', 
                            values='max_volatility_mean')
        pivot_std = df.pivot(index='experiment', columns='amm_type', 
                            values='max_volatility_std')
        pivot_mean.plot(kind='bar', ax=ax6, color=['#3498db', '#e74c3c'], 
                        yerr=pivot_std, capsize=4, error_kw={'linewidth': 2})
        ax6.set_title('Max Volatility Experienced', fontweight='bold')
        ax6.set_ylabel('Percent')
        ax6.set_xlabel('')
        ax6.legend(title='AMM Type')
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "comparative_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparative plots saved to: {plot_path}")
        
        plt.show()
    
    def plot_heatmap_analysis(self):
        """Generate heatmap showing performance across dimensions"""
        if not self.results:
            print("No results to plot. Run experiments first!")
            return
        
        df = pd.DataFrame(self.results)
        
        # Select key metrics (use MEAN values)
        metrics = [
            'avg_price_gap_mean', 
            'price_stability_mean', 
            'k_growth_percent_mean', 
            'arb_profit_mean', 
            'lp_panic_events_mean'
        ]
        
        metric_labels = [
            'Avg Price Gap',
            'Price Stability',
            'K Growth %',
            'Arb Profit',
            'LP Panic Events'
        ]
        
        # Create pivot table for each metric
        fig, axes = plt.subplots(1, len(metrics), figsize=(22, 4))
        fig.suptitle('Performance Heatmaps: Uniswap vs Constant Sum (mean values)', 
                    fontsize=14, fontweight='bold')
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            pivot = df.pivot(index='experiment', columns='amm_type', values=metric)
            
            # Choose colormap based on metric
            if 'profit' in metric.lower() or 'growth' in metric.lower():
                cmap = 'RdYlGn'  # Red=bad, Green=good
            else:
                cmap = 'RdYlGn_r'  # Red=bad (high values), Green=good (low values)
            
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap=cmap, 
                    ax=axes[idx], cbar_kws={'label': label},
                    center=0 if 'profit' in metric.lower() or 'growth' in metric.lower() else None)
            axes[idx].set_title(label, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('')
        
        plt.tight_layout()
        
        # Save plot
        heatmap_path = os.path.join(self.output_dir, "performance_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {heatmap_path}")
        
        plt.show()

    def plot_uncertainty_analysis(self):
        """Plot showing variability across runs (error bars focus)"""
        if not self.results:
            print("No results to plot. Run experiments first!")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Uncertainty Analysis: Variability Across Runs', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price Gap with error bars
        ax1 = axes[0, 0]
        for amm in ['uniswap', 'constant_sum']:
            data = df[df['amm_type'] == amm]
            x = range(len(data))
            y = data['avg_price_gap_mean'].values
            yerr = data['avg_price_gap_std'].values
            
            ax1.errorbar(x, y, yerr=yerr, marker='o', markersize=8, 
                        capsize=5, capthick=2, linewidth=2,
                        label=amm.capitalize())
        
        ax1.set_xticks(range(len(data)))
        ax1.set_xticklabels(data['experiment'].values, rotation=45, ha='right')
        ax1.set_ylabel('Avg Price Gap (USDC)', fontweight='bold')
        ax1.set_title('Price Gap Uncertainty')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Arbitrageur Profit with error bars
        ax2 = axes[0, 1]
        for amm in ['uniswap', 'constant_sum']:
            data = df[df['amm_type'] == amm]
            x = range(len(data))
            y = data['arb_profit_mean'].values
            yerr = data['arb_profit_std'].values
            
            ax2.errorbar(x, y, yerr=yerr, marker='s', markersize=8, 
                        capsize=5, capthick=2, linewidth=2,
                        label=amm.capitalize())
        
        ax2.set_xticks(range(len(data)))
        ax2.set_xticklabels(data['experiment'].values, rotation=45, ha='right')
        ax2.set_ylabel('Arbitrageur Profit (USDC)', fontweight='bold')
        ax2.set_title('Profit Uncertainty')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Coefficient of Variation (CV = std/mean) for Price Gap
        ax3 = axes[1, 0]
        for amm in ['uniswap', 'constant_sum']:
            data = df[df['amm_type'] == amm]
            cv = (data['avg_price_gap_std'] / data['avg_price_gap_mean'].abs()) * 100
            
            ax3.bar(range(len(data)), cv, alpha=0.7, label=amm.capitalize())
        
        ax3.set_xticks(range(len(data)))
        ax3.set_xticklabels(data['experiment'].values, rotation=45, ha='right')
        ax3.set_ylabel('Coefficient of Variation (%)', fontweight='bold')
        ax3.set_title('Relative Variability (Price Gap)')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. LP Exit Rate across runs
        ax4 = axes[1, 1]
        pivot = df.pivot(index='experiment', columns='amm_type', values='lp_exit_rate')
        pivot.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'])
        ax4.set_title('LP Exit Rate (% of runs where LP exited)')
        ax4.set_ylabel('Exit Rate (0-1)', fontweight='bold')
        ax4.set_xlabel('')
        ax4.legend(title='AMM Type')
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # Save plot
        uncertainty_path = os.path.join(self.output_dir, "uncertainty_analysis.png")
        plt.savefig(uncertainty_path, dpi=300, bbox_inches='tight')
        print(f"✓ Uncertainty analysis saved to: {uncertainty_path}")
        
        plt.show()

def main():
    """Run complete experimental suite with repetitions"""
    
    # Initialize experiment runner
    runner = ExperimentRunner(master_seed=42, output_dir="experiments_output")
    
    # Ask user for number of repetitions
    print("\n" + "="*60)
    print("How many runs per experiment?")
    print("  • 1 = Quick test (not statistically significant)")
    print("  • 10 = Good balance (recommended)")
    print("  • 30 = High confidence (takes longer)")
    print("="*60)
    
    try:
        num_runs = int(input("Number of runs (default 10): ").strip() or "10")
        if num_runs < 1:
            num_runs = 10
    except ValueError:
        num_runs = 10
    
    print(f"\n✓ Will run {num_runs} repetitions per experiment")
    
    # Run all experiments with repetitions
    runner.run_all_experiments_with_repetitions(num_runs=num_runs)
    
    # Generate reports
    runner.generate_comparison_report()
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    runner.plot_comparative_analysis()
    runner.plot_heatmap_analysis()
    runner.plot_uncertainty_analysis()  # NEW!
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTAL SUITE COMPLETED")
    print(f"{'='*60}")
    print(f"Total individual runs: {num_runs * 4 * 2} (= {num_runs} × 4 scenarios × 2 AMMs)")
    print(f"Check '{runner.output_dir}' directory for:")
    print(f"  • experiment_results.csv")
    print(f"  • experiment_results.json")
    print(f"  • comparative_analysis.png")
    print(f"  • performance_heatmap.png")
    print(f"  • uncertainty_analysis.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
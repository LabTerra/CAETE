# profile_analyzer.py
import pstats
import sys

import numpy as np
import matplotlib.pyplot as plt

def analyze_profile(profile_file):
    """Analyze cProfile output file"""

    print(f"\n{'='*60}")
    print(f"PROFILE ANALYSIS: {profile_file}")
    print(f"{'='*60}")

    stats = pstats.Stats(profile_file)

    # Top 20 functions by cumulative time
    print("\nTop 20 Functions by Cumulative Time:")
    print("-" * 50)
    stats.sort_stats('cumulative').print_stats(20)

    # Top 20 functions by total time (self time)
    print("\nTop 20 Functions by Total Time (Self):")
    print("-" * 50)
    stats.sort_stats('tottime').print_stats(20)

    # Functions that call specific modules (e.g., numpy, multiprocessing)
    print("\nNumPy/NumPy-related calls:")
    print("-" * 30)
    stats.print_stats('numpy')

    print("\nMultiprocessing-related calls:")
    print("-" * 30)
    stats.print_stats('multiprocessing')

    print("\nCAETE module calls:")
    print("-" * 30)
    stats.print_stats('caete')


def visualize_profile_results(profile_file, title="Profile Analysis", top_n=15):
    """Generate a visualization of profiling results filtered by CAETE and NumPy modules"""
    try:
        import pstats

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Load stats
        p = pstats.Stats(profile_file)
        total_time = p.total_tt

        # Use the raw stats dict: key is (filename, lineno, funcname)
        # value is (ncalls, totcalls, tottime, cumtime, callers)
        stats = p.stats

        # Filter functions - only CAETE and NumPy related
        caete_funcs = []
        numpy_funcs = []

        for key, value in stats.items():
            filename, lineno, funcname = key
            ncalls, totcalls, tottime, cumtime, callers = value
            short_filename = filename.split('\\')[-1].split('/')[-1]

            # Check if it's a CAETE module
            is_caete = 'caete' in filename.lower() or 'hydro_caete' in filename.lower()
            # Check if it's NumPy related
            is_numpy = 'numpy' in filename.lower() or 'numpy' in funcname.lower()

            func_dict = {
                'funcname': funcname,
                'filename': short_filename,
                'lineno': lineno,
                'cumtime': cumtime,
                'tottime': tottime,
                'ncalls': ncalls,
                'pct_cum': (cumtime / total_time * 100) if total_time > 0 else 0,
                'pct_tot': (tottime / total_time * 100) if total_time > 0 else 0,
            }

            if is_caete:
                caete_funcs.append(func_dict)
            if is_numpy:
                numpy_funcs.append(func_dict)

        # Sort by total time (self time) as shown in the output
        caete_funcs.sort(key=lambda x: x['tottime'], reverse=True)
        numpy_funcs.sort(key=lambda x: x['tottime'], reverse=True)

        # Take top entries
        caete_top = caete_funcs[:top_n]
        numpy_top = numpy_funcs[:top_n]

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        def plot_module_analysis(ax_time, ax_calls, func_list, module_name, color_time, color_calls):
            """Helper to plot time and calls for a module"""
            if not func_list:
                ax_time.text(0.5, 0.5, f'No {module_name} functions found',
                           ha='center', va='center', fontsize=12)
                ax_calls.text(0.5, 0.5, f'No {module_name} functions found',
                            ha='center', va='center', fontsize=12)
                return

            labels = []
            tot_times = []
            cum_times = []
            calls = []

            for item in func_list:
                fname = item['funcname']
                if len(fname) > 20:
                    fname = fname[:17] + "..."
                label = f"{fname}\n({item['filename']}:{item['lineno']})"
                labels.append(label)
                tot_times.append(item['tottime'])
                cum_times.append(item['cumtime'])
                calls.append(item['ncalls'])

            y_pos = np.arange(len(labels))

            # Time plot - show both self time and cumulative time
            colors_tot = plt.colormaps[color_time](np.linspace(0.8, 0.4, len(labels)))
            colors_cum = plt.colormaps['Greys'](np.linspace(0.5, 0.3, len(labels)))

            # Self time (colored)
            bars_tot = ax_time.barh(y_pos - 0.2, tot_times, height=0.4, label='Self time',
                                   color=colors_tot, edgecolor='black', linewidth=0.5)
            # Cumulative time (grey, behind)
            bars_cum = ax_time.barh(y_pos + 0.2, cum_times, height=0.4, label='Cumulative',
                                   color=colors_cum, edgecolor='grey', linewidth=0.5, alpha=0.7)

            ax_time.set_yticks(y_pos)
            ax_time.set_yticklabels(labels, fontsize=8, fontfamily='monospace')
            ax_time.invert_yaxis()
            ax_time.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
            ax_time.set_title(f'{module_name} - Execution Time', fontsize=12, fontweight='bold', pad=8)
            ax_time.legend(loc='lower right', fontsize=9)

            # Add annotations
            max_time = max(max(cum_times), max(tot_times)) if cum_times else 0.001
            for i, (tot, cum) in enumerate(zip(tot_times, cum_times)):
                if tot > 0.0005:
                    ax_time.text(tot + max_time * 0.02, i - 0.2, f'{tot:.4f}s',
                               va='center', ha='left', fontsize=7, fontweight='bold')
                if cum > 0.0005 and cum != tot:
                    ax_time.text(cum + max_time * 0.02, i + 0.2, f'{cum:.4f}s',
                               va='center', ha='left', fontsize=7, color='grey')

            ax_time.set_xlim(0, max_time * 1.4)
            ax_time.spines['top'].set_visible(False)
            ax_time.spines['right'].set_visible(False)

            # Calls plot
            colors_calls = plt.colormaps[color_calls](np.linspace(0.8, 0.4, len(labels)))

            bars_calls = ax_calls.barh(y_pos, calls, height=0.6,
                                      color=colors_calls, edgecolor='black', linewidth=0.5)

            ax_calls.set_yticks(y_pos)
            ax_calls.set_yticklabels(labels, fontsize=8, fontfamily='monospace')
            ax_calls.invert_yaxis()
            ax_calls.set_xlabel('Number of Calls', fontsize=10, fontweight='bold')
            ax_calls.set_title(f'{module_name} - Call Count', fontsize=12, fontweight='bold', pad=8)

            # Use log scale if needed
            max_calls = max(calls)
            min_calls = min(c for c in calls if c > 0)
            if max_calls / min_calls > 50:
                ax_calls.set_xscale('log')
                ax_calls.set_xlabel('Number of Calls (log scale)', fontsize=10, fontweight='bold')

            # Add call count annotations
            for i, ncalls in enumerate(calls):
                if ncalls >= 1000:
                    call_str = f'{ncalls/1000:.1f}k'
                else:
                    call_str = f'{ncalls}'
                x_pos = ncalls * 1.15 if ax_calls.get_xscale() == 'log' else ncalls + max_calls * 0.03
                ax_calls.text(x_pos, i, call_str, va='center', ha='left', fontsize=7, fontweight='bold')

            ax_calls.spines['top'].set_visible(False)
            ax_calls.spines['right'].set_visible(False)

        # Plot CAETE module (top row)
        plot_module_analysis(axes[0, 0], axes[0, 1], caete_top, 'CAETE Module', 'Blues', 'Greens')

        # Plot NumPy module (bottom row)
        plot_module_analysis(axes[1, 0], axes[1, 1], numpy_top, 'NumPy', 'Oranges', 'Purples')

        # Main title
        fig.suptitle(f'{title}\nTotal runtime: {total_time:.3f}s | CAETE: {len(caete_funcs)} funcs | NumPy: {len(numpy_funcs)} funcs',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the figure
        output_path = profile_file.replace('.prof', '_profile.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Visualization saved to {output_path}")
        plt.close()

    except ImportError as e:
        print(f"Required package missing: {e}. Install with: pip install matplotlib")
    except Exception as e:
        import traceback
        print(f"Error generating visualization: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import os

    if len(sys.argv) > 1:
        profile_path = sys.argv[1]
        profile_name = os.path.basename(profile_path).replace('.prof', '')
        analyze_profile(profile_path)
        visualize_profile_results(profile_path, title=f"CAETE Profile: {profile_name}")
    else:
        # Analyze all profile files in current directory
        import glob
        profile_files = glob.glob("*.prof")
        if not profile_files:
            print("No .prof files found. Usage: python profile_analyzer.py <profile_file.prof>")
        for pf in profile_files:
            profile_name = os.path.basename(pf).replace('.prof', '')
            analyze_profile(pf)
            visualize_profile_results(pf, title=f"CAETE Profile: {profile_name}")
from dol import analyze_complexity
from dol import utils

def run_alife21_analysis(indir, csvdir, plotdir):
    for combined_complexity in [False, True]:
        analyze_complexity.main_box_plot(
            num_dim = 1,
            num_neurons = 2,
            analyze_sensors = True,
            analyze_brain = True,
            analyze_motors = False,
            combined_complexity = combined_complexity,
            only_part_n1n2 = True,
            input_dir = indir,
            csv_dir = csvdir,
            plot_dir = plotdir
        )
    
    line_plots_params = [
        {'sim_type': 'individuals', 'tse_max':2, 'combined_complexity':False},
        {'sim_type': 'generalists', 'tse_max':4, 'combined_complexity':False},
        {'sim_type': 'generalists', 'tse_max':6, 'combined_complexity':True},
        {'sim_type': 'specialists', 'tse_max':2, 'combined_complexity':False},
        {'sim_type': 'specialists', 'tse_max':3, 'combined_complexity':True}
    ]
    for params in line_plots_params:
        analyze_complexity.main_line_plot(
            num_dim = 1, 
            num_neurons = 2, 
            sim_type = params['sim_type'],
            analyze_sensors = True, 
            analyze_brain = True, 
            analyze_motors = False,
            tse_max = params['tse_max'],
            combined_complexity = params['combined_complexity'],
            only_part_n1n2 = True, 
            input_dir = indir,
            csv_dir = csvdir,
            plot_dir = plotdir
        )
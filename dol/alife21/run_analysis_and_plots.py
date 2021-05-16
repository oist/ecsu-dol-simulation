import os
import argparse
from dol import utils 
from dol.alife21 import analysis, plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Get analysis and plots for Alife21 paper'
    )
    parser.add_argument('--indir', type=str, default='data', help='Input directory path')
    parser.add_argument('--outdir', type=str, default='data/alife21', help='Output directory path')

    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir    

    utils.make_dir_if_not_exists(outdir)    
    csvdir = os.path.join(outdir, 'csv')
    plotdir = os.path.join(outdir, 'plots')
    utils.make_dir_if_not_exists(csvdir)
    utils.make_dir_if_not_exists(plotdir)
    analysis.run_alife21_analysis(indir, csvdir, plotdir)    
    plots.plot_alife21(indir, plotdir)    
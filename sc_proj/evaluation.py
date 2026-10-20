import os
import warnings
import argparse
from run_pipeline import evaluate

# Filter out the specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")

def process_file(models, datasets, config_path, base_path, setting, evaluation_metrics, save_folder, pid_percentage, do_hvg, hvg_size):
    data_config_path = os.path.join(config_path, 'data')
    
    # Run the pipeline    
    evaluate(models, datasets, data_config_path, base_path, setting, evaluation_metrics, save_folder, pid_percentage, do_hvg, hvg_size)
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process models and datasets for training.")

    # Add the arguments
    parser.add_argument('--models', nargs='+', required=True, help="List of models, e.g., model1 model2 model3")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of datasets, e.g., dataset1 dataset2 dataset3")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the configuration directory")
    parser.add_argument('--setting', type=str, required=True, help="Define OOD or PID settings")
    parser.add_argument('--base_path', type=str, required=True, help="Path to the save evaluation results")
    parser.add_argument('--evaluation_metrics', nargs='+', required=True, help="Methods to be used for evaluation")
    parser.add_argument('--save_folder', type=str, required=True, help="Path to the save folder")
    parser.add_argument('--pid_percentage', type=float, required=False, help="The percentage of the ood cell to be seen by the model")
    parser.add_argument('--do_hvg', action="store_true", help="Does evaulate on highly variable gene subset?")
    parser.add_argument('--hvg_size', type=int, default=None, help="Number of HVGs (default: 500 if --do_hvg is used).")

    # Parse the arguments
    args = parser.parse_args()

    if args.do_hvg and args.hvg_size is None:
        args.hvg_size = 500

    if args.hvg_size is not None and not args.do_hvg:
        parser.error("--hvg_size can only be used together with --do_hvg")

    # Call the process_file function with parsed arguments
    process_file(args.models, args.datasets, args.config_path, args.base_path, args.setting, args.evaluation_metrics, 
                args.save_folder, args.pid_percentage, args.do_hvg, args.hvg_size)

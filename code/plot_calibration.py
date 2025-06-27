import os
from pdb import set_trace


def plot_calibration_diagram(x_data, y_data, ECE, plot_title, plot_save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.hist(x_data[:-1], bins=x_data, weights=y_data, edgecolor='black', align='mid')

    plt.plot([0, 1], [0, 1], linestyle='--', color='red')

    # Ensure the origin is centered at (0, 0)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))

    # Add labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{plot_title}', fontsize=15)
    plt.text(0.02, 0.95, f'ECE = {ECE:.3f}', transform=plt.gca().transAxes,
             fontsize=20, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.savefig(plot_save_path, format='pdf')
    plt.close()


def read_data_from_file(input_file_path):
    """
    :param input_file_path:
    :return: a list of numpy arrays
    """
    import ast  # for safe evaluations
    import numpy as np

    # Initialize an empty list to hold the data
    loaded_data = []

    # Read the file
    with open(input_file_path, 'r') as f:
        for line in f:
            # Convert the string representation of the list back to a Python object
            item = ast.literal_eval(line.strip())

            converted_item = []
            if isinstance(item, list):
                for element in item:
                    if isinstance(element, list):
                        converted_item.append(np.array(element))
                    else:
                        converted_item.append(np.array([element]))

            else:
                converted_item.append(np.array(item))

            loaded_data.append(converted_item[0])

    return loaded_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    parser.add_argument("--output_probing_dir")  # better termed 'output_probing_dir' ?
    parser.add_argument("--num_bins", type=int, default=5)
    parser.add_argument("--plot_title_original")
    parser.add_argument("--ECE_original", type=float)
    parser.add_argument("--plot_title_calibrated")
    parser.add_argument("--ECE_calibrated", type=float)
    args = parser.parse_args()

    output_probing_model_id_dir = os.path.join(args.output_probing_dir, f'calibrated_{args.model_id}')
    avg_acc_per_bin_orig = read_data_from_file(os.path.join(output_probing_model_id_dir, 'avg_acc_per_bin_orig.txt'))
    avg_acc_per_bin_scaled = read_data_from_file(os.path.join(output_probing_model_id_dir, 'avg_acc_per_bin_scaled.txt'))
    confidence_bins = read_data_from_file(os.path.join(output_probing_model_id_dir, 'confidence_bins.txt'))
    calibration_plots_save_dir = os.path.join(output_probing_model_id_dir, 'calibration_plots')
    print(f'bins : {confidence_bins}, avg_acc_per_bin_orig, {avg_acc_per_bin_orig}, '
          f'avg_acc_per_bin_scaled : {avg_acc_per_bin_scaled}')
    os.makedirs(calibration_plots_save_dir, exist_ok=True)
    plot_calibration_diagram(x_data=confidence_bins, y_data=avg_acc_per_bin_orig,
                             plot_title=f'{args.plot_title_original}',
                             ECE=args.ECE_original,
                             plot_save_path=os.path.join(calibration_plots_save_dir, f'original.pdf'))

    plot_calibration_diagram(x_data=confidence_bins, y_data=avg_acc_per_bin_scaled,
                             plot_title=f'{args.plot_title_calibrated}',
                             ECE=args.ECE_calibrated,
                             plot_save_path=os.path.join(calibration_plots_save_dir, f'calibrated.pdf'))

    print(f'Calibration plots saved to {calibration_plots_save_dir}!')


if __name__ == '__main__':
    main()
from evaluation.datasets.zenodo import process_zenodo_data
from evaluation.datasets.denoising import process_denoising_data
from evaluation.datasets.pa_experiment import process_pa_experiment_data
from evaluation.datasets.scd_swfd import process_scd_swfd
from evaluation.datasets.msfd import process_msfd
from evaluation.datasets.mat_based import process_mat_dataset
from config.data_config import DATASETS

def evaluate(dataset, config, full_config, file_key=None, metric_type="all", test_mode=False):
    dataset_info = DATASETS[dataset]
    results = []

    print("In evaluate")


    if dataset == "zenodo":
        process_zenodo_data(dataset_info, results, metric_type)
    elif dataset == "denoising_data":
        process_denoising_data(dataset_info, results, metric_type)
    elif dataset == "pa_experiment_data":
        process_pa_experiment_data(dataset_info, results, metric_type)
    elif dataset == "MSFD":
        process_msfd(dataset, dataset_info, full_config, results, metric_type)
    elif dataset in ["SCD", "SWFD"]:
        print("Processing SCD or SWFD")
        process_scd_swfd(dataset, dataset_info, full_config, file_key, results, metric_type)
    elif dataset in ["mice", "phantom", "v_phantom"]:
        process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type)

    return results


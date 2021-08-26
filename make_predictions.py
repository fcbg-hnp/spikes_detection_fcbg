from preprocessing import load_raws_from_dir
from utils import parse_config_file, print_message
from sd_pipelines import naive, var1_svm, var1_abdt, full_pipeline_svm
from utils.postprocessing import generate_mrk_from_stim
from utils import suppress_stdout

methods_dict = {
    'naive': naive.predict,
    'var1_svm': var1_svm.predict,
    'var1_abdt': var1_abdt.predict,
    'full_pipeline_svm': full_pipeline_svm.predict
}

partial_fit_methods_dict = {
    'naive': naive.predict,
    'var1_svm': var1_svm.refit_svm_and_predict,
    'var1_abdt': var1_abdt.predict,
    'full_pipeline_svm': full_pipeline_svm.refit_svm_and_predict
}


def check_raws_sfreq(raws_list_):
    class SamplingFrequencyError(Exception):
        pass
    sfreq_init = raws_list_[0].info['sfreq']
    if len(raws_list_) > 1:
        for raw in raws_list_[1:]:
            if not raw.info['sfreq'] == sfreq_init:
                raise SamplingFrequencyError("All raws should have exactly the ")


if __name__ == '__main__':
    # parse config_file
    configs = parse_config_file(file_path='config.ini')
    # features computation configurations
    features_configs = {
        'time_params': dict(),
        'freq_params': {'sfreq': configs['sfreq']},
        'information_params': {'sfreq': configs['sfreq'], 'n_jobs': configs['n_jobs']},
        'dwt_params': {'n_jobs': configs['n_jobs']}
    }
    # load_data
    raws_list = load_raws_from_dir(configs['data_path'], configs['extension'], seizures=False)
    # list of initial sfreq
    init_sfreq_list = [raw.info['sfreq'] for raw in raws_list]
    # methods params
    if configs['refit_svm']:
        prediction_method = partial_fit_methods_dict[configs['method']]
    else:
        prediction_method = methods_dict[configs['method']]
    stim_pred_list = prediction_method(raws_list=raws_list, configs=configs, features_configs=features_configs)
    # convert stim to .mrk
    if generate_mrk_from_stim(stim_pred_list, init_sfreq_list, configs):
        print_message('Finished', 'subsection')

from preprocessing.resampling import *
import joblib, os
from imblearn.over_sampling import RandomOverSampler, SVMSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from functools import partial
from sklearn.utils.validation import check_is_fitted

if __name__ == '__main__':
    print(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_abdt.joblib'))
    model = joblib.load(os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_abdt.joblib'))

    model.steps.pop(1)
    model.steps.pop(1)


    # model.steps.insert(1, ['over2',
    #                        RandomOverSampler(random_state=42,
    #                                          sampling_strategy=partial(random_over_sampling_rate, orate=1.5,
    #                                                                    non_spikes_label=0))])
    # model.steps.insert(1, ['over1',
    #                        SMOTE(random_state=42,
    #                              sampling_strategy=partial(smote_over_sampling_rate, non_spikes_label=0))])
    for step in model.steps:
        try:
            print(check_is_fitted(step))
        except Exception as e:
            print(e)

    print(model)

    joblib.dump(model, filename=os.path.join(os.getcwd(), "", 'sd_pipelines', "", "models", "", 'var1_abdt.joblib'))
"""
Deconvolution on group of subjects
==================================

The `GroupResponseFitter`-object of `Nideconv` offers an easy way to fit the data of many subjects together
"""
##############################################################################
# well-specified model
# --------------------
# `nideconv.simulate` can simulate data from multiple subjects.

from nideconv import simulate

data, onsets, pars = simulate.simulate_fmri_experiment(n_subjects=8,
                                                       n_rois=2,
                                                       n_runs=2)
##############################################################################
# Now we have an indexed Dataframe, `data`, that contains time series for every 
# subject, run, and ROI:


print(data.head())

##############################################################################
# 
print(data.tail())


##############################################################################
# We also have onsets for every subject and run:
print(onsets.head())
##############################################################################
# 
##############################################################################
print(onsets.tail())


##############################################################################
# We can now use `GroupResponseFitter` to fit all the subjects with one object: 

from nideconv import GroupResponseFitter


# `concatenate_runs` means that we concatenate all the runs of a single subject
# together, so that we have to fit only a single GLM per subject.
# A potential advantage is that the GLM has less degrees-of-freedom
# compared to the amount of data, so that the esitmate are potentially more
# stable.
# A potential downside is that different runs might have different intercepts
# and/or correlation structure.
# Therefore, by default, the `GroupResponseFitter` does _not_ concatenate
# runs.
g_model = GroupResponseFitter(data,
                              onsets,
                              input_sample_rate=1.0,
                              concatenate_runs=False)


##############################################################################
# We use the `add_event`-method to add the events we are interested in. The `GroupResponseFitter`
# then automatically collects the right onset times from the `onsets`-object.
#
# We choose here to use the `Fourier`-basis set, with 9 regressors.
g_model.add_event('Condition A',
                  basis_set='fourier',
                  n_regressors=9)

g_model.add_event('Condition B',
                  basis_set='fourier',
                  n_regressors=9,
                  interval=[0, 20])

##############################################################################
# We can fit all the subjects at once using the `fit`-method

g_model.fit()

##############################################################################
# We can plot the mean timecourse across subjects

print(g_model.get_subjectwise_timecourses().head())
g_model.plot_groupwise_timecourses()

##############################################################################
# As well as individual time courses
print(g_model.get_conditionwise_timecourses())
g_model.plot_subject_timecourses()

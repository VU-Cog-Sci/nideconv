import seaborn as sns
import matplotlib.pyplot as plt


def plot_timecourses(tc,
                     plots='roi',
                     col='covariate',
                     row=None,
                     col_wrap=None,
                     hue='event type',
                     max_n_plots=40,
                     oversample=None,
                     extra_axes=True,
                     sharex=True,
                     sharey=False,
                     aspect=1.5,
                     col_order=None,
                     size=3,
                     legend=True,
                     *args,
                     **kwargs):

    
    facs = []
    idxs = []

    if plots is None:
        fac = sns.FacetGrid(tc,
                            col_wrap=col_wrap,
                            col=col,
                            row=row,
                            sharex=sharex,
                            sharey=sharey,
                            aspect=aspect,
                            col_order=col_order,
                            size=size)

        fac.map_dataframe(sns.tsplot,
                          time='time',
                          unit='subject',
                          condition=hue,
                          value='value',
                          color=sns.color_palette(),
                          *args,
                          **kwargs)

        facs.append(fac)
        idxs.append('')

    else:
        if len(tc[plots].unique()) > max_n_plots:
            raise Exception('Splitting over %s would mean more than %d plots!')


        for idx, plot_df in tc.groupby(plots):
            fac = sns.FacetGrid(plot_df,
                                col_wrap=col_wrap,
                                col=col,
                                row=row,
                                sharex=sharex,
                                sharey=sharey,
                                aspect=aspect,
                                size=size)

            fac.map_dataframe(sns.tsplot,
                              time='time',
                              unit='subj_idx',
                              condition=hue,
                              value='value',
                              color=sns.color_palette(),
                              *args,
                              **kwargs)
            facs.append(fac)
            idxs.append(idx)

    for fac, idx in zip(facs, idxs):
        if extra_axes:
            fac.map(plt.axhline, y=0, c='k', ls='--')
            fac.map(plt.axvline, x=0, c='k', ls='--')

        fac.fig.subplots_adjust(top=0.8)

        if len(facs) > 1:
            fac.fig.suptitle(idx, fontsize=16)

        if len(fac.col_names) == 1:
            fac.set_titles('')

        if legend:
            fac.add_legend()


    return facs

def plot_design_matrix(X, palette=None, vertical=True):
    
    X = X.copy()
    
    n_regressors = X.shape[1]
    palette = sns.color_palette(palette, n_colors=n_regressors)    
    
    if len(X.columns.get_level_values('covariate').unique()) == 1:
        X.columns = X.columns.droplevel('covariate')
    
    max_reg = (X.max() - X.min()).max()
    
    previous_level = X.columns[0][:-1]
    
    offsets = [0]
    for i in range(1, X.shape[1]):
        offset = offsets[i-1] + max_reg * 1.1 

        if X.columns[i][:-1] != previous_level:
            offset += max_reg * 1.1 
            previous_level = X.columns[i][:-1]
        
        offsets.append(offset)            

    for i in range(X.shape[1]):
        offset = offsets[i]
        plt.plot(X.iloc[:, i] + offset, X.index.get_level_values('time'), c=palette[i])
        
    plt.gca().invert_yaxis()
    plt.ylabel('time (s)')
    
    labels = [', '.join(l) for l in X.columns]
    
    plt.xticks(offsets, labels, rotation='vertical')
    
    sns.despine()

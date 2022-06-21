import seaborn as sns
import matplotlib.pyplot as plt


def plot_timecourses(tc,
                     plots='roi',
                     col='covariate',
                     row=None,
                     col_wrap=None,
                     hue='event type',
                     max_n_plots=40,
                     oversample=1,
                     extra_axes=True,
                     sharex=True,
                     sharey=False,
                     aspect=1.75,
                     col_order=None,
                     height=3,
                     legend=True,
                     unit='subject',
                     *args,
                     **kwargs):

    if len(tc[plots].unique()) > max_n_plots:
        raise Exception('Splitting over %s would mean more than %d plots!')

    facs = []

    n_plots = len(tc[plots].unique())

    for idx, plot_df in tc.groupby(plots):
        fac = sns.FacetGrid(plot_df,
                            col_wrap=col_wrap,
                            col=col,
                            row=row,
                            sharex=sharex,
                            sharey=sharey,
                            aspect=aspect,
                            col_order=col_order,
                            height=height)

        fac.map_dataframe(sns.lineplot,
                          x='time',
                          hue=hue,
                          y='value',
                          # palette=sns.color_palette(),
                          *args,
                          **kwargs)

        if extra_axes:
            fac.map(plt.axhline, y=0, c='k', ls='--')
            fac.map(plt.axvline, x=0, c='k', ls='--')

        fac.fig.subplots_adjust(top=0.8)

        if n_plots > 1:
            fac.fig.suptitle(idx, fontsize=16)

        if len(fac.col_names) == 1:
            fac.set_titles('')

        if legend:
            fac.add_legend()

        if len(fac.col_names) == 1:
            fac.set_titles('')

        facs.append(fac)

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
        plt.plot(X.iloc[:, i] + offset,
                 X.index.get_level_values('time'), c=palette[i])

    plt.gca().invert_yaxis()
    plt.ylabel('time (s)')

    labels = [', '.join(l) for l in X.columns]

    plt.xticks(offsets, labels, rotation='vertical')

    sns.despine()

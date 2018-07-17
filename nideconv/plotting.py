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

    
    if len(tc[plots].unique()) > max_n_plots:
        raise Exception('Splitting over %s would mean more than %d plots!')

    facs = []

    multiple_plots = len(tc[plots].unique()) > 1

    for idx, plot_df in tc.groupby(plots):
        fac = sns.FacetGrid(plot_df,
                            col_wrap=col_wrap,
                            col=col,
                            row=row,
                            sharex=sharex,
                            sharey=sharey,
                            aspect=aspect,
                            col_order=col_order)

        fac.map_dataframe(sns.tsplot,
                          time='time',
                          unit='subject',
                          condition=hue,
                          value='value',
                          color=sns.color_palette(),
                          *args,
                          **kwargs)

        if extra_axes:
            fac.map(plt.axhline, y=0, c='k', ls='--')
            fac.map(plt.axvline, x=0, c='k', ls='--')

        fac.fig.subplots_adjust(top=0.8)
        fac.fig.suptitle(idx, fontsize=16)


        if len(fac.col_names) == 1:
            fac.set_titles('')

        facs.append(fac)

        if legend:
            fac.add_legend()

    return facs

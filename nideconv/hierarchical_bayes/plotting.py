import matplotlib.pyplot as plt
import seaborn as sns
from .utils import get_hpd

def plot_hpd(data, alpha=0.05, transparency=0.1, melted=None, *args, **kwargs):

    if melted is None:
        melted = data.columns.names[-1] != 't'
    
    hpd = get_hpd(data, alpha=alpha)   
    plt.fill_between(hpd.index.get_level_values('t'), hpd.values[:, 0], hpd.values[:, 1],alpha=transparency, *args, **kwargs)
    
    if melted:
        plt.plot(data.groupby(['t']).value.mean())
    else:
        mean_course = data.mean()
        plt.plot(mean_course.index.get_level_values('t'), mean_course.values, *args, **kwargs)
        
    sns.despine()
    plt.xlabel('Time (s)')

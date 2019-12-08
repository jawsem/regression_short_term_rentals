from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from model_building_helpers import summarize_dist
import matplotlib.patches as mpatches
import numpy as np
def autolabel(rects, ax, loc = 1.02, perc = True, ha = 'center'):
    # attach some text labels

    for rect in rects:
        height = rect.get_height()
        if perc:
            bar_label = '%1.1f' % float(height) + "%"
        else:
            bar_label = '%1.0f' % float(height)
        ax.text(rect.get_x() + rect.get_width()/2., loc*height,
                bar_label,
                ha=ha, va='bottom', fontsize = 16)


def plot_dist_bar(df,column, ax = False, save_fig=None):

    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
    if not ax:
        fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.set_title("{} Distribution".format(column))
    value_counts = (df[column].value_counts()/df.shape[0]).sort_index()*100
    plot = ax.bar(x=value_counts.index,height=value_counts,width=1/(len(value_counts)))
    ax.set_ylabel("Percent of Customers")
    ax.yaxis.set_major_formatter(formatter)
    autolabel(plot,ax)
    if save_fig:
        fig.savefig(save_fig)
    if not ax:
        return fig,ax
    else:
        return ax
    
def plot_dist_compare(df,field,target_field,ax=False, save_fig = None, thresh=.001):
    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
    if not ax:
        fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.set_title("{} vs {} Distribution".format(field,target_field))
    distr = summarize_dist(df,field,target_field,thresh=thresh).sort_index()
    plot = ax.bar(x=np.arange(distr.shape[0]),height=distr['{}_dist'.format(field)]*100,width=1/distr.shape[0],color='lightblue')
    ax2 = ax.twinx()
    plot2 = ax2.plot(np.arange(distr.shape[0]),(distr['{}_dist'.format(target_field)]*100),color = 'orange')

    ax.set_ylabel("Overall Distribution")
    ax.set_xticks(np.arange(distr.shape[0]))
    ax.set_xticklabels(distr.index)
    for i in np.arange(distr.shape[0]):
        xy_tuple = tuple(plot2[0].get_xydata()[i])
        height = xy_tuple[1]
        x = xy_tuple[0]
        bar_label = '%1.1f' % float(height) + "%"
        ax2.text(x,height*1.05,bar_label,horizontalalignment='center')
    ax.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylabel('{} Rate'.format(target_field))
    ax2.set_ylim(0,100)
    autolabel(plot,ax)
    plt.show()
    if save_fig:
        fig.savefig(save_fig)
    if not ax:
        return fig,ax
    else:
        return ax

def compare_dists(df1, df2, tf1, tf2, labels, title = None, savefig=None,thresh=.001):
	formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
	fig,ax=plt.subplots(1,1,figsize=(10,10))
	if not title:
		ax.set_title("{} vs {} Distribution".format(tf1,tf2))
	else:
		ax.set_title(title)

	distr1 = (df1[tf1].value_counts()/df1.shape[0]).sort_index()*100
	distr2 = (df2[tf2].value_counts()/df2.shape[0]).sort_index()*100
	offset = (1/len(distr1))/2
	plot = ax.bar(np.arange(len(distr1))-offset,height=distr1,width=offset*2,color='lightblue')
	plot2 = ax.bar(np.arange(len(distr1))+offset,height=distr2,width=offset*2,color='orange')
	ax.set_xticks(np.arange(len(distr1)))
	ax.set_xticklabels(distr1.index)
	ax.yaxis.set_major_formatter(formatter)
	ax.set_ylabel("Percent of Policies")
	autolabel(plot,ax)
	autolabel(plot2,ax)
	if len(labels)==2:
		blue_patch = mpatches.Patch(color='lightblue', label=labels[0])
		orange_patch = mpatches.Patch(color='orange', label=label[1])
		ax.legend(handles=[blue_patch,orange_patch])
	else:
		print("Length of Patches does not match length of data no legend")
	if savefig:
		fig.savefig(save_fig)
	return ax

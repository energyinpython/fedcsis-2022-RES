import os
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyrepo_mcda.mcda_methods import SPOTIS
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import normalizations as norms


# Create dictionary class
class Create_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value

# SWARA weighting
def swara_weighting(s, info = ''):
    if info == '':
        list_of_years = [str(y) for y in range(2015, 2020)]
        df_swara = pd.DataFrame(index = list_of_years[::-1])
    else:
        list_of_crits = [str(y + 1) for y in range(len(s))]
        df_swara = pd.DataFrame(index = list_of_crits)
    df_swara['cp'] = s
    k = np.ones(len(s))
    q = np.ones(len(s))
    for j in range(1, len(s)):
        k[j] = s[j] + 1
        q[j] = q[j - 1] / k[j]

    df_swara['kp'] = k
    df_swara['vp'] = q
    df_swara['wp'] = q / np.sum(q)
    df_swara.to_csv('results_res/swara_results' + info + '.csv')

    return q / np.sum(q)


# Functions for result visualizations
def plot_barplot_weights(df_plot, legend_title):
    """
    Visualization method to display column chart of alternatives rankings obtained with 
    different methods.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different methods.
            The particular rankings are included in subsequent columns of DataFrame.
        title : str
            Title of the legend (Name of group of explored methods, for example MCDA methods or Distance metrics).
    
    Examples
    ----------
    >>> plot_barplot(df_plot, legend_title='MCDA methods')
    """

    ax = df_plot.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel('Criteria', fontsize = 14)
    ax.set_ylabel('Weight value', fontsize = 14)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 14)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=5, mode="expand", borderaxespad=0., edgecolor = 'black', title = legend_title, fontsize = 14)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('./results_res/' + 'bar_chart_' + legend_title + '_weights.pdf')
    plt.show()



# chart bar
def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display stacked column chart of weights for criteria for `x_name == Weighting methods`
    and column chart of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights calculated different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of chart title, Weighting methods or Criteria

    Examples
    ----------
    >>> plot_barplot(df_plot, x_name, y_name, title)
    """
    
    list_rank = np.arange(1, len(df_plot) + 2, 2)
    stacked = True
    width = 0.6
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
        ncol = 2
    else:
        ncol = 5
    
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (10,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=ncol, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 11)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('results_res/bar_chart_' + title[-4:] + '.pdf')
    plt.show()


# chart scatter
def plot_scatter(data, model_compare, year):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        data: dataframe
        model_compare : list[list]

    Examples
    ----------
    >>> plot_scatter(data. model_compare)
    """

    #sns.set_style("darkgrid")
    list_rank = np.arange(0, 35, 5)
    list_alt_names = data.index
    for it, el in enumerate(model_compare):
        
        xx = [min(data[el[0]]), max(data[el[0]])]
        yy = [min(data[el[1]]), max(data[el[1]])]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(xx, yy, linestyle = '--', zorder = 1)

        ax.scatter(data[el[0]], data[el[1]], marker = 'o', color = 'royalblue', zorder = 2)
        for i, txt in enumerate(list_alt_names):
            ax.annotate(txt, (data[el[0]][i], data[el[1]][i]), fontsize = 14, style='italic',
                         verticalalignment='bottom', horizontalalignment='right')

        ax.set_xlabel(el[0], fontsize = 16)
        ax.set_ylabel(el[1], fontsize = 16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xticks(list_rank)
        ax.set_yticks(list_rank)

        x_ticks = ax.xaxis.get_major_ticks()
        y_ticks = ax.yaxis.get_major_ticks()

        ax.set_xlim(-1.5, len(data) + 2)
        ax.set_ylim(0, len(data) + 2)

        ax.grid(True, linestyle = '--')
        ax.set_axisbelow(True)

        plt.title('Year: ' + year, fontsize = 16)
    
        plt.tight_layout()
        plt.savefig('results_res/scatter_' + year + '.pdf')
        plt.show()


# chart scatter
def plot_rankings(results, year):
    """
    Display scatter plot comparing real and predicted ranking.

    Parameters
    -----------
        results : dataframe
            Dataframe with columns containing real and predicted rankings.

    Examples
    ---------
    >>> plot_rankings(results)
    """

    model_compare = []
    names = list(results.columns)
    model_compare = [[names[0], names[1]]]
    #sns.set_style("darkgrid")
    plot_scatter(data = results, model_compare = model_compare, year = year)


def main():
    
    path = 'dataset'
    # Number of countries
    m = 30
    n = 15

    str_years = [str(y) for y in range(2015, 2020)]
    list_alt_names_latex = [r'$A_{' + str(i + 1) + '}$' for i in range(0, m)]
    list_cols_latex = [r'$C_{' + str(j + 1) + '}$' for j in range(0, n)]
    preferences = pd.DataFrame(index = list_alt_names_latex)
    rankings = pd.DataFrame(index = list_alt_names_latex)

    # criteria weights
    df_weights = pd.DataFrame(index = list_cols_latex)

    method_name = 'SPOTIS'

    spotis = SPOTIS()

    for el, year in enumerate(str_years):
        file = 'RES_EU_' + str(year) + '.csv'
        pathfile = os.path.join(path, file)
        data = pd.read_csv(pathfile, index_col = 'Ai')
        
        # matrix
        matrix = data.to_numpy()
        # weights
        weights = mcda_weights.critic_weighting(matrix)
        
        df_weights[year] = weights
        # types: all criteria are profit type
        types = np.ones(data.shape[1])

        # SPOTIS preferences are sorted in ascending order
        bounds_min = np.amin(matrix, axis = 0)
        bounds_max = np.amax(matrix, axis = 0)
        bounds = np.vstack((bounds_min, bounds_max))

        pref = spotis(matrix, weights, types, bounds)
        rank = rank_preferences(pref, reverse = False)

        df_plot = pd.DataFrame(index = list_alt_names_latex)
        df_plot['SPOTIS'] = rank
        
        preferences[year] = pref
        rankings[year] = rank

    plot_barplot_weights(df_weights, 'Years')
    df_weights = df_weights.T
    df_weights.to_csv('results_res/critic_weights.csv')

    # SWARA weighting for determining periods' significance
    s = np.ones(len(str_years) - 1) * 0.5
    new_s = np.insert(s, 0, 0)
    swara_weights = swara_weighting(new_s)[::-1]

    # save SWARA weights to csv
    df_weights = pd.DataFrame(swara_weights.reshape(1, -1), index = ['Weights'], columns = str_years)
    df_weights.to_csv('results_res/weights_swara.csv')

    matrix_swara = preferences.to_numpy()
    
    # SPOTIS SWARA
    swara_types = np.ones(len(str_years)) * (-1)

    bounds_min = np.amin(matrix_swara, axis = 0)
    bounds_max = np.amax(matrix_swara, axis = 0)
    bounds = np.vstack((bounds_min, bounds_max))

    pref_swara = spotis(matrix_swara, swara_weights, swara_types, bounds)
    rank_swara = rank_preferences(pref_swara, reverse = False)


    # save results
    preferences['Temporal'] = pref_swara
    rankings['Temporal'] = rank_swara
    
    preferences = preferences.rename_axis('Ai')
    preferences.to_csv('results_res/scores_' + method_name + '.csv')

    rankings = rankings.rename_axis('Ai')
    rankings.to_csv('results_res/rankings_' + method_name + '.csv')
    
    # correlations
    method_types = list(rankings.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    dict_new_heatmap_rs = copy.deepcopy(dict_new_heatmap_rw)

    # heatmaps for correlations coefficients
    for i in method_types[::-1]:
        for j in method_types:
            dict_new_heatmap_rw[j].append(corrs.weighted_spearman(rankings[i], rankings[j]))
            dict_new_heatmap_rs[j].append(corrs.spearman(rankings[i], rankings[j]))

    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    df_new_heatmap_rs = pd.DataFrame(dict_new_heatmap_rs, index = method_types[::-1])
    df_new_heatmap_rs.columns = method_types

    df_new_heatmap_rw.to_csv('results_res/df_new_heatmap_rw_' + method_name + '.csv')
    df_new_heatmap_rs.to_csv('results_res/df_new_heatmap_rs_' + method_name + '.csv')



if __name__ == '__main__':
    main()
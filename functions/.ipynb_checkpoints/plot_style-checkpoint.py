# myproject/plot_style.py
import seaborn as sns
import matplotlib.pyplot as plt

PALETTE = sns.color_palette("colorblind")

#colors_by_resolution = {
#    '4e-3x': ['#fee0d2', '#deebf7'],
#    '2x':    ['#fc9272', '#9ecae1'],
#    '1x':    ['#de2d26', '#3182bd']
#}

colors_figures = {
'4e-3x': ['#fee0d2', '#deebf7'],
'2x':    ['#fc9272', '#9ecae1'],
'1x':    ['#de2d26', '#3182bd'],
'ann' : '#de2d26', 
'bms' : '#3182bd',
'data': sns.color_palette("colorblind")[7],
'noise': sns.color_palette("colorblind")[7]
}

text_size = {
    'axis':16,
    'ticks':14,
    'legend':14,
    'title':16
}

#Text sizes  
#--------------------------------
size_axis=16;size_ticks=14;
size_title=16;
line_w=3;marker_s=8;m_size=10
size_legend=14
#--------------------------------

def set_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.alpha": 0.3,
    })


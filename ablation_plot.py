import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


def save_bar_figures(data, x_names, colors, fig_name, n_groups=5):
    fig, ax1 = plt.subplots()  

    index = np.array(np.arange(n_groups))*2.5
    bar_width = 0.5
    parameters = {'axes.labelsize': 24,
            }
    plt.rcParams.update(parameters)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    for i in range(len(x_names)):
        ax1.bar(index[:5] + i *bar_width, data[i][:5], bar_width,
                alpha=opacity, color=colors[i],
                error_kw=error_config,
                label=x_names[i])
    
    # plt.axvline(x=index[5:]-bar_width, color='black', linestyle='dashed')  
    ax1.set_ylabel('Success Rate (%)', fontsize=18)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks(index + bar_width*2)
    ax1.set_xticklabels(('Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'), fontsize=14)
    ax1.legend(prop={'size':14},ncol=4,columnspacing=1.0, borderpad=0.1,
        labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
        loc='lower center' , bbox_to_anchor=(0.5, -0.1)
    )

    fig.set_size_inches(12, 8) 
    fig.tight_layout()
    plt.savefig(f'abl_figs/{fig_name}.pdf', dpi=300)
    plt.savefig(f'abl_figs/{fig_name}.png')

# n_groups = 6
if __name__ == "__main__":
    colors = ['#6fe7dd', '#3490de', '#cc99ff', '#6600cc', '#33ffff', '#35858B', '#072227']

    data = [
        (0.895, 0.704,0.498,0.388,0.284),
        (0.900,0.725,0.578,0.463,0.363),
        (0.895, 0.762, 0.628, 0.547, 0.455)
    ]

    x_names = ["no_vision", "lora", "full"]
    fig_name = "train_recipe"
    save_bar_figures(data, x_names, colors, fig_name)
    exit(0)

    data = [
        (0.804, 0.572, 0.396, 0.272, 0.202),
        (0.895, 0.762, 0.628, 0.547, 0.455)
    ]

    x_names = ["single", "two-stage",]
    fig_name = "multi_stage"
    
    save_bar_figures(data, x_names, colors, fig_name)
    exit(0)

    data = [
        (0.804, 0.572, 0.396, 0.272, 0.202),
        (0.770, 0.463, 0.224, 0.104, 0.036),
        (0.855, 0.605, 0.400, 0.320, 0.220),
        (0.895, 0.762, 0.628, 0.547, 0.455)
    ]

    x_names = ["base", "img-pred", "vid-cap", "vl-pred"]
    fig_name = "cotrain_task"
    data = [
        (0.903, 0.635, 0.397, 0.284, 0.174),
        (0.895, 0.762, 0.628, 0.547, 0.455),
        (0.675, 0.238, 0.088, 0.025, 0.013),
        (0.883, 0.618, 0.380, 0.256, 0.174),
        (0.825, 0.638, 0.425, 0.325, 0.238)
    ]
    x_names = ["llava-disc", "llava-ph", "llava-inter", "qwen-disc", "qwen-ph"]
    fig_name = "input_formulation"
    save_bar_figures(data, x_names, colors, fig_name)
    exit(0)
    
    MLP_nohist=(52.1, 12.4, 3.2, 0.9, 0.2, 0.688)
    MLP_hist=(74.3, 37.8, 16.4, 5.3, 1.9, 1.357)
    GPT=(96.0, 88.2, 80.7, 74.1, 67.1, 4.06)
    LSTM=(96.4, 89.6, 82.4, 74.0, 66.2, 4.086)

    data = [MLP_nohist, MLP_hist, GPT, LSTM]
    colors = ['#6fe7dd', '#3490de', '#cc99ff', '#6600cc', '#33ffff', '#35858B', '#072227']
    x_names = ['MLP w/o hist', 'MLP w hist', 'GPT', 'LSTM']
    fig_name = 'multi_action'
    save_bar_figures(data, x_names, colors, fig_name)
    exit(0)
    # task1 = (52.1, 74.3, 96.0, 96.4)
    # task2 = (12.4, 37.8, 88.2, 89.6)
    # task3 = (3.2, 16.4, 80.7, 82.4)
    # task4 = (0.9, 5.3, 74.1, 74.0)
    # task5 = (0.2, 1.9, 67.1, 66.2)
    # avg_len = (0.688, 1.357, 4.06, 4.0896)

    fig, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  

    index = np.array(np.arange(n_groups))*2.5
    bar_width = 0.5
    parameters = {'axes.labelsize': 24,
            }
    plt.rcParams.update(parameters)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    #"""
    rects1 = ax1.bar(index[:5], MLP_nohist[:5], bar_width,
                    alpha=opacity, color='#6fe7dd',
                    error_kw=error_config,
                    label='MLP w/o hist')
    rects2 = ax1.bar(index[:5] + bar_width, MLP_hist[:5], bar_width,
                    alpha=opacity, color='#3490de',
                    error_kw=error_config,
                    label='MLP w hist')
    rects3 = ax1.bar(index[:5] + 2*bar_width, GPT[:5], bar_width,
                    alpha=opacity, color='#cc99ff',
                    error_kw=error_config,
                    label='GPT')
    rects4 = ax1.bar(index[:5] + 3*bar_width, LSTM[:5], bar_width,
                    alpha=opacity, color='#6600cc',
                    error_kw=error_config,
                    label='LSTM')


    rects11 = ax2.bar(index[5:], MLP_nohist[5:], bar_width,
                    alpha=opacity, color='#6fe7dd',
                    error_kw=error_config,
                    label='MLP w/o hist')
    rects22 = ax2.bar(index[5:] + bar_width, MLP_hist[5:], bar_width,
                    alpha=opacity, color='#3490de',
                    error_kw=error_config,
                    label='MLP w hist')
    rects33 = ax2.bar(index[5:] + 2*bar_width, GPT[5:], bar_width,
                    alpha=opacity, color='#cc99ff',
                    error_kw=error_config,
                    label='GPT')
    rects44 = ax2.bar(index[5:] + 3*bar_width, LSTM[5:], bar_width,
                    alpha=opacity, color='#6600cc',
                    error_kw=error_config,
                    label='LSTM')

    plt.axvline(x=index[5:]-bar_width, color='black', linestyle='dashed')  

    # ax1.set_xlabel('Task')
    ax1.set_ylabel('Success Rate (%)', fontsize=18)
    ax2.set_ylabel('Average Achieved Task Length', fontsize=18)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks(index + bar_width*2)
    # ax1.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    # ax1.set_yticklabels(['0', '20', '40', '60', '80', '100', '', '', ''])
    ax1.set_xticklabels(('Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Avg. Len.'), fontsize=14)
    ax1.legend(prop={'size':14},ncol=4,columnspacing=1.0, borderpad=0.1,
        labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
        loc='lower center' , bbox_to_anchor=(0.5, -0.1)
        )
    # ax2.legend(prop={'size':14},ncol=1,columnspacing=1.0, borderpad=0.1,
    #     labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
    #     loc='lower right', bbox_to_anchor=(2, 0.5)
    #     )

    fig.set_size_inches(8, 6) 
    fig.tight_layout()
    plt.savefig('multi_action.pdf', dpi=300)
    plt.savefig('multi_action.png')

    exit(0)

    no_finetune = (33.6, 9.7, 1.8, 0.2, 0.0, 0.45)
    no_pretrain=(73.3, 44.4, 27.0, 16.1, 7.7, 1.69)
    full=(95.5, 87.9, 78.4, 71.4, 63.4, 3.97)
        

    # task1 = (52.1, 74.3, 96.0, 96.4)
    # task2 = (12.4, 37.8, 88.2, 89.6)
    # task3 = (3.2, 16.4, 80.7, 82.4)
    # task4 = (0.9, 5.3, 74.1, 74.0)
    # task5 = (0.2, 1.9, 67.1, 66.2)
    # avg_len = (0.688, 1.357, 4.06, 4.0896)
    #072227
    #35858B
    #4FBDBA
    #AEFEFF
    fig, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  

    index = np.array(np.arange(n_groups))*2.5
    bar_width = 0.5
    parameters = {'axes.labelsize': 24,
            }
    plt.rcParams.update(parameters)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    #"""
    rects1 = ax1.bar(index[:5], no_finetune[:5], bar_width,
                    alpha=opacity, color='#33ffff',
                    error_kw=error_config,
                    label='No VL Finetune')
    rects2 = ax1.bar(index[:5] + bar_width, no_pretrain[:5], bar_width,
                    alpha=opacity, color='#35858B',
                    error_kw=error_config,
                    label='No VL Pretrain')
    rects3 = ax1.bar(index[:5] + 2*bar_width, full[:5], bar_width,
                    alpha=opacity, color='#072227',
                    error_kw=error_config,
                    label='Full')

    rects1 = ax2.bar(index[5:], no_finetune[5:], bar_width,
                    alpha=opacity, color='#33ffff',
                    error_kw=error_config,
                    label='No VL Finetune')
    rects2 = ax2.bar(index[5:] + bar_width, no_pretrain[5:], bar_width,
                    alpha=opacity, color='#35858B',
                    error_kw=error_config,
                    label='No VL Pretrain')
    rects3 = ax2.bar(index[5:] + 2*bar_width, full[5:], bar_width,
                    alpha=opacity, color='#072227',
                    error_kw=error_config,
                    label='Full')

    plt.axvline(x=index[5:]-bar_width, color='black', linestyle='dashed')  

    # ax1.set_xlabel('Task')
    ax1.set_ylabel('Success Rate (%)', fontsize=18)
    ax2.set_ylabel('Average Achieved Task Length', fontsize=18)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks(index + bar_width*2)
    # ax1.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    # ax1.set_yticklabels(['0', '20', '40', '60', '80', '100', '', '', ''])
    ax1.set_xticklabels(('Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Avg. Len.'), fontsize=14)
    ax1.legend(prop={'size':14},ncol=4,columnspacing=1.0, borderpad=0.1,
        labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
        loc='lower center' , bbox_to_anchor=(0.5, -0.1)
        )
    # ax2.legend(prop={'size':14},ncol=1,columnspacing=1.0, borderpad=0.1,
    #     labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
    #     loc='lower right', bbox_to_anchor=(2, 0.5)
    #     )

    fig.set_size_inches(8, 6) 
    fig.tight_layout()
    plt.savefig('train_abl.pdf', dpi=300)
    plt.savefig('train_abl.png')

    exit(0)

    no_train = (70.8, 45.9, 27.3, 15.6, 8.5, 1.68)
    retrain=(82.6, 61.9, 42.7, 28.9, 17.8, 2.34)
    close_loop=(96.4, 89.6, 82.4, 74.0, 66.2, 4.086)

    fig, ax1 = plt.subplots()  
    ax2 = ax1.twinx()  

    index = np.array(np.arange(n_groups))*2.5
    bar_width = 0.5
    parameters = {'axes.labelsize': 24,
            }
    plt.rcParams.update(parameters)
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    #"""
    #219C90
    #E9B824
    #EE9322
    #D83F31
    rects1 = ax1.bar(index[:5], no_train[:5], bar_width,
                    alpha=opacity, color='#E9B824',
                    error_kw=error_config,
                    label='No Retrain')
    rects2 = ax1.bar(index[:5] + bar_width, retrain[:5], bar_width,
                    alpha=opacity, color='#D83F31',
                    error_kw=error_config,
                    label='Retrain')
    rects3 = ax1.bar(index[:5] + 2*bar_width, close_loop[:5], bar_width,
                    alpha=opacity, color='#219C90',
                    error_kw=error_config,
                    label='Close Loop')

    rects1 = ax2.bar(index[5:], no_train[5:], bar_width,
                    alpha=opacity, color='#E9B824',
                    error_kw=error_config,
                    label='No Retrain')
    rects2 = ax2.bar(index[5:] + bar_width, retrain[5:], bar_width,
                    alpha=opacity, color='#D83F31',
                    error_kw=error_config,
                    label='Retrain')
    rects3 = ax2.bar(index[5:] + 2*bar_width, close_loop[5:], bar_width,
                    alpha=opacity, color='#219C90',
                    error_kw=error_config,
                    label='Close Loop')

    plt.axvline(x=index[5:]-bar_width, color='black', linestyle='dashed')  

    # ax1.set_xlabel('Task')
    ax1.set_ylabel('Success Rate (%)', fontsize=18)
    ax2.set_ylabel('Average Achieved Task Length', fontsize=18)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks(index + bar_width*2)
    # ax1.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
    # ax1.set_yticklabels(['0', '20', '40', '60', '80', '100', '', '', ''])
    ax1.set_xticklabels(('Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Avg. Len.'), fontsize=14)
    ax1.legend(prop={'size':14},ncol=4,columnspacing=1.0, borderpad=0.1,
        labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
        loc='lower center' , bbox_to_anchor=(0.5, -0.1)
        )
    # ax2.legend(prop={'size':14},ncol=1,columnspacing=1.0, borderpad=0.1,
    #     labelspacing=0.1, handlelength=0.7, handletextpad=0.3, borderaxespad=-0.3,
    #     loc='lower right', bbox_to_anchor=(2, 0.5)
    #     )

    fig.set_size_inches(8, 6) 
    fig.tight_layout()
    plt.savefig('open_loop.pdf', dpi=300)
    plt.savefig('open_loop.png')
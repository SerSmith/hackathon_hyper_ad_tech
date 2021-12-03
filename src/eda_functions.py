import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

def nan_estimation_graph(df, 
                         autosize=True,
                         fig_width=8,
                         fig_height=8,
                         x_ticks_step=10, 
                         nan_values=None,
                         size_coef=0.3):
    """Строит диаграмму пропущенных значений
    """
    df = df.copy()
    title_part = 'NaN'
    if nan_values is not None:
        replace_dict = {value: np.nan for value in nan_values}
        df = df.replace(replace_dict)
        title_part = f"{nan_values + ['NaN']}"
    
    if df.isna().sum().sum() > 0:
        nan_df = pd.DataFrame()
        nan_df['percent'] = (100*df.isna().sum()/len(df)).sort_values()
        nan_df['count'] = df.isna().sum().sort_values()
        nan_df = nan_df[nan_df['count'] > 0]

        nan_percent_max = nan_df['percent'].max()

        colors = (nan_df['percent']  / nan_percent_max ).tolist()
        #подготовка градиентной окраски
        colors = [(color, 0.5*(1-color), 0.5*(1-color)) for color in colors]

        if autosize:
            fig_height = int(nan_df.shape[0]*size_coef) + 2
        
        figsize=(fig_width, fig_height)

        plt.figure(figsize=figsize)
        plt.grid(alpha=0.8)
        plt.xticks(range(0, 100+x_ticks_step, x_ticks_step))
        plt.xlim(0, nan_percent_max + 5)
        plt.xlabel(f'% {title_part}')
        plt.ylim(-1, len(nan_df))
        plt.ylabel('признак')

        xpos = nan_df['percent'] + nan_percent_max*0.02

        bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.8))

        for x, y, txt in zip( xpos, nan_df.index, nan_df['count'] ):
            plt.text(x, y, f'{txt} шт.', verticalalignment='center', bbox=bbox)

        plt.hlines(y=nan_df.index, xmin = 0, xmax = nan_df['percent'], alpha=0.7, 
                   linewidth=10, colors=colors)
        plt.title(f'Оценка количества и доли (%) {title_part} в данных\nВсего записей: {len(df)}, из них с {title_part}:', 
                  size=14)
        plt.tight_layout()
        plt.show()
    else:
        print(f'В наборе данных нет {title_part} значений')


def show_label_balance(df, label='is_defected', figsize=(8,4)):
    """Строит барплот с количеством уникальных значений label
    """
    print(df[label].value_counts())
    plt.figure(figsize=figsize)
    sns.countplot(x=df[label], alpha=0.7)
    plt.title('label balance')
    plt.show()


def plot_confusion_matrix(y_true, y_predicted, class_names=None, figsize=(8,6)):
    # Get and reshape confusion matrix data
    matrix = metrics.confusion_matrix(y_true, y_predicted)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    # Build the plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        annot_kws={'size':10},
        cmap=plt.get_cmap('Blues'),
        linewidths=0.2)

    if class_names is not None:
        # Add labels to the plot
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=25)
        plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
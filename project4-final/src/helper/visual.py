import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import List, Tuple, Dict

sns.set()

def plot_error_terms(
    y: np.array, 
    y_pred: np.array, 
    title: str
) -> plt.figure:
    """
    Plots the error terms
    """
    # Plot the error terms
    errors = (y_pred-y)

    df_errors = pd.DataFrame(
        np.array([np.arange(1, errors.shape[0]+1), y_pred, errors]).T,
        columns = ["Index","Predicted Value","Error"]
    )
    
    fig = plt.figure(figsize=(12,7))
    ax = plt.gca()
    sns.scatterplot(data=df_errors, x="Predicted Value", y="Error", ax=ax)
    ax.axhline(y=errors.mean())
    # ax.axvline(x=35, color="red")
    plt.title(title)
    
    return fig

def plot_confusion_matrix(
    cf_matrix:np.array,
    confusion_matrix_labels:List[str],
    figsize:Tuple[int, int] = (6,4),
) -> plt.figure:
    """
    Plot the confusion matrix
    """
    fig = plt.figure(figsize=figsize)

    ax = sns.heatmap(cf_matrix, annot=True, fmt='', cmap='Blues')

    ax.set_title('Confusion Matrix\n\n')
    ax.set_xlabel('\nPredicted Beer Category')
    ax.set_ylabel('Actual Beer Category ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(confusion_matrix_labels)
    ax.yaxis.set_ticklabels(confusion_matrix_labels)

    return fig

def plot_feature_importance(
    feature_importance_arr: np.array, 
    feature_names: np.array, 
    figsize:Tuple[int, int] = (8,30),
    topk: int = 50
) -> plt.figure:
    """
    Plots the error terms
    """
    # Plot the error terms
    df_feature_importances = pd.DataFrame(
        np.array([feature_names, feature_importance_arr]).T,
        columns=["Feature", "Importance"]
    )
    
    df_feature_importances["Importance"] = df_feature_importances["Importance"].astype(int)

    df_feature_importances = df_feature_importances.sort_values("Importance", ascending=False)
    
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=df_feature_importances.head(topk), x="Importance", y="Feature")
    plt.title("Feature Importances")
    
    return fig

def plot_bar_chart(
    df,
    x:str,
    y:str,
    ylabel:str,
    xlabel:str,
    huecol:str,
    title:str,
    figsize:Tuple[int, int] = (8,4),
    palette:Dict[str, str] = {
        "regular":"orange",
        "light":"green",
        "strong":"red"
    }
):
    """
    Helper function to plot bar chart by percentage
    """
    df_plot = df[[y,x]] \
        .value_counts() \
        .reset_index() \
        .sort_values(y) \
        .copy()

    df_plot.columns = [y, x, "beer_count"]

    df_plot = df_plot.join(
        df_plot.groupby(y).agg({"beer_count":np.sum}).rename(columns={"beer_count":"total_beer"}),
        how="left",
        on=y
    ).reset_index(drop=True)

    df_plot["category_percentage"] = df_plot["beer_count"]/df_plot["total_beer"]*100

    df_plot = df_plot.rename(columns={x:huecol})

    fig = plt.figure(figsize=figsize)
    sns.barplot(data=df_plot, y=y, x="category_percentage", hue=huecol, palette=palette)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()

def plot_word_cloud(
    df:pd.DataFrame,
    col:str,
    width:int=800,
    height:int=800,
    figsize:Tuple[int, int] = (8,8),
    words_to_exclude:set = set()
):
    """
    Helper function to plot wordcloud
    """

    # Import here because this function is not needed for training
    from wordcloud import WordCloud, STOPWORDS

    comment_words = ''
    stopwords = set(STOPWORDS).union(words_to_exclude)
    
    # iterate through the csv file
    for val in df[col].unique():
        
        # typecaste each val to string
        val = str(val)
    
        # split the value
        tokens = val.split()
        
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = width, height = height,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)
    
    # plot the WordCloud image                      
    plt.figure(figsize = figsize, facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    plt.show()

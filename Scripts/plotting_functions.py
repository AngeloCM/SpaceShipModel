import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import probplot


def create_bar_count_plot(
    data: pd.DataFrame,
    x: str,
    hue: str,
    title1: str,
    title2: str,
    ylabel: str,
    rotate: bool,
) -> None:
    """
    Create bar and count plots for a given dataset.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data to plot.
    x (str): The column name for the x-axis.
    hue (str): The column name for grouping in the count plot.
    title1 (str): The title for the first subplot (bar plot).
    title2 (str): The title for the second subplot (count plot).
    xlabel (str): The label for the x-axis of the first subplot.
    ylabel (str): The label for the y-axis of the first subplot.

    Returns:
    None
    """
    f, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the bar plot
    grouped_data = data.groupby(x, observed=False)[x].count()
    ax0 = sns.barplot(x=grouped_data.index, y=grouped_data.values, ax=ax[0])
    add_percentage_to_plot(ax0)
    ax[0].set_title(title1)
    ax[0].set_ylabel(ylabel)

    # Plot the count plot
    ax1 = sns.countplot(x=x, hue=hue, data=data, ax=ax[1])
    add_percentage_to_plot(ax1)
    ax[1].set_title(title2)

    if rotate:
        plt.setp(ax[0].get_xticklabels(), rotation=45)
        plt.setp(ax[1].get_xticklabels(), rotation=45)

    plt.show()


def create_barplot(
    data: pd.DataFrame, title: str, xlabel: str, ylabel: str
) -> plt.Axes:
    """
    Creates a bar plot using Seaborn with the specified data and plot attributes.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be plotted.
        title (str): The title of the bar plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        ax (matplotlib.axes.Axes): The Axes object containing the bar plot.
    """
    ax = sns.barplot(data=data)
    sns.despine()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax


def create_barplot_hue(
    data: pd.DataFrame, x: str, y: str, hue: str, title: str, xlabel: str, ylabel: str
):
    """
    Creates a bar plot using Seaborn with the specified data and plot attributes, incorporating hue.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be plotted.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        hue (str): The column name to differentiate bars by color.
        title (str): The title of the bar plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.

    Returns:
        None
    """
    ax = sns.barplot(x=x, y=y, hue=hue, data=data)
    sns.despine()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return ax


def create_heatmap(data: np.ndarray, mask: np.ndarray, title: str) -> plt.Axes:
    """
    Generate a heatmap to visualize the given data.

    Parameters:
        data (array-like): The 2D array or DataFrame containing the data.
        mask (array-like): The mask for data.
        title (str): Title of the heatmap.

    Returns:
        ax (matplotlib.axes.Axes): The Axes object containing the heatmap.
    """
    ax = sns.heatmap(
        data,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title(title)
    plt.show()
    return ax


def add_percentage_to_plot(ax: plt.Axes):
    """
    Adds percentage labels to each bar in a bar plot.

    Parameters:
        ax (matplotlib.axes.Axes): The Axes object containing the bar plot.

    Returns:
        None
    """
    total_height = sum(bar.get_height() for bar in ax.patches if bar.get_height() > 0)

    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            percentage = height / total_height
            if percentage > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.5,
                    f"{percentage:.1%}",
                    ha="center",
                    va="bottom",
                )


def plot_distributions(group1: pd.Series, group2: pd.Series, title1: str, title2: str):
    """
    Plot distributions and Q-Q plots for visual inspection.

    Parameters:
        group1 (pd.Series): A Pandas Series containing data for group 1.
        group2 (pd.Series): A Pandas Series containing data for group 2.
        title1 (str): Title for the distribution plot of group 1.
        title2 (str): Title for the distribution plot of group 2.

    Returns:
        None
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(group1, kde=True, ax=axs[0, 0])
    axs[0, 0].set_title(title1)
    sns.histplot(group2, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title(title2)

    probplot(group1, dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title(f"{title1} Q-Q Plot")
    probplot(group2, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title(f"{title2} Q-Q Plot")

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plot_size = (14, 6)

def plot_config():
    """
    Show plot and apply tight layout.
    """
    #plt.show()
    plt.tight_layout()

def horizontal_kde_box_plot(df: pd.DataFrame, x: str, hue: str) -> None:
    """
    Plot horizontal kde plot.
    Parameters:
        df: Data dataframe.
        x:  Target feature.
        hue:  Target feature.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plot_size)

    sns.boxplot(data=df, x=x, hue=hue, ax=ax1)
    sns.kdeplot(data=df, x=x, hue=hue, ax=ax2)

    ax1.set_title(f"{x} Distribution Over {hue}")
    ax1.set_xlabel(f"{x}")
    ax2.set_title(f"{x} Denstity")
    ax2.set_xlabel(f"{x}")
    ax2.set_xlim(df[x].min() - 0.1)

    plot_config()

def plot_percentage_bars_categorical(
        data: pd.DataFrame, x_feature: str, target_feature: str, max_values=3
) -> None:
    counts = data[x_feature].value_counts()

    # If the number of unique categories is less than or equal to max_values,
    # create a list of all categories
    if len(counts) <= max_values:
        top_categories = counts.index.tolist()
        top_counts = counts.tolist()
    else:
        # Get the top categories and sum others
        top_categories = counts.nlargest(max_values).index.tolist()
        other_count = counts[~counts.index.isin(top_categories)].sum()

        # Create a new DataFrame for plotting with top categories plus 'Others'
        top_counts = counts[top_categories].tolist()
        top_counts.append(other_count)
        top_categories.append('Others')

    plot_data = pd.DataFrame({
        x_feature: top_categories,
        'count': top_counts
    })

    # Ensure x_feature is not a categorical type to allow adding 'Others'
    if pd.api.types.is_categorical_dtype(data[x_feature]):
        data[x_feature] = data[x_feature].astype(str)

    merged_data = data.copy()
    if len(counts) > max_values:
        merged_data[x_feature] = merged_data[x_feature].where(
            merged_data[x_feature].isin(top_categories[:-1]), 'Others'
        )

    # Group by the new x_feature and target_feature to calculate counts
    feature_target = merged_data.groupby(
        [x_feature, target_feature]).size().reset_index(name="count")

    # Calculate total counts for each category in feature_target
    feature_total = feature_target.groupby(x_feature)['count'].sum().reset_index(name="total")

    feature_target = feature_target.merge(feature_total, on=x_feature)
    feature_target["percent"] = (feature_target["count"] / feature_target["total"]) * 100

    plt.figure(figsize=plot_size)

    # If the new categories do not match the prepared plot_data, merge to keep consistent categories
    plot_data = plot_data.merge(
        feature_target[[x_feature, 'percent']],
        on=x_feature,
        how='left').fillna(0)

    feature_target = feature_target.sort_values(by='count', ascending=False)

    ax = sns.barplot(
        data=feature_target,
        x=x_feature,
        y='percent',
        hue=target_feature,
        order=feature_target[x_feature])

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", label_type="edge")

    plt.ylabel("Percentage")
    plt.xlabel(f"{x_feature}")
    plt.title(f"Percentage of {target_feature} within Each {x_feature} Group")
    plot_config()


def cm_matrix(cm: np.ndarray, place: int, model_name: str, axes: np.ndarray) -> None:
    """
    Plots a confusion matrix heatmap.

    Parameters:
        cm: Confusion matrix (2D array-like).
        place: Index of the subplot position.
        model_name: Name of the model for the title.
        axes: Array of axes from plt.subplots.
    """
    # Flatten the axes to handle 2D grid
    flat_axes = axes.flatten()

    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".3g", ax=flat_axes[place], cbar=True)
    flat_axes[place].set_title(f"{model_name}")
    flat_axes[place].set_xlabel("Predicted Label")
    flat_axes[place].set_ylabel("True Label")

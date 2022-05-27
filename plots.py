import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from librosa import display

sns.set_theme(style="whitegrid")


def _palette(x): return sns.color_palette("Blues_d", n_colors=x, desat=0.6)


def load_metadata(path):
    return pd.DataFrame(path)


def class_imbalance(df):
    plt.clf()
    palette = _palette(10)
    palette.reverse()

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x=df["class"].value_counts().index, y=df["class"].value_counts() / len(df) * 100,
                palette=palette)
    plt.xlabel("Sounds")
    plt.ylabel("Distribution (relative)")
    plt.title("Relative Distribution - Class Labels")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    return fig, ax


def classes_per_fold(df):
    plt.clf()
    palette = _palette(10)
    palette.reverse()

    df = df[["fold", "class"]]
    cross = pd.crosstab(df["fold"], df["class"])

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(cross, cmap="GnBu")

    ax.set(ylabel="Fold",
           xlabel="Classes")
    sns.despine(left=True, bottom=True)

    plt.title("Distribution of classes over folds")
    return fig, ax


def duration(df):
    df["len"] = df["end"] - df["start"]

    # bin the len column into the following bins: 0-1, 1-2, 2-3, 3-4
    df["len_bin"] = pd.cut(df["len"], bins=[0, 1, 2, 3, 4], labels=["(0 - 1]", "(1 - 2]", "(2 - 3]", "(3 - 4]"])

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x=["(2 - 3]", "(0 - 1]", "(1 - 2]", "(3 - 4]"], y=df["len_bin"].value_counts().sort_values(), color="b")
    ax.set(ylabel="Total observations",
           xlabel="Length interval in seconds",
           title="Binned length of sound excerpts")
    sns.despine(left=True, bottom=True)


def waveplots(paths, names):
    arrays = []
    for path in paths:
        arr, sr = librosa.load(path, mono=True, sr=8000)
        arrays.append(arr)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle("Example waveforms of classes", fontsize=16)
    for i, sample in enumerate(arrays):
        ax = fig.add_subplot(5, 2, i + 1)
        ax.set_title(names[i])
        librosa.display.waveshow(sample, sr=8000)

    plt.tight_layout()

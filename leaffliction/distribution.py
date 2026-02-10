#!/usr/bin/env python3

import os
import sys
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt


class Dataset:
    """
    A class representing a dataset for analysing.

    Attributes:
        dataset (dict[str, list[str]]):
            key - path to a directory, value - list of valid images

        chart_data (dict[str, int]):
            key - directory name, value - amount of valid images
    """


    def __init__(self, dataset: dict[str, list[str]],
                chart_data: dict[str, int]):
        """
        Initialize an Dataset object.

        Parameters:
            dataset (dict[str, list[str]]):
                key - path to a directory, value - list of valid images

            chart_data (dict[str, int]):
                key - directory name, value - amount of valid images
        """
        self.dataset = dataset
        self.chart_data = chart_data


    def show_charts(self):
        """
        Show pie and bar charts depending on a dataset.
        """

        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(self.chart_data.values(), labels=self.chart_data.keys(),
                    autopct='%1.2f%%')
        ax_pie.set_title('Plant types')
        fig_pie.canvas.manager.set_window_title('Pie Chart')

        fig_bar, ax_bar = plt.subplots()
        colors = plt.cm.tab10(range(len(self.chart_data)))
        ax_bar.bar(self.chart_data.keys(), self.chart_data.values(),
                    color=colors)
        ax_bar.set_ylabel('Amount of images')
        ax_bar.set_title('Plant types')
        fig_bar.canvas.manager.set_window_title('Bar Chart')

        plt.show()


def is_valid_image(filepath):
    """
    Check if image is valid.

    Args:
        filepath (str): path to a file

    Returns:
        bool: True if image is valid, False otherwise
    """

    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception:
        return False


def load_dataset(root_dir) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Fetches images in a given directory and its subdirectories.

    Args:
        root_dir (str): root directory

    Returns:
        tuple[dict[str, list[str]], dict[str, int]]:
            dataset and chart_data
    """

    dataset: dict[str, list[str]] = {}
    chart_data: dict[str, int] = {}

    image_extensions = \
        {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                if is_valid_image(join(root, file)):
                    dir_name = os.path.basename(root).replace("_", " ")

                    if root not in dataset:
                        dataset[root] = []
                        chart_data[dir_name] = 0
                    dataset[root].append(file)
                    chart_data[dir_name] += 1

    return dataset, chart_data


if __name__ == "__main__":
    if (len(sys.argv) != 2):
        sys.exit("Usage: ./Distribution.py <directory>")

    root_dir = sys.argv[1]
    if (not root_dir):
        sys.exit("Directory path is invalid")

    dataset_dict, chart_data = load_dataset(root_dir)
    dataset = Dataset(dataset_dict, chart_data)
    dataset.show_charts()

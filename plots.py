import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ModelPlot:

    def __draw_mean_plot(self, df: pd.DataFrame, path: str) -> str:
        df = df.drop(columns=["name"])
        mean_values = df.mean()
        plot_path = os.path.join(path, "mean_plot.png")

        plt.figure(figsize=(10, 6))

        plt.plot(mean_values.index, mean_values.values)

        plt.xlabel("Columns")
        plt.ylabel("Mean Values")
        plt.title("Mean Values of Different Columns")
        plt.yticks(list(mean_values.values))

        plt.tight_layout()
        plt.savefig(plot_path)
        return plot_path

    def __draw_max_plot(self, df: pd.DataFrame, path: str) -> str:
        df = df.drop(columns=["name"])
        plot_path = os.path.join(path, "max_plot.png")
        max_values = df.max()

        plt.figure(figsize=(10, 6))

        plt.plot(max_values.index, max_values.values)

        plt.xlabel("Columns")
        plt.ylabel("Max Values")
        plt.title("Max Values of Different Columns")

        plt.tight_layout()

        plt.savefig(plot_path)
        return plot_path

    def __draw_histogram(self, df: pd.DataFrame, path: str) -> str:
        histogram_path = os.path.join(path, "histogram.png")
        deviation_columns = [
            "mean",
            "max",
            "min",
            "floor_mean",
            "floor_max",
            "floor_min",
            "ceiling_mean",
            "ceiling_max",
            "ceiling_min",
        ]
        num_cols = 3
        num_rows = (len(deviation_columns) + num_cols - 1) // num_cols

        plt.figure(figsize=(15, 10))

        for i, column in enumerate(deviation_columns, start=1):
            plt.subplot(num_rows, num_cols, i)
            plt.hist(df[column], bins=20, alpha=0.7)
            plt.title(column)
            plt.xlabel("Deviation (degrees)")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(histogram_path)
        return histogram_path

    def __draw_boxplots(self, df: pd.DataFrame, path: str) -> str:
        boxplots_path = os.path.join(path, "boxplots.png")
        deviation_columns = [
            "mean",
            "max",
            "min",
            "floor_mean",
            "floor_max",
            "floor_min",
            "ceiling_mean",
            "ceiling_max",
            "ceiling_min",
        ]

        plt.figure(figsize=(12, 6))

        df.boxplot(column=deviation_columns)

        plt.title("Boxplots of Deviations")
        plt.xlabel("Deviation Type")
        plt.ylabel("Deviation (degrees)")

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(boxplots_path)
        return boxplots_path

    def __draw_scatter(self, df: pd.DataFrame, path: str) -> str:
        scatter_path = os.path.join(path, "scatter.png")
        plt.figure(figsize=(8, 6))

        plt.scatter(df["gt_corners"], df["rb_corners"], color="blue", alpha=0.5)

        plt.title("Ground Truth vs. Predicted Corners")
        plt.xlabel("Ground Truth Corners")
        plt.ylabel("Predicted Corners")

        plt.grid(True)

        plt.tight_layout()
        plt.savefig(scatter_path)
        return scatter_path

    def __draw_heatmap(self, df: pd.DataFrame, path: str) -> str:
        heatmap_path = os.path.join(path, "heatmap.png")
        df = df.drop(columns=["name"])

        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")

        plt.tight_layout()
        plt.savefig(heatmap_path)
        return heatmap_path

    def draw_plots(self, df: pd.DataFrame) -> dict:
        path = os.path.join(os.curdir, "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        mean_plot = self.__draw_mean_plot(df, path)
        max_plot = self.__draw_max_plot(df, path)
        histogram_plot = self.__draw_histogram(df, path)
        boxplots_plot = self.__draw_boxplots(df, path)
        scatter_plot = self.__draw_scatter(df, path)
        heatmap_plot = self.__draw_heatmap(df, path)

        return {
            "mean_plot": mean_plot,
            "max_plot": max_plot,
            "histogram_plot": histogram_plot,
            "box_plot": boxplots_plot,
            "scatter_plot": scatter_plot,
            "heatmap_plot": heatmap_plot,
        }

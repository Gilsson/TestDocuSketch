import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plots import ModelPlot


if __name__ == "__main__":
    df = pd.read_json("deviation.json")
    print(df.loc[df["gt_corners"] != df["rb_corners"]])
    plot = ModelPlot()
    plot.draw_plots(df)

import pandas as pd

if __name__ == '__main__':
  df = pd.read_csv("../sample_data/modi_BPI_2012_dropna_filter_act.csv")
  del df["StartTimestamp"]
  df = df[:1000]
  df.to_csv("../sample_data/test.csv")
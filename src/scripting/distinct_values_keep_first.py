
import pandas as pd

def main():

    col1 = ["A", "B", "C"]
    col2 = ["1", "1", "3"]
    col3 = ["X", "X", "Z"]

    data = zip(col1, col2, col3)

    df = pd.DataFrame(data, columns=["first", "second", "third"])

    print(df.drop_duplicates(subset=["second", "third"], ignore_index=True))

if __name__ == "__main__":
    main()

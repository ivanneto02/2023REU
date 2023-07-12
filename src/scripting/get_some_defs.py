import pandas as pd

def main():
    df = pd.read_csv("/mnt/c/Users/ivana/Desktop/Documents/Research/UCR/DS-PATH/working_dir/data/ready/ready_to_embed.csv", nrows=None)
    print(df["definition"])

if __name__ == "__main__":
    main()
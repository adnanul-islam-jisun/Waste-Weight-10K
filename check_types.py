import pandas as pd

csv_path = "/home/asiful/adnan_workspace/Dataset/disaster_data/waste_dataset/image.csv"
df = pd.read_csv(csv_path)
print("Unique Types:", df['Type'].unique())
print("\nValue Counts:")
print(df['Type'].value_counts())

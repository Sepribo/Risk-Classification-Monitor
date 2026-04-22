

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_and_plot_classes(file_path):
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return "Error: File not found. Please check the path."

    # 2. Extract the last column as the class variable
    # .iloc[:, -1] selects all rows and the very last column
    class_name = df.columns[-1]
    classes = df.iloc[:, -1]

    print(f"Target variable identified: {class_name}")
    print(f"Total data points: {len(df)}")

    # 3. Count the occurrences of each class
    class_counts = classes.value_counts()
    print("\nData points per class:")
    print(class_counts)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create a bar plot
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")

    # Add labels and title
    plt.title(f'Distribution of Data Points by Class ({class_name})', fontsize=15)
    plt.xlabel('Class Label', fontsize=12)
    plt.ylabel('Number of Data Points', fontsize=12)

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.show()
    
    
process_and_plot_classes("C:/Users/user/Intelligent Monitoring System/preprocess1_dataset.csv")
import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd

def create_and_execute_eda_notebook():
    notebook_dir = os.path.join("notebooks", "data-pipeline")
    os.makedirs(notebook_dir, exist_ok=True)
    notebook_path = os.path.join(notebook_dir, "phase1_eda.ipynb")
    
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown Cell 1
    cells.append(nbf.v4.new_markdown_cell("# Phase 1: Exploratory Data Analysis\n\nThis notebook performs normal EDA including identifying categorical vs continuous features, and plotting their distributions."))
    
    # Code Cell 1: Setup
    setup_code = """\
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic parameters
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries imported successfully.")
"""
    cells.append(nbf.v4.new_code_cell(setup_code))
    
    # Code Cell 2: Load Data
    load_code = """\
# Load the transactions data
data_path = '../../data/original/transactions.csv'
df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape}")
df.head()
"""
    cells.append(nbf.v4.new_code_cell(load_code))
    
    # Code Cell 3: Identify Features
    features_code = """\
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Total Features: {len(df.columns)}")
print(f"Number of Categorical Features: {len(categorical_cols)}")
print(f"Number of Numerical Features: {len(numerical_cols)}")

print("\\nCategorical Features:")
for col in categorical_cols:
    print(f" - {col}")

print("\\nNumerical Features:")
for col in numerical_cols:
    print(f" - {col}")
"""
    cells.append(nbf.v4.new_code_cell(features_code))
    
    # Code Cell 4: Distribution of Numerical Features
    num_dist_code = """\
# Distribution Analysis for Numerical Features
for col in numerical_cols:
    # Skip columns that look like IDs or very large
    if df[col].nunique() > 1000 and 'id' in col.lower():
        print(f"Skipping {col} (looks like an ID)")
        continue
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df[col].dropna(), bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
"""
    cells.append(nbf.v4.new_code_cell(num_dist_code))
    
    # Code Cell 5: Distribution of Categorical Features
    cat_dist_code = """\
# Distribution Analysis for Categorical Features
for col in categorical_cols:
    unique_vals = df[col].nunique()
    if unique_vals > 50:
        print(f"Skipping {col} - too many unique values ({unique_vals})")
        continue
        
    plt.figure(figsize=(10, 5))
    val_counts = df[col].value_counts().head(20)  # Top 20 max
    sns.barplot(x=val_counts.index, y=val_counts.values, palette='viridis')
    plt.title(f'Distribution of {col} (Top 20)')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
"""
    cells.append(nbf.v4.new_code_cell(cat_dist_code))
    
    nb['cells'] = cells
    
    # Write unexecuted notebook first
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created notebook at {notebook_path}")

    # Execute notebook
    print("Executing notebook to generate plots (this may take a minute)...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': notebook_dir}})
        
        # Save executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print("Notebook executed and saved successfully.")
    except Exception as e:
        print(f"Failed to execute notebook: {e}")
        print("You can manually run the notebook inside Kaggle or Jupyter.")

if __name__ == "__main__":
    create_and_execute_eda_notebook()

import pandas as pd
import sys

def convert_textual_to_numerical(input_file: str, output_file: str) -> None:
    """
    Reads a CSV file, converts all textual (categorical/string) features to numerical values,
    and saves the updated DataFrame as a new CSV file.
    
    - Numeric string columns are automatically converted to int/float.
    - Purely textual/categorical columns are label-encoded (0, 1, 2, ...) using pd.factorize.
    - Missing values in categorical columns are treated as 'Missing' category.
    - The resulting CSV contains ONLY numerical values.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        print(f"✅ Successfully loaded: {input_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    # Process each column
    for col in df.columns:
        # Only process object/string columns (textual features)
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Step 1: Try to convert to numeric (handles "123", "45.6", etc.)
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            # Step 2: Check if ALL non-null values can be converted to numbers
            non_null = df[col].dropna()
            if len(non_null) > 0 and pd.to_numeric(non_null, errors='coerce').notna().all():
                # It was a numeric string column → convert safely
                df[col] = numeric_series
                print(f"   → Converted numeric strings: {col}")
            else:
                # Truly textual/categorical → label encode
                # Handle missing values gracefully
                df[col] = df[col].fillna('Missing').astype(str)
                # Convert to numerical codes (0, 1, 2, ...)
                df[col] = pd.factorize(df[col])[0]
                print(f"   → Label-encoded categorical: {col}")

    # Verify that all columns are now numeric
    if df.select_dtypes(include=['object', 'string']).empty:
        print("✅ All features are now numerical!")
    else:
        print("⚠️  Some columns could not be converted (check data).")

    # Save the updated CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"✅ Updated CSV saved successfully: {output_file}")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    print("=== CSV Textual-to-Numerical Converter ===\n")
    
    # Handle command-line arguments or interactive input
    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        input_path = input("Enter input CSV path (e.g., data.csv): ").strip()
        output_path = input("Enter output CSV path (e.g., data_numerical.csv): ").strip()
        if not output_path:
            # Auto-generate output name if user presses Enter
            output_path = input_path.replace('.csv', '_numerical.csv')
            if output_path == input_path:
                output_path = "output_numerical.csv"
    
    convert_textual_to_numerical(input_path, output_path)
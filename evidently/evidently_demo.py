# 1. Load the dataset
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

# Load the CSV data (ensure the file is in the working directory or provide the correct path)
df = pd.read_csv("evidently/online_shoppers_intention.csv")

# Check the shape and a quick preview of the data
print("Dataset shape:", df.shape)
print(df.columns.tolist())  # Print all column names
print(
    df[["Month", "VisitorType", "Weekend", "Revenue"]].head(5)
)  # Sample of key columns

# 2. Split data into reference (past) and current (present) sets based on Month
# Define first half and second half of the year
first_half_months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
]  # Months for reference period
second_half_months = [
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]  # Months for current period

# Filter the DataFrame by month
reference_df = df[df["Month"].isin(first_half_months)].copy()
current_df = df[df["Month"].isin(second_half_months)].copy()

print("Reference period size:", len(reference_df))
print("Current period size:", len(current_df))

# 3. Install and import Evidently for data drift analysis

# 4. Create a data drift report using Evidently
# Add warning suppression for the numpy warnings
import warnings
import numpy as np

# Suppress specific numpy warnings that occur during correlation calculation
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)

# Initialize and run the drift report
data_drift_report = Report(
    metrics=[DataDriftPreset()]
)  # initialize a Report with the Data Drift preset
my_eval = data_drift_report.run(
    reference_data=reference_df, current_data=current_df
)  # run the comparison
my_eval.save_html("file.html")

# 5. Extract drift information from the DataDriftPreset
# Since we know we're using a DataDriftPreset, we can directly access its properties
drift_preset = data_drift_report.metrics[0]

# Print information about available properties
print("\nDataDriftPreset Information:")
print(f"Metric type: {type(drift_preset)}")

# Since the API has changed and we don't have direct access to drift metrics,
# we'll need to examine what properties are available
drift_preset_attrs = [attr for attr in dir(drift_preset) if not attr.startswith("_")]
print(f"Available attributes: {', '.join(drift_preset_attrs)}")

# Extract whatever drift information we can find
try:
    # Attempt to extract column drift information
    column_count = 0
    drift_detected_count = 0

    # Look for column information in the metric
    if hasattr(drift_preset, "get_drift_by_columns"):
        drift_by_columns = drift_preset.get_drift_by_columns()
        column_count = len(drift_by_columns)
        drift_detected_count = sum(
            1
            for col_info in drift_by_columns.values()
            if col_info.get("drift_detected", False)
        )

    # If we can't find column drift info, look for other drift indicators
    if column_count == 0 and hasattr(drift_preset, "result"):
        result = drift_preset.result
        if hasattr(result, "dataset_drift"):
            overall_drift = result.dataset_drift
            if hasattr(result, "share_of_drifted_columns"):
                drift_share = result.share_of_drifted_columns
            else:
                drift_share = 0.0
    else:
        # Calculate overall drift and drift share
        overall_drift = drift_detected_count > 0
        drift_share = drift_detected_count / column_count if column_count > 0 else 0.0

except Exception as e:
    print(f"Error extracting drift metrics: {e}")
    # Default fallback values
    overall_drift = False
    drift_share = 0.0

print(f"Overall Data Drift Detected: {overall_drift}")
print(f"Drifted Features Proportion: {drift_share*100:.2f}%")


# In the latest Evidently API, the save_html method might have been renamed or replaced
# Let's try to save the HTML report using various methods that might be available
print("\nTrying to save HTML report...")

# Instead of attempting various imports which might not work, let's check what methods are available on our report object
available_methods = [
    method for method in dir(data_drift_report) if not method.startswith("_")
]
print(f"Available methods on Report object: {available_methods}")

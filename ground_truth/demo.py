import streamlit as st
import pandas as pd
import ast  # For safely evaluating string-formatted lists/dicts
import json  # For pretty printing JSON strings


# Helper function to safely parse string representations of lists/dicts
def parse_complex_string(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s  # Return as is if not parsable, or handle error


# Helper function to format event/request details for display
def format_for_display(data):
    if isinstance(data, (list, dict)):
        return json.dumps(data, indent=2)
    return str(data)


st.set_page_config(layout="wide")
st.title("Network Request Ground Truth Labeling Tool")

# --- Session State Initialization ---
if "labels" not in st.session_state:
    st.session_state.labels = {}
if "df_data" not in st.session_state:
    st.session_state.df_data = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


uploaded_file = st.sidebar.file_uploader("Upload your CSV candidate file", type=["csv"])

if uploaded_file is not None:
    # If a new file is uploaded, reset state
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_data = df
            st.session_state.labels = {}  # Reset labels for the new file

            # Determine the name of the last column for pre-existing labels.
            # Pandas might read an unnamed last column as 'Unnamed: X'.
            # If the CSV header is "col1,col2,col3," (trailing comma), the last column will be 'Unnamed: 3' (0-indexed).
            # Your CSV example had "...,occurance_rate,1", so the last column name was '1'.
            # The screenshot shows "Unnamed: 4", which means it's the 5th column and was unnamed in the header.

            # We'll assume the last column is the pre-existing label column.
            if df.shape[1] > 0:  # Check if there are any columns
                pre_existing_label_col = df.columns[-1]
                # Check if this column is suitable for labels (numeric, 0 or 1)
                if (
                    pd.api.types.is_numeric_dtype(df[pre_existing_label_col])
                    and df[pre_existing_label_col].dropna().isin([0, 1]).all()
                ):
                    st.sidebar.success(
                        f"Using column '{pre_existing_label_col}' for pre-existing labels."
                    )
                    # Initialize labels from this column, defaulting to 0 if NaN
                    st.session_state.labels = (
                        df[pre_existing_label_col].fillna(0).astype(int).to_dict()
                    )
                else:
                    st.sidebar.warning(
                        f"Last column ('{pre_existing_label_col}') doesn't seem to contain valid labels (0 or 1). Defaulting all to 'Not Relevant'."
                    )
                    st.session_state.labels = {i: 0 for i in range(len(df))}
            else:  # No pre-existing label column found or empty df
                st.sidebar.info(
                    "No pre-existing label column detected or empty DataFrame. Defaulting all to 'Not Relevant'."
                )
                st.session_state.labels = {i: 0 for i in range(len(df))}

            st.session_state.data_loaded = True
            st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

        except Exception as e:
            st.error(f"Error loading or processing CSV: {e}")
            st.session_state.data_loaded = False
            st.session_state.df_data = None


if st.session_state.data_loaded and st.session_state.df_data is not None:
    df = st.session_state.df_data

    # Display common Start and End event (assuming they are the same for the whole file)
    if not df.empty:
        st.header("Scenario Context")
        col1, col2 = st.columns(2)

        # Check if 'start_event' and 'end_event' columns exist
        start_event_col_exists = "start_event" in df.columns
        end_event_col_exists = "end_event" in df.columns

        with col1:
            st.subheader("Start Event")
            if start_event_col_exists:
                parsed_start_event = parse_complex_string(df["start_event"].iloc[0])
                st.json(
                    parsed_start_event
                    if isinstance(parsed_start_event, (list, dict))
                    else {"raw": str(parsed_start_event)}
                )
            else:
                st.warning("Column 'start_event' not found in CSV.")
        with col2:
            st.subheader("End Event")
            if end_event_col_exists:
                parsed_end_event = parse_complex_string(df["end_event"].iloc[0])
                st.json(
                    parsed_end_event
                    if isinstance(parsed_end_event, (list, dict))
                    else {"raw": str(parsed_end_event)}
                )
            else:
                st.warning("Column 'end_event' not found in CSV.")
        st.markdown("---")

    st.header("Label Network Requests")

    # Pagination
    items_per_page = st.sidebar.slider("Requests per page", 5, 50, 10)
    total_items = len(df)
    total_pages = (
        (total_items + items_per_page - 1) // items_per_page
        if items_per_page > 0
        else 0
    )

    if (
        total_pages == 0 and total_items > 0
    ):  # If items_per_page is 0 but there are items
        total_pages = 1
        items_per_page = total_items  # Show all items on one page

    if total_items == 0:
        st.warning("The CSV file seems to be empty or has no data rows.")
    else:
        current_page = st.sidebar.number_input(
            "Page", min_value=1, max_value=max(1, total_pages), value=1, step=1
        )

        start_idx = (current_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        # Check if 'network_request' and 'occurance_rate' columns exist
        network_request_col_exists = "network_request" in df.columns
        occurance_rate_col_exists = "occurance_rate" in df.columns

        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            occurrence_rate_display = (
                f"{row.get('occurance_rate', 'N/A'):.4f}"
                if occurance_rate_col_exists and pd.notna(row.get("occurance_rate"))
                else "N/A"
            )
            st.markdown(
                f"**Request {i+1} of {total_items}** (Occurrence Rate: {occurrence_rate_display})"
            )

            if network_request_col_exists:
                parsed_request = parse_complex_string(row["network_request"])
                if isinstance(parsed_request, list) and len(parsed_request) == 2:
                    url_path, domain = parsed_request
                    st.code(
                        f"URL Path: {url_path}\nDomain:   {domain}",
                        language="plaintext",
                    )
                else:
                    st.code(
                        f"Network Request (raw): {row['network_request']}",
                        language="plaintext",
                    )
            else:
                st.warning("Column 'network_request' not found in CSV for this row.")

            # Get current label for this index from session state
            current_label_value = st.session_state.labels.get(
                i, 0
            )  # Default to 0 if not found

            radio_options = ["Not Relevant", "Relevant"]
            default_radio_index = current_label_value

            label_choice_str = st.radio(
                "Is this request relevant to the event transition?",
                options=radio_options,
                index=default_radio_index,
                key=f"label_{st.session_state.current_file_name}_{i}",
                horizontal=True,
            )

            st.session_state.labels[i] = radio_options.index(label_choice_str)
            st.markdown("---")

    if not df.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("Export Labels")
        if st.sidebar.button("Prepare Labeled CSV for Download"):
            labeled_df = df.copy()
            labeled_df["ground_truth_label"] = (
                labeled_df.index.map(st.session_state.labels).fillna(0).astype(int)
            )

            st.subheader("Labeled Data Preview (first 100 rows)")
            st.dataframe(labeled_df.head(100))

            output_columns_present = []
            expected_output_columns = [
                "start_event",
                "end_event",
                "network_request",
                "occurance_rate",
            ]
            for col in expected_output_columns:
                if col in labeled_df.columns:
                    output_columns_present.append(col)
                else:  # Add missing columns with N/A if they were expected for structure
                    labeled_df[col] = "N/A"
                    # We won't add it to output_columns_present if it wasn't there originally,
                    # but it will exist in labeled_df for the next step.
            output_columns_present.append("ground_truth_label")

            # Prepare CSV for download using only columns that were originally present + ground_truth_label
            final_output_df = labeled_df[output_columns_present]

            csv_data = final_output_df.to_csv(index=False).encode("utf-8")

            download_file_name = (
                f"labeled_{st.session_state.current_file_name}"
                if st.session_state.current_file_name
                else "labeled_data.csv"
            )

            st.sidebar.download_button(
                label="Download Labeled CSV",
                data=csv_data,
                file_name=download_file_name,
                mime="text/csv",
            )
else:
    st.info("Upload a CSV file using the sidebar to start labeling.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by AI for labeling.")

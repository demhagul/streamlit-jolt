import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="JOLT Interim Report Tool",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #1E3A8A;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<p class="main-header">JOLT Interim Report Tool</p>', unsafe_allow_html=True)
st.markdown("""
    This tool helps transform and analyze adverse events data from the JOLT study. 
    Upload your CSV file to generate summary statistics and detailed event listings.
""")

# Define adverse events mapping
ADVERSE_EVENTS = {
    'Hypoxia': 'Hypoxia CTCAE grade (highest grade this cycle)?',
    'Atrial fibrillation': 'Atrial fibrillation CTCAE grade (highest grade this cycle)',
    'Atelectasis': 'Atelectasis CTC Adverse Event Grade (highest grade this cycle)',
    'Lung infection': 'Lung infection CTC Adverse Event Grade  (highest grade this cycle)',
    'Pneumonitis': 'Pneumonitis CTC Adverse Event Grade  (highest grade this cycle)',
    'Cough': 'Cough CTC Adverse Event Grade (highest grade this cycle)',
    'Pulmonary fibrosis': 'Pulmonary fibrosis CTC Adverse Event Grade  (highest grade this cycle)',
    'Chest wall pain': 'Chest wall pain CTC Adverse Event Grade  (highest grade this cycle)',
    'Fracture': 'Fracture CTC Adverse Event Grade (highest grade this cycle)'
}

def extract_attribution_number(attribution_value):
    """Extract the numeric part from attribution string (e.g., '1=Unrelated' -> 1)"""
    if pd.isna(attribution_value):
        return None
    try:
        # Extract first character and convert to integer
        return int(str(attribution_value)[0])
    except (ValueError, IndexError):
        return None

def transform_data(df):
    """Transform the raw data into one adverse event per row."""
    transformed_rows = []
    
    for _, row in df.iterrows():
        for event_name, grade_column in ADVERSE_EVENTS.items():
            grade = row[grade_column]
            
            # Only process if grade is 1-5
            if pd.notna(grade) and 1 <= grade <= 5:
                # Get the attribution column (next column after grade)
                cols = df.columns.tolist()
                grade_idx = cols.index(grade_column)
                attribution_column = cols[grade_idx + 1]
                attribution_value = row[attribution_column]
                
                # Extract numeric attribution
                attribution = extract_attribution_number(attribution_value)
                
                transformed_rows.append({
                    'Study ID': str(row['Study ID']),
                    'Event Name': str(row['Event Name']),
                    'Date of evaluation': str(row['Date of evaluation']) if pd.notna(row['Date of evaluation']) else '',
                    'Hospitalized': str(row['Was patient hospitalized during this reporting interval? ']) if pd.notna(row['Was patient hospitalized during this reporting interval? ']) else '',
                    'Adverse Event': str(event_name),
                    'Grade': int(grade),
                    'Attribution': attribution,
                    'Attribution Text': str(attribution_value) if pd.notna(attribution_value) else ''
                })
    
    df_transformed = pd.DataFrame(transformed_rows)
    # Ensure all column names are strings
    df_transformed.columns = df_transformed.columns.astype(str)
    return df_transformed

def create_summary_table(df, selected_grades, selected_attributions):
    """Create summary table of adverse events by grade."""
    # Filter data based on selections
    mask = (df['Grade'].isin(selected_grades)) & (df['Attribution'].isin(selected_attributions))
    filtered_df = df[mask]
    
    # Create pivot table
    summary = pd.pivot_table(
        filtered_df,
        values='Study ID',
        index='Adverse Event',
        columns='Grade',
        aggfunc='count',
        fill_value=0
    )
    
    # Ensure all grade columns exist
    for grade in range(1, 6):
        if grade not in summary.columns:
            summary[grade] = 0
    
    # Sort columns numerically
    summary = summary.reindex(sorted(summary.columns), axis=1)
    
    # Add total row and column
    summary['Total'] = summary.sum(axis=1)
    summary.loc['Total'] = summary.sum()
    
    # Convert column names to strings
    summary.columns = summary.columns.astype(str)
    
    return summary

# File uploader
uploaded_file = st.file_uploader("Upload JOLT CSV Data File", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    try:
        with st.spinner('Reading and processing data...'):
            df = pd.read_csv(uploaded_file)
            
            # Transform the data
            transformed_df = transform_data(df)
            
            # Success message
            st.success('Data processed successfully!')
            
            # Display data info
            st.markdown('<p class="sub-header">Data Overview</p>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", len(transformed_df['Study ID'].unique()))
            with col2:
                st.metric("Total Events", len(transformed_df))
            with col3:
                st.metric("Timepoints", len(transformed_df['Event Name'].unique()))
            
            # Filters
            st.markdown('<p class="sub-header">Filters</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                selected_grades = st.multiselect(
                    "Filter by Grade",
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5]
                )
            with col2:
                selected_attributions = st.multiselect(
                    "Filter by Attribution",
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3, 4, 5],
                    help="1=Unrelated, 2=Unlikely, 3=Possible, 4=Probable, 5=Definite"
                )
            
            # Create summary table
            if len(selected_grades) > 0 and len(selected_attributions) > 0:
                summary_df = create_summary_table(transformed_df, selected_grades, selected_attributions)
                
                # Display summary table
                st.markdown('<p class="sub-header">Summary Table</p>', unsafe_allow_html=True)
                st.dataframe(
                    summary_df.style.format("{:.0f}").apply(lambda x: ['background-color: #E5E7EB' if x.name == 'Total' else '' for i in x], axis=1),
                    use_container_width=True
                )
                
                # Filter the transformed data
                mask = (transformed_df['Grade'].isin(selected_grades)) & \
                       (transformed_df['Attribution'].isin(selected_attributions))
                filtered_df = transformed_df[mask].copy()
                
                # For display purposes, combine attribution number with text
                filtered_df['Attribution Display'] = filtered_df.apply(
                    lambda x: f"{x['Attribution']} ({x['Attribution Text']})" if pd.notna(x['Attribution Text']) else str(x['Attribution']),
                    axis=1
                )
                
                # Display preview of transformed data
                st.markdown('<p class="sub-header">Data Preview</p>', unsafe_allow_html=True)
                preview_df = filtered_df.drop(['Attribution Text'], axis=1)  # Remove the extra attribution column
                st.dataframe(
                    preview_df.head(10),
                    use_container_width=True
                )
                
                # Download section
                st.markdown('<p class="sub-header">Export Data</p>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    # Download transformed data
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Detailed Data",
                        data=csv,
                        file_name="jolt_adverse_events_detailed.csv",
                        mime="text/csv",
                        help="Download the complete filtered dataset with one event per row"
                    )
                with col2:
                    # Download summary table
                    summary_csv = summary_df.to_csv()
                    st.download_button(
                        label="Download Summary Table",
                        data=summary_csv,
                        file_name="jolt_adverse_events_summary.csv",
                        mime="text/csv",
                        help="Download the summary statistics table"
                    )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("JOLT Interim Report Tool v1.0")
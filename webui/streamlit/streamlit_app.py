import os
import streamlit as st

from datetime import datetime
from dotenv import load_dotenv
from llmetrics.langfuse_adaptor import LangfuseUtils, SensitivityConsistencyMetrics

st.set_page_config(layout="wide")

# Load environment variables from .env file: path is relative to streamlit_app.py
load_dotenv(dotenv_path='../../.env')

plots = st.Page("st_pages/plots.py", title="What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering", default=True)

pages = {
    "Plots": [plots],
}

with st.sidebar:
    with st.expander("### ⚡ Langfuse configuration"):
        langfuse_host = st.text_input("Host", value=os.getenv('LANGFUSE_INTERNAL_HOST'), disabled=True)
        langfuse_public_key = st.text_input("Public Key", value=os.getenv('LANGFUSE_PUBLIC_KEY'), disabled=True)
        langfuse_private_key = st.text_input("Private Key", value=os.getenv('LANGFUSE_SECRET_KEY'), type="password", disabled=True)

    col1, col2 = st.columns(2)
    start_date = col1.date_input("📆 Start Date", value=datetime(2024, 1, 1), disabled='langfuse_traces' in st.session_state)
    end_date = col2.date_input("📅 End Date", value=datetime(2026, 1, 1), disabled='langfuse_traces' in st.session_state)
    # Convert to datetime objects
    start_date = datetime(start_date.year, start_date.month, start_date.day)
    end_date = datetime(end_date.year, end_date.month, end_date.day)

    if 'langfuse_datasets' not in st.session_state and 'langfuse_traces' not in st.session_state:
        if st.button("⬇️ Load data from Langfuse", type="primary", use_container_width=True):
            with st.spinner("Loading data from Langfuse..."):
                langfuse_datasets = LangfuseUtils.get_datasets_names()
                all_traces_names = LangfuseUtils.get_traces_names(start_date=start_date, end_date=end_date)
                
                st.session_state['langfuse_datasets'] = langfuse_datasets
                st.session_state['langfuse_traces'] = all_traces_names

                st.rerun()
        else:
            st.stop()


    if 'langfuse_data' not in st.session_state:
        selected_traces_names = st.multiselect("Select Langfuse Traces", options=st.session_state['langfuse_traces'], placeholder="Select one or more traces")
        st.session_state['selected_traces_names'] = selected_traces_names

        selected_dataset = st.selectbox("Select Langfuse Dataset (optional)", options=[''] + st.session_state['langfuse_datasets'])
        st.session_state['selected_dataset'] = selected_dataset
        st.write("*A valid Langfuse Dataset is required to compute **Consistency** and **Sensitivity split by Label**.")

        if st.button("📊 Compute Metrics", type="primary", use_container_width=True):
            if len(selected_traces_names) == 0:
                st.error("❌ Please select at least one Trace!")
                st.stop()

            with st.spinner("Computing metrics..."):
                st.session_state['langfuse_data'] = {}
                for traces_group_id in selected_traces_names:
                    # XXX By default, the granularity to aggregate traces and compute metrics is based on the trace name.
                    # We call it `traces_group_id` to make it future-proof for the Advanced Mode where we can aggregate traces
                    # by using a set of custom Lanfguse filters on trace name and/or tags and/or metadata.
                    st.session_state['langfuse_data'][traces_group_id] = {}
                    
                    metrics_calculator = SensitivityConsistencyMetrics(trace_name=traces_group_id)
                    traces = metrics_calculator.fetch_traces(start_date=start_date, end_date=end_date)
                    raw_data_df = SensitivityConsistencyMetrics.traces_to_dataframe(traces)
                    sensitivity_df = metrics_calculator.compute_sensitivity(traces)
                    
                    st.session_state['langfuse_data'][traces_group_id]['raw_data_df'] = raw_data_df
                    st.session_state['langfuse_data'][traces_group_id]['sensitivity_df'] = sensitivity_df.rename(columns={'entropy': 'sensitivity'})

                    if selected_dataset != '':
                        sensitivity_df_with_gt = metrics_calculator.compute_sensitivity(traces, dataset_name=selected_dataset)
                        st.session_state['langfuse_data'][traces_group_id]['sensitivity_df_with_gt'] = sensitivity_df_with_gt.rename(columns={'entropy': 'sensitivity'})

                        # We can compute Consistency only if we have ground truth labels!
                        consistency_dict, consistency_matrix_dict = metrics_calculator.compute_consistency(traces, dataset_name=selected_dataset)
                        st.session_state['langfuse_data'][traces_group_id]['consistency_dict'] = consistency_dict
                        st.session_state['langfuse_data'][traces_group_id]['consistency_matrix_dict'] = consistency_matrix_dict

                st.rerun()
    else:
        st.multiselect("Select Langfuse Traces", options=st.session_state['langfuse_traces'], default=st.session_state['selected_traces_names'], disabled=True)
        st.selectbox("Select Langfuse Dataset (optional)", options=st.session_state['selected_dataset'], disabled=True)
        if st.session_state['selected_dataset'] == '':
            st.warning("⚠️ No dataset selected! **Consistency** and **Sensitivity split by Label** will be skipped.")
        if st.button("♻️ Reset", type="secondary", use_container_width=True):
            del st.session_state['langfuse_data']
            st.rerun()

pg = st.navigation(pages, expanded=True)

pg.run()

# Sensitivity and Consistency of Large Language Models
#
#     File: streamlit_app.py
#
#     Authors:  Federico Errica (federico.errica@neclab.eu)
#               Giuseppe Siracusano (giuseppe.siracusano@neclab.eu)
#               Davide Sanvito (davide.sanvito@neclab.eu)
#               Roberto Bifulco (roberto bifulco@neclab.eu)
#
# NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.
#
# THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
# PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement (Agreement) between your academic institution
# or non-profit organization or self (called Licensee or You in this
# Agreement) and NEC Laboratories Europe GmbH (called Licensor in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term Software means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED AS-IS WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#


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

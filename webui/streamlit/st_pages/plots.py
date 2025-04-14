import numpy as np
import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

@st.dialog(title="🔍 Network Graph", width="large")
def plot_nx_graph(G: nx.Graph, fully_connected_components: list[list[int]]) -> None:    
    # Get a color for each component from a colormap
    n_components = len(fully_connected_components)
    colors = plt.cm.tab10(np.linspace(0, 1, n_components))  # Using Set3 colormap
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, k=0.75)

    # Create a figure
    fig = go.Figure()
    
    # First add all edges in light gray
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace for all edges in light gray
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(200,200,200,0.5)'),  # light gray
        hoverinfo='none',
        mode='lines'
    )
    fig.add_trace(edge_trace)

    # Then add colored edges for each fully connected component
    for i, (component, color) in enumerate(zip(fully_connected_components, colors)):
        edge_x = []
        edge_y = []
        # Get all edges between nodes in this component
        component_edges = [(i, j) for i in component for j in component if i < j and G.has_edge(i, j)]
        for edge in component_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace for this component
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color=f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'),
            hoverinfo='none',
            mode='lines'
        )
        fig.add_trace(edge_trace)

        color_hex = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'
        st.write(f'<span style="color:{color_hex}">FCC #{i+1}: {component}</span>', unsafe_allow_html=True)

    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node}')  # Node label

    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=5,
            color='blue',
            line=dict(width=2, color='black')
        ),
        text=node_text,
        textposition="top center",
        hoverinfo='text'
    )
    fig.add_trace(node_trace)

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    fig.update_layout(width=800)
    st.plotly_chart(fig, use_container_width=True)

    st.write("""<style>
        .stDialog *[role="dialog"] {
            width: 80vw;
            height: 80vh;
        }
        </style>""",
        unsafe_allow_html=True,
        )

def main() -> None:
    if 'langfuse_data' not in st.session_state:
        st.warning("⚠️ No Langfuse data found: please load data first.")
        st.stop()
    
    assert 'selected_dataset' in st.session_state
    
    if 'top_k_samples' not in st.session_state:
        st.session_state['top_k_samples'] = 5
    
    traces_group_ids = st.session_state['langfuse_data'].keys()
    st.markdown(f"#### Selected Traces Groups: `[{', '.join(sorted(traces_group_ids))}]`")
    
    selected_dataset = st.session_state['selected_dataset']
    if selected_dataset != '':
        st.markdown(f"#### Selected Dataset: `{st.session_state['selected_dataset']}`")

    with st.expander("🔍 View Raw Data", expanded=False):
        tabs = st.tabs(list(traces_group_ids))
        for tab, traces_group_id in zip(tabs, traces_group_ids):
            with tab:
                # traces_group_data includes both 'raw_data_df' and 'sensitivity_df', plus other dataframes
                traces_group_data = st.session_state['langfuse_data'][traces_group_id]

                st.write("Raw Traces Group Data")
                st.dataframe(traces_group_data['raw_data_df'])
                if 'sensitivity_df_with_gt' in traces_group_data:
                    st.write("Sensitivity data with Ground Truth labels")
                    st.dataframe(traces_group_data['sensitivity_df_with_gt'])
                else:
                    st.write("Sensitivity data")
                    st.dataframe(traces_group_data['sensitivity_df'])
                
                if 'consistency_dict' in traces_group_data:
                    st.write("Consistency data")
                    st.write(traces_group_data['consistency_dict'])
                if 'consistency_matrix_dict' in traces_group_data:
                    st.write("Consistency matrix data")
                    st.write(traces_group_data['consistency_matrix_dict'])

    st.markdown("# Sensitivity Analysis")

    data_by_traces_group_id = st.session_state['langfuse_data']

    col1, col2 = st.columns(2)

    ##### Sensitivity by Traces Group ##########################################

    with st.spinner("Generating 'Sensitivity by Traces Group' plot..."):
        fig1_plotly = go.Figure()
        for traces_group_id, traces_group_data in data_by_traces_group_id.items():
            # traces_group_data includes both 'raw_data_df' and 'sensitivity_df', plus other dataframes
            sensitivity_df = traces_group_data['sensitivity_df']
            fig1_plotly.add_trace(go.Violin(y=sensitivity_df['sensitivity'], 
                                        name=traces_group_id,
                                        box_visible=True, 
                                        meanline_visible=True))

        fig1_plotly.update_layout(title_text="Sensitivity by Traces Group",
                                width=600, height=400, yaxis=dict(rangemode='nonnegative'),
                                xaxis_title="Traces Group",
                                yaxis_title="Sensitivity")
        
        with col1:
            st.plotly_chart(fig1_plotly)

    ##### Sensitivity by Traces Group and Class ################################

    with st.spinner("Generating 'Sensitivity by Traces Group and Class' plot..."):
        fig2_plotly = go.Figure()
        marker_symbols = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'star'] * 10

        # Make sure that all traces have ground truth labels
        has_gt_labels = True
        for traces_group_id, traces_group_data in data_by_traces_group_id.items():
            # traces_group_data includes both 'raw_data_df' and 'sensitivity_df', plus other dataframes
            if 'sensitivity_df_with_gt' not in traces_group_data:
                has_gt_labels = False
                break

        if has_gt_labels:
            for (traces_group_id, traces_group_data), marker in zip(data_by_traces_group_id.items(), marker_symbols):
                # traces_group_data includes both 'raw_data_df' and 'sensitivity_df', plus other dataframes
                raw_data_df = traces_group_data['raw_data_df']
                assert len(raw_data_df['labels'].value_counts()) == 1
                labels_list = raw_data_df['labels'].iloc[0]

                df = traces_group_data['sensitivity_df_with_gt']
                values = [df[df['expected_output'] == label]['sensitivity'].mean() for label in labels_list]
                fig2_plotly.add_trace(go.Scatter(x=labels_list, 
                                                y=values, 
                                                mode='lines+markers',
                                                name=traces_group_id, 
                                                marker_symbol=marker))
            
            fig2_plotly.update_layout(title_text="Sensitivity by Traces Group and Class",
                                    width=800, height=400,
                                    showlegend=True,
                                    xaxis_title="Class",
                                    yaxis_title="Sensitivity",
                                    yaxis=dict(rangemode='nonnegative'))
            fig2_plotly.update_xaxes(gridcolor='rgba(128, 128, 128, 0.2)', gridwidth=1)
            fig2_plotly.update_yaxes(gridcolor='rgba(128, 128, 128, 0.2)', gridwidth=1)

            with col2:
                st.plotly_chart(fig2_plotly)
        else:
            with col2:
                st.warning("⚠️ No Ground Truth labels available: please add the dataset to Langfuse to plot the **Sensitivity split by Class**.")

    ##### Most sensitive samples ##############################################################

    for traces_group_id in traces_group_ids:
        # traces_group_data includes both 'raw_data_df' and 'sensitivity_df', plus other dataframes
        traces_group_data = st.session_state['langfuse_data'][traces_group_id]

        raw_data_df = traces_group_data['raw_data_df']
        assert len(raw_data_df['labels'].value_counts()) == 1
        labels_list = raw_data_df['labels'].iloc[0]

        st.dataframe(raw_data_df['labels'].head(1).rename("Order of labels in the 'distribution' columns below"),
                     hide_index=True)

        if 'sensitivity_df_with_gt' in traces_group_data:
            has_gt_labels = True
            sensitivity_df = traces_group_data['sensitivity_df_with_gt']
        else:
            has_gt_labels = False
            sensitivity_df = traces_group_data['sensitivity_df']

        st.markdown(f"**Most sensitive samples for '{traces_group_id}' trace**")
        if st.session_state['top_k_samples'] < len(sensitivity_df):
            if st.button("🔄 Load more samples", key=f"load_more_samples_{traces_group_id}"):
                st.session_state['top_k_samples'] *= 2
                st.rerun()

        # Select top-k most sensitive samples
        most_sensitive_samples_df = sensitivity_df.sort_values(by='sensitivity', ascending=False).head(st.session_state['top_k_samples'])

        # Add 'input' column by looking up the input_tag in raw_data_df
        most_sensitive_samples_df['input'] = ''
        for index, row in most_sensitive_samples_df.iterrows():
            input_ = raw_data_df[raw_data_df['input_tag'] == row['input_tag']]['input']
            assert len(set(input_)) == 1, "Multiple inputs found for the same input tag"
            most_sensitive_samples_df.at[index, 'input'] = input_.values[0]
               
        # Add a dummy column to include distribution data both as an array and as an histogram
        most_sensitive_samples_df['distribution_barchart'] = most_sensitive_samples_df['distribution']

        column_config = {
            "distribution": st.column_config.ListColumn(
                label="distribution (raw data)",
                help=f"Distribution of predicted labels based on the following order: \n \n `[{', '.join(labels_list)}]`"
            ),
            "distribution_barchart": st.column_config.BarChartColumn(
                label="distribution (barchart)",
                help=f"Distribution of predicted labels based on the following order: \n \n `[{', '.join(labels_list)}]`",
                y_min=0, y_max=1
            )
        }

        if has_gt_labels:
            # Add 'prob(expected_output)' column by looking at the value of the expected_output in the distribution column
            most_sensitive_samples_df['prob(expected_output)'] = most_sensitive_samples_df.apply(
                lambda row: row['distribution'][labels_list.index(row['expected_output'])] if row['expected_output'] in labels_list else None, 
                axis=1
            )

            st.dataframe(most_sensitive_samples_df[['input', 'sensitivity', 'distribution', 'distribution_barchart', 'expected_output', 'prob(expected_output)']],
                         column_config=column_config)
        else:
            st.dataframe(most_sensitive_samples_df[['input', 'sensitivity', 'distribution', 'distribution_barchart']],
                         column_config=column_config)
        
        with st.expander("🔍 Individual Samples Analysis", expanded=True):
            selected_sample_id = st.selectbox("Select a sample ID", [''] + list(most_sensitive_samples_df.index), key=f"select_sample_id_{traces_group_id}")
            if selected_sample_id != '':
                selected_sample_df = most_sensitive_samples_df.loc[selected_sample_id]

                st.write(f"**Input:** {selected_sample_df['input']}")
                st.write(f"**Sensitivity:** {selected_sample_df['sensitivity']:.4f}")
                if has_gt_labels:
                    st.write(f"**Expected Output:** {selected_sample_df['expected_output']}")
                
                distribution_df = pd.DataFrame({
                    'Class': labels_list,
                    'Probability': selected_sample_df['distribution']
                })
                colors = ['red'] * len(labels_list)
                if has_gt_labels and selected_sample_df['expected_output'] in labels_list:
                    # Set green color for the bar matching expected output
                    expected_label_index = labels_list.index(selected_sample_df['expected_output'])
                    colors[expected_label_index] = 'green'
                
                fig = px.bar(distribution_df, x='Class', y='Probability', 
                           title='Distribution of Predicted Classes')
                # Update bars colors
                fig.update_traces(marker_color=colors)
                fig.update_layout(width=400, height=300)
                st.plotly_chart(fig, use_container_width=False)

                ########################################

                st.write(f"##### **Prompts sorted by frequency of predicted label**")

                # Select the row in raw_data_df that has the same input_tag as the selected sample
                prompts_df = raw_data_df[raw_data_df['input_tag'] == selected_sample_df['input_tag']][['prompt', 'predicted_label']]

                # Sort DataFrame based on frequency of predicted_label
                label_counts = prompts_df['predicted_label'].value_counts()
                prompts_df['label_frequency'] = prompts_df['predicted_label'].map(label_counts)
                prompts_df = prompts_df.sort_values(by='label_frequency', ascending=False).drop(columns=['label_frequency'])
                if has_gt_labels:
                    prompts_df = prompts_df.style.applymap(lambda x: 'background-color: green; color: white' if x == selected_sample_df['expected_output'] else '', subset=['predicted_label'])
                st.dataframe(prompts_df, hide_index=True)

    ###########################################################################

    st.divider()
    st.markdown("# Consistency Analysis")

    if 'consistency_dict' not in st.session_state['langfuse_data'][traces_group_id] or len(st.session_state['langfuse_data'][traces_group_id]['consistency_dict']) == 0:
        st.warning("⚠️ No Ground Truth labels available: **Consistency** metrics cannot be computed.")
        st.stop()

    consistency_tabs = st.tabs(list(traces_group_ids))
    for tab, traces_group_id in zip(consistency_tabs, traces_group_ids):
        with tab:
            traces_group_data = st.session_state['langfuse_data'][traces_group_id]
            consistency_dict = traces_group_data['consistency_dict']
            consistency_matrix_dict = traces_group_data['consistency_matrix_dict']
            raw_data_df = traces_group_data['raw_data_df']
            sensitivity_df = traces_group_data['sensitivity_df_with_gt']

            st.write("##### Pairwise-Consistency Matrix by Class")

            consistency_matrix_cols = st.columns(len(consistency_matrix_dict))
            for col_ix, (label, consistency_matrix) in enumerate(consistency_matrix_dict.items()):
                with consistency_matrix_cols[col_ix]:
                    fig = px.imshow(consistency_matrix,
                                    title=label,
                                    labels=dict(x="Sample ID", y="Sample ID", color="C<sub>y</sub>(x,x')"),
                                    zmin=0, zmax=1,
                                    # color_continuous_scale='Blues_r'
                                    )
                    st.plotly_chart(fig, key=f"consistency_matrix_{label}_{traces_group_id}")
            
            ########################################

            st.write("##### Distribution of consistency across samples of a given class")
            
            consistency_histogram_cols = st.columns(len(consistency_matrix_dict))
            for col_ix, (label, consistency_matrix) in enumerate(consistency_matrix_dict.items()):
                with consistency_histogram_cols[col_ix]:
                    upper_triangle = consistency_matrix[np.triu_indices_from(consistency_matrix)]
                    fig = px.histogram(upper_triangle,
                                    title=label,
                                    x=upper_triangle,
                                    nbins=30,
                                    histnorm='probability')
                    fig.update_layout(height=300, xaxis_title="C<sub>y</sub>(x,x')", yaxis_title="Probability")
                    st.plotly_chart(fig, key=f"consistency_histogram_{label}_{traces_group_id}")

                    st.write(f"**Average Consistency : {consistency_dict[label]:.2f}**")
            
            ########################################

            with st.expander("🔍 Samples Groups Analysis", expanded=True):
                selected_label = st.selectbox("Select a class", [''] + list(consistency_matrix_dict.keys()), key=f"select_label_{traces_group_id}_deep_analysis")
                if selected_label != '':
                    consistency_matrix = consistency_matrix_dict[selected_label]

                    fig = px.imshow(consistency_matrix,
                                    title=selected_label,
                                    labels=dict(x="Sample ID", y="Sample ID", color="C<sub>y</sub>(x,x')"),
                                    zmin=0, zmax=1,
                                    # color_continuous_scale='Blues_r'
                                    )
                    fig.update_layout(height=500, width=500)
                    st.plotly_chart(fig, key=f"consistency_matrix_{selected_label}_deep_analysis_{traces_group_id}")

                    if consistency_matrix.shape[0] == 1:
                        st.warning("⚠️ Only one sample found for the selected class.")
                        st.stop()

                    samples_selected_class = sensitivity_df[sensitivity_df['expected_output'] == selected_label].reset_index(drop=True).copy()

                    # Add a dummy column to include distribution data both as an array and as an histogram
                    samples_selected_class['distribution_barchart'] = samples_selected_class['distribution']

                    column_config = {
                        "distribution": st.column_config.ListColumn(
                            label="distribution (raw data)",
                            help=f"Distribution of predicted labels based on the following order: \n \n `[{', '.join(labels_list)}]`"
                        ),
                        "distribution_barchart": st.column_config.BarChartColumn(
                            label="distribution (barchart)",
                            help=f"Distribution of predicted labels based on the following order: \n \n `[{', '.join(labels_list)}]`",
                            y_min=0, y_max=1
                        )
                    }

                    # Add 'input' column by looking up the input_tag in raw_data_df
                    samples_selected_class['input'] = ''
                    for index, row in samples_selected_class.iterrows():
                        input_ = raw_data_df[raw_data_df['input_tag'] == row['input_tag']]['input']
                        assert len(set(input_)) == 1, "Multiple inputs found for the same input tag"
                        samples_selected_class.at[index, 'input'] = input_.values[0]
                    
                    # Add 'prob(expected_output)' column by looking at the value of the expected_output in the distribution column
                    samples_selected_class['prob(expected_output)'] = samples_selected_class['distribution'].apply(lambda x: x[labels_list.index(selected_label)])

                    # Replace 'input_tag' column with 'input', i.e. move input as first oclumn in place of input_tag
                    samples_selected_class = samples_selected_class[['input', 'sensitivity', 'distribution', 'distribution_barchart', 'expected_output', 'prob(expected_output)']]

                    # st.write(samples_selected_class)

                    ########################################

                    thr = st.slider("Select minimum Consistency threshold to identify **Fully-Connected Components (FCC)** or **Cliques**, i.e. samples which are **all** consistent enough with each other", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
                    # Check values above threshold excluding the diagonal and the lower triangle submatrix
                    high_consistency_indexes = np.argwhere((consistency_matrix >= thr) & (np.triu(np.ones(consistency_matrix.shape), k=1) == 1))
                    st.write(f"Sample pairs with consistency values above {thr}:\n`{high_consistency_indexes}`")

                    G = nx.Graph()
                    G.add_edges_from(high_consistency_indexes)

                    with st.spinner("Identifying fully-connected components..."):
                        fully_connected_components = list(nx.find_cliques(G))
                        # st.write(fully_connected_components)

                    fully_connected_components = [[int(x) for x in sorted(component)] for component in fully_connected_components]
                    # Sort components by size
                    fully_connected_components = sorted(fully_connected_components, key=len, reverse=True)

                    if st.button("🔍 Network Graph"):
                        plot_nx_graph(G, fully_connected_components)

                    # Step 3: Print results
                    for i, component in enumerate(fully_connected_components):
                        st.write(f"FCC #{i+1}: {component}")
                        st.dataframe(samples_selected_class.iloc[list(component)], column_config=column_config)

main()

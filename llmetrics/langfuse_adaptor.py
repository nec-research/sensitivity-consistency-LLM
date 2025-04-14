from datetime import datetime, timedelta
import hashlib
from typing import List, Sequence
from langfuse import Langfuse
from langfuse.decorators import LangfuseDecorator
from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
from langfuse.api.resources.commons.types.dataset_item import DatasetItem
import logging
import os
import numpy as np
from openai import BaseModel
import pandas as pd

from llmetrics.metadata import Metadata
from llmetrics.metrics import compute_consistency, compute_entropy, consistency_matrix

log = logging.getLogger(__name__)

BATCH_SIZE = 50
MAX_TRACES = 1000 # hard limit for now
 

class SensitivityConsistencyMetrics:
    """
    Class to compute sensitivity and consistency metrics from Langfuse traces
    """
    def __init__(self, trace_name: str):
        self.trace_name = trace_name
        self.langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_INTERNAL_HOST'),
        )

    @classmethod
    def append_sensitivity_metadata_to_trace(cls, prompt: str, 
                                             response_model: BaseModel, 
                                             user_input: str,  
                                             predicted_label: str,
                                             labels_list: List[str], 
                                             model: str, 
                                             langfuse_context: LangfuseDecorator
                                             ):

        _prompt = prompt + f"\n {response_model.model_json_schema()}"
        
        prompt_tag = hashlib.sha256(_prompt.encode()).hexdigest()
        input_tag = hashlib.sha256(user_input.encode()).hexdigest()
        
        meta = Metadata(
            model=model,
            prompt=_prompt,
            prompt_tag=f"prompt-{prompt_tag}",
            input=user_input,
            input_tag=f"input-{input_tag}",
            predicted_label=predicted_label,
            labels=sorted(labels_list)
        )
        
        langfuse_context.update_current_observation(
            metadata=meta
        )

        langfuse_context.update_current_trace(
            tags=[meta.prompt_tag, meta.input_tag, meta.predicted_label]
        )

    @classmethod
    def traces_to_dataframe(cls, traces: List[TraceWithDetails]) -> pd.DataFrame:
        data = [(t.metadata['input'], 
                t.metadata['labels'], 
                t.metadata['prompt'], 
                t.metadata['input_tag'], 
                t.metadata['prompt_tag'], 
                t.metadata['predicted_label'],
                t.metadata['model']) for t in traces if t.metadata is not None]
        return pd.DataFrame(data, columns=['input', 'labels', 'prompt', 'input_tag', 'prompt_tag', 'predicted_label', 'model'])


    def fetch_traces(self, start_date: datetime | None = None, end_date: datetime | None = None) -> Sequence[TraceWithDetails]:
        now = datetime.now()
        if end_date is None:
            end_date = datetime(now.year, now.month, now.day, 23, 59)
        else:
            end_date = end_date

        if start_date is None:
            start_date = end_date - timedelta(days=1)
        else:
            start_date = start_date
        
        all_traces = []
        limit = BATCH_SIZE  # Adjust as needed to balance performance and data retrieval.
        page = 1
        while True:
            traces = self.langfuse.fetch_traces(limit=limit, 
                                                name=self.trace_name, 
                                                from_timestamp=start_date,
                                                to_timestamp=end_date,
                                                page=page)
            all_traces.extend(traces.data)
            if len(traces.data) < limit or len(all_traces) >= MAX_TRACES:
                break
            page += 1
        
        log.info(f"Retrieved {len(all_traces)} traces for {self.trace_name} from {start_date} to {end_date}.")
        print(f"Retrieved {len(all_traces)} traces for {self.trace_name} from {start_date} to {end_date}.")
        
        return all_traces
    
    # TODO: medatata is not typed
    # TODO: balance dataset?
    # TODO: external visualization for proper graphs?
    # def compute_metric(self):
    #     traces = self.fetch_traces()
    #     input_aggregated_traces: dict[str, list[TraceWithDetails]] = {}
    #     for t in traces:
    #         if t.metadata is None:
    #             continue
    #         if t.metadata['input_tag'] not in input_aggregated_traces.keys():
    #             input_aggregated_traces[t.metadata['input_tag']] = []
    #         input_aggregated_traces[t.metadata['input_tag']].append(t)

       
    #     for k, agg_traces in input_aggregated_traces.items():

    #         outputs = [t.metadata['predicted_label'] for t in agg_traces if 'predicted_label' in t.metadata.keys()]
    #         prompts = [t.metadata['prompt_tag'] for t in agg_traces if 'prompt_tag' in t.metadata.keys()]
    #         label_counts = np.array(list(dict(zip(self.labels, [np.sum(np.array(outputs) == label) for label in self.labels])).values()))
    #         label_entropy, label_distribution = compute_entropy(len(prompts), label_counts)

    #         for t in agg_traces:
    #             self.langfuse.score(
    #                 trace_id=t.id,
    #                 name="sensitivity",
    #                 value=label_entropy
    #             )
    #         log.info(f"Processed {k} with entropy {label_entropy}, distribution {label_distribution}")
    #         print(f"Processed {k} with entropy {label_entropy}, distribution {label_distribution}")


    def _get_datset(self, dataset_name: str) -> dict[str, DatasetItem]:
        try:    
            dataset = self.langfuse.get_dataset(dataset_name)
            data: dict[str, DatasetItem] = {}
            data = {'input-'+hashlib.sha256(item.input.encode()).hexdigest(): item for item in dataset.items}
            return data
        except Exception as e: # TODO: handle this better 
            log.error(f"Error getting dataset {dataset_name}: {e}")
            return {}

    # TODO: medatata is not typed
    # TODO: balance dataset?
    def compute_sensitivity(self, traces: List[TraceWithDetails], dataset_name: str | None = None) -> pd.DataFrame | None:
        
        df = SensitivityConsistencyMetrics.traces_to_dataframe(traces)
        grouped_traces = df.groupby('input_tag')

        data = []
        if dataset_name is not None:
            labeled_data = self._get_datset(dataset_name)
        else:
            labeled_data = {}
            
        for input_tag, group in grouped_traces:
            # prompts in the group
            prompts = group['prompt_tag']           
            # Get the value counts of 'predicted_label' in the group
            predicted_label_counts = group['predicted_label'].value_counts()
            predicted_label_counts_dict = predicted_label_counts.to_dict()

            assert len(group['labels'].value_counts()) == 1
            labels_list = group['labels'].iloc[0]
            label_counts = [predicted_label_counts_dict.get(l, 0) for l in sorted(labels_list)]
            # Compute entropy
            label_entropy, label_distribution = compute_entropy(len(prompts), np.array(label_counts))
            
            # get the expected output if in dataset
            expected_output = labeled_data.get(input_tag, None)
            if expected_output is not None:
                expected_output = expected_output.expected_output
            
            data.append({
                'input_tag': input_tag,
                # 'predicted_label':group['predicted_label'].to_list(),
                # 'input': group['input'].iloc[0],
                # 'prompt_tag': group['prompt_tag'].to_list(),
                # 'prompts': prompts,
                'entropy': label_entropy,
                'distribution': label_distribution,
                'expected_output': expected_output
            })
        return pd.DataFrame(data)
    
    def compute_consistency(self, traces: List[TraceWithDetails], dataset_name: str ) -> tuple[dict[str, float], dict[str, np.array]]:
    
        df = SensitivityConsistencyMetrics.traces_to_dataframe(traces)
        dataset = self._get_datset(dataset_name)        
        records = df.to_dict('records')

        for record in records:
            if record['input_tag'] in dataset:
                record['expected_output'] = dataset[record['input_tag']].expected_output
            else:
                record['expected_output'] = None

        labeled_traces = pd.DataFrame(records).fillna(np.nan).dropna() # TODO: handle duplicates
        
        
        consistency_dict: dict[str, float] = {}
        consistency_matrix_dict: dict[str, np.array] = {}

        assert len(df['labels'].value_counts()) == 1
        labels_list = df['labels'].iloc[0]
        
        # TODO: handle N/A
        for label in labels_list:
            
            per_class_traces = labeled_traces[labeled_traces['expected_output'] == label]
            
            if len(per_class_traces) == 0:
                continue
            

            samples_distributions = []

            for input_tag, group in per_class_traces.groupby('input_tag'):
                prompts = group['prompt_tag']           
                # Get the value counts of 'predicted_label' in the group
                predicted_label_counts = group['predicted_label'].value_counts()
                predicted_label_counts_dict = predicted_label_counts.to_dict()
                label_counts = [predicted_label_counts_dict.get(l, 0) for l in sorted(labels_list)]
                label_distribution = np.array(label_counts) / len(prompts)
                samples_distributions.append(label_distribution)
                
                #TODO if n_samples_c > 0:
                
            samples_distributions = np.array(samples_distributions)
            

            
            _consistency_matrix = consistency_matrix(samples_distributions)
            consistency = compute_consistency(_consistency_matrix)
            consistency_dict[label] = consistency
            consistency_matrix_dict[label] = _consistency_matrix

        return consistency_dict, consistency_matrix_dict




class LangfuseUtils:
    """
    Class to extract datasets and traces from Langfuse
    """

    @classmethod
    def get_datasets_names(cls):
        langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_INTERNAL_HOST'),
        )
        datasets = langfuse.client.datasets.list()
        return [dataset.name for dataset in datasets.data]
    
    @classmethod
    def get_traces_names(cls, start_date: datetime | None = None, end_date: datetime | None = None):
        langfuse = Langfuse(
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            host=os.getenv('LANGFUSE_INTERNAL_HOST'),
        )
        now = datetime.now()
        if end_date is None:
            end_date = datetime(now.year, now.month, now.day, 23, 59)

        if start_date is None:
            start_date = end_date - timedelta(days=1)

        all_traces_names = set()
        limit = BATCH_SIZE  # Adjust as needed to balance performance and data retrieval.

        page = 1
        while True:
            traces = langfuse.fetch_traces(limit=limit, 
                                            from_timestamp=start_date,
                                            to_timestamp=end_date,
                                            page=page)
            for trace in traces.data:
                all_traces_names.add(trace.name)
            if len(traces.data) < limit:
                break
            page += 1

        return list(all_traces_names)

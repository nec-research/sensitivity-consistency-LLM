"""
       Sensitivity and Consistency of Large Language Models

  File:     util.py
  Authors:  Federico Errica (federico.errica@neclab.eu)
            Giuseppe Siracusano (giuseppe.siracusano@neclab.eu)
	    Davide Sanvito (davide.sanvito@neclab.eu)
	    Roberto Bifulco (roberto bifulco@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2025-, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import json
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, f1_score

from constants import LLAMA3_openai_api_base, MIXTRAL_openai_api_base


class Table:
    def __init__(self, row_headers, col_headers, data):
        """
        Initialize the Table object.

        Args:
        - row_headers (list): List of strings representing row headers.
        - col_headers (list): List of strings representing column headers.
        - data (list of lists): 2D list containing the data for the table.
          Each inner list represents a row of data corresponding to the column headers.
        """
        self.row_headers = row_headers
        self.col_headers = col_headers
        self.data = data

        # Validate data dimensions
        if len(row_headers) != len(data):
            raise ValueError("Number of row headers must match the number of rows in data.")
        for row in data:
            if len(col_headers) != len(row):
                raise ValueError("Number of column headers must match the number of columns in data.")

    def __str__(self):
        """
        Return a formatted string representation of the table.
        """
        # Determine maximum width needed for each column
        col_widths = [len(header) for header in self.col_headers]
        for row in self.data:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Create the formatted table
        formatted_table = []

        # Header row
        header_row = ' | '.join(f"{self.col_headers[i]:{col_widths[i]}}" for i in range(len(self.col_headers)))
        formatted_table.append(header_row)
        formatted_table.append('-' * len(header_row))

        # Data rows
        for j, row in enumerate(self.data):
            row_str = ' | '.join(f"{str(cell):{col_widths[i]}}" for i, cell in enumerate(row))
            formatted_table.append(f"{self.row_headers[j]:{col_widths[0]}} | {row_str}")

        return '\n'.join(formatted_table)


def DefaultNLELlama3_70bChatOpenAI(temperature: float):
    if temperature > 0.:
        return ChatOpenAI(
            model="llama3instruct",
            temperature=temperature,
            openai_api_base=LLAMA3_openai_api_base,
            openai_api_key='EMPTY',
            timeout=36000000,  # 3600 seconds
            model_kwargs={"extra_body": {"stop_token_ids": [128009]}}
        )
    else:
        return ChatOpenAI(
            model="llama3instruct",
            temperature=temperature,
            openai_api_base=LLAMA3_openai_api_base,
            openai_api_key='EMPTY',
            timeout=36000000,  # 3600 seconds
            model_kwargs={"extra_body": {"stop_token_ids": [128009]},
                          "seed": 42}
        )

def DefaultNLEMixtral_8x7bChatOpenAI(temperature: float):
    if temperature > 0.:
        return ChatOpenAI(
            model="mixtral", # Mixtral-8x7B-Instruct-v0.1
            temperature=temperature,
            openai_api_base=MIXTRAL_openai_api_base,
            openai_api_key='EMPTY',
            timeout=36000000,  # 3600 seconds
        )
    else:
        return ChatOpenAI(
            model="mixtral",  # Mixtral-8x7B-Instruct-v0.1
            temperature=temperature,
            openai_api_base=MIXTRAL_openai_api_base,
            openai_api_key='EMPTY',
            timeout=36000000,  # 3600 seconds
            model_kwargs={"seed": 42}
        )


llm_map = {
    'llama3': DefaultNLELlama3_70bChatOpenAI,
    'mixtral': DefaultNLEMixtral_8x7bChatOpenAI
}


def generate_questions(llm_name: str,
                       question: str,
                       temperature: float,
                       no_questions: int = 10) -> List[str]:
    """
    Generates a number of semantically equivalent questions to the input
    questions using an LLM (optionally fine-tuned for rephrasing).
    A deterministic output will be produced.

    :param llm_name: the llm name to call for rephrasing
    :param question: the question to be rephrased
    :param temperature: the temperature of the LLM
    :param no_questions: the number of questions to create, including the original one
    :return: a list of questions
    """
    # Instantiate LLM
    llm = llm_map[llm_name](temperature=temperature)

    questions = [question]

    if 'mixtral' not in llm_name:
        prompt = [('system', 'You are asked to rephrase a question in a semantically equivalent but syntactically different way.'
                             ' Vary the length of the question as long as you do not alter the meaning of the question.'
                             ' Provide only the rephrased sentence.'
                             ' The original question is the following: {question}.'
                             ' Also, the following list contains some questions that you already generated, do not repeat yourself:\n {alternative_questions}'
                   ),
            ('human', f'Rephrase the original question.')]
    else:
        prompt = [('user',
                   'You are asked to rephrase a question in a semantically equivalent but syntactically different way.'
                   ' Vary the length of the question as long as you do not alter the meaning of the question.'
                   ' Provide only the rephrased sentence with no additional notes.'
                   ' The original question is the following: {question}.'
                   ' Also, the following list contains some questions that you already generated, do not repeat yourself:\n {alternative_questions}.'
                   )]

    i = 0
    while i < no_questions-1:

        chat_prompt = ChatPromptTemplate.from_messages(prompt)

        alternative_questions_str = ''
        for q in questions[-10:]:
            alternative_questions_str += f'- {q}\n'

        alternative_questions_str = re.escape(alternative_questions_str)

        messages = chat_prompt.format_messages(question=question,
                                               alternative_questions=alternative_questions_str)

        new_question = llm.invoke(messages).content

        # print(f'Message {i}: {messages}')
        # print(new_question)

        questions.append(new_question)

        i += 1

    return questions


def classify_llm(llm_name:str,
                          prompts: List,
                          summary: str,
                          temperature: float,
                          no_parallel_calls: int = 1) -> List[str]:
    """
    Calls an LLM to perform a classification with a given prompt
    :param llm_name: the llm to call (depends on internal/external models)
    :param prompts: the list of prompts (one for each alternative rephrasing) of the model in the right format
    :param summary: the summary to be replaced in the prompt
    :param temperature: the temperature of the LLM
    :param no_parallel_calls: the numer of identical calls to the LLM
    :return: a dictionary of answers of the LLM, where keys are of the form "questionID_answerID"
    """
    llm = llm_map[llm_name](temperature=temperature)

    chains = {f"{str(q)}_{str(k)}": (ChatPromptTemplate.from_messages(map(lambda x: tuple(x), prompt)) | llm) for k in range(no_parallel_calls) for q, prompt in enumerate(prompts)}

    map_chain = RunnableParallel(chains)

    return map_chain.invoke({"summary": summary})


def parse_json_to_dataframe(input_file: Path) -> pd.DataFrame:
    """
    Parse a JSON file containing a list of dictionaries and convert it into a
     Pandas DataFrame. It assumes a multilabel classification task.

    Each dictionary in the input JSON file should have the following structure:
    {
        "cat_name": str,
        "possible_tags": List[str],
        "prompt": str,
        "question": str,
        "entries": [
            {
                "id": str/int,
                "summary": str,
                "tags": List[str]  # ground truth (multilabel classification)
            }
        ]
    }

    For each entry in the "entries" list of each dictionary, a row will be added to the CSV file.
    The CSV file will contain the following columns:
    - cat_name
    - possible_tags
    - prompt
    - question
    - id
    - summary
    - tags (comma-separated)

    :param input_file: Path to the input JSON file.
    :param output_file: Path to the output CSV file.
    :return: pandas DataFrame containing the parsed data.
    """

    # Open the JSON file
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    rows = []
    # Process each dictionary in the JSON data
    for item in data:
        cat_name = item["cat_name"]
        possible_tags = item["possible_tags"]
        prompt = item["prompt"]
        question = item["question"]

        entries = item["entries"]
        for entry in entries:
            entry_id = entry["id"]
            if entry_id == '':
                continue
            summary = entry["summary"]

            difficulty = entry["difficulty"]

            assert len(entry["tags"]) > 0
            tags = ", ".join(entry["tags"])

            # Append row to the list
            rows.append([cat_name, possible_tags, prompt, question,
                         entry_id, summary, tags, difficulty])

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows, columns=["cat_name", "classes", "prompt",
                                     "question", "id", "summary", "ground_truth", "difficulty"])
    return df


def TVD(distribution1, distributions):
    return 0.5*np.abs(np.expand_dims(distribution1, 0) - distributions).sum(1)


def plot_TVD_info(sample_ids: List[int],
                  prompt_types: dict,
                  llms: List[str],
                  Qs: List[int],
                  temp_questions: List[float],
                  As: List[int],
                  temp_answers: List[float],
                  class_labels: List[str],
                  results_folder: Path):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            # compute distribution over class labels predicted over the Q questions
                            samples_distributions = np.zeros((n_samples, n_classes - 1))

                            # compute boolean class assignment matrix
                            boolean_class_matrix = np.zeros((n_samples, n_classes - 1),
                                                            dtype='bool')

                            for idx, s_id in enumerate(sample_ids):
                                key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                experiment = data_dict[key]

                                target = experiment['target']
                                if target == 'not_entailment':
                                    target = 'contradiction/neutral'

                                target_id = class_labels_to_id[target]

                                boolean_class_matrix[
                                    idx, class_labels_to_id[target]] = True

                                samples_distributions[idx] = np.array(experiment["distribution"])[:-1]

                            TVD_matrix = np.zeros((n_samples, n_samples))

                            for idx, s_id in enumerate(sample_ids):
                                TVD_matrix[idx, :] = 1. - TVD(samples_distributions[idx, :-1],
                                                          samples_distributions[:, :-1])

                            fig, axes = plt.subplots(1, n_classes - 1, figsize=(n_classes * 5, 4))  # Create a grid of subplots
                            for c in range(n_classes - 1):
                                class_filter = boolean_class_matrix[:, c]
                                if class_filter.sum() == 0:
                                    continue

                                sns.heatmap(TVD_matrix[class_filter][:, class_filter],
                                            ax=axes[c])
                                axes[c].set_title(f'Pairwise TVD - Class {class_labels[c]}')

                            # plt.tight_layout()
                            plt.savefig(Path(results_folder, f'TVD_matrix_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()

                            fig, axes = plt.subplots(1, n_classes - 1, figsize=(n_classes * 5, 4))  # Create a grid of subplots
                            for c in range(n_classes-1):
                                class_filter = boolean_class_matrix[:, c]
                                if class_filter.sum() == 0:
                                    continue

                                sns.histplot(
                                    np.reshape(TVD_matrix[class_filter][:,
                                               class_filter], -1),
                                    bins=20, stat='probability', kde=False,
                                    ax=axes[c])
                                axes[c].set_xlabel(f'TVD Value')
                                axes[c].set_ylabel(f'Frequency')

                            # plt.tight_layout()
                            plt.savefig(Path(results_folder, f'TVD_histogram_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))



def print_consistency(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.,
                       filter_zero_sensitivity=False):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"
    print(f'Prompt type: {prompt_type}')

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    # compute distribution over class labels predicted over the Q questions
    samples_distributions = np.zeros(
        (n_samples, n_classes))

    # compute boolean class assignment matrix
    boolean_class_matrix = np.zeros(
        (n_samples, n_classes - 1),
        dtype='bool')

    for idx, s_id in enumerate(sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        target = experiment['target']
        if target == 'not_entailment':
            target = 'contradiction/neutral'

        target_id = class_labels_to_id[target]

        boolean_class_matrix[
            idx, class_labels_to_id[
                target]] = True

        samples_distributions[idx] = np.array(
            experiment["distribution"])[:]

        # Generate a random float between 0 and 1
        random_float = random.random()

        # If the generated float is less than or equal to p, pick a random value from the list
        if random_float < noise_amount:
            sd = np.zeros(len(class_labels))

            for _ in range(Q):
                pred_id = random.choice([i for i in range(len(class_labels))])
                sd[pred_id] += 1

            sd = sd / Q
            print(entropy(sd) / np.log(n_classes))

            samples_distributions[idx] = sd

    consistency = np.zeros(n_classes-1)
    consistency_not_averaged = []
    TVD_matrix_per_class = []
    for c in range(n_classes-1):  # avoid NA

        samples_distributions_c = samples_distributions[boolean_class_matrix[:, c], :]

        if filter_zero_sensitivity: # filter out elements with zero sensitivity, where the prompt rephrasing has no effect
            zero_sensitivity_mask = entropy(samples_distributions_c, axis=1) / np.log(n_classes) == 0.
            samples_distributions_c = samples_distributions_c[~zero_sensitivity_mask]

        n_samples_c = samples_distributions_c.shape[0]

        if n_samples_c > 0:
            TVD_matrix_c = np.zeros((n_samples_c, n_samples_c))

            for idx in range(n_samples_c):
                TVD_matrix_c[idx, :] = 1. - TVD(
                    samples_distributions_c[idx, :],
                    samples_distributions_c[:, :])

        else:
            TVD_matrix_c = np.zeros((1, 1))

        consistency[c] = TVD_matrix_c.mean()
        consistency_not_averaged.append(TVD_matrix_c.reshape(-1))
        TVD_matrix_per_class.append(TVD_matrix_c)

    consistency_not_averaged = np.concatenate(consistency_not_averaged)
    print(f"Avg consistency: {consistency_not_averaged.mean()},"
          f"Std consistency: {consistency_not_averaged.std()}")

    return TVD_matrix_per_class

def print_consistency_over_samples(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.,
                       filter_zero_sensitivity=False):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"
    print(f'Prompt type: {prompt_type}')

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    # compute distribution over class labels predicted over the Q questions
    samples_distributions = np.zeros(
        (n_samples, n_classes))

    # compute boolean class assignment matrix
    boolean_class_matrix = np.zeros(
        (n_samples, n_classes - 1),
        dtype='bool')

    for idx, s_id in enumerate(sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        boolean_class_matrix[
            idx, class_labels_to_id[
                experiment['target']]] = True

        samples_distributions[idx] = np.array(
            experiment["distribution"])[:]

        # Generate a random float between 0 and 1
        random_float = random.random()

        # If the generated float is less than or equal to p, pick a random value from the list
        if random_float < noise_amount:
            sd = np.zeros(len(class_labels))

            for _ in range(Q):
                pred_id = random.choice([i for i in range(len(class_labels))])
                sd[pred_id] += 1

            sd = sd / Q
            print(entropy(sd) / np.log(n_classes))

            samples_distributions[idx] = sd

    consistency = np.zeros(n_classes-1)
    consistency_not_averaged = []
    TVD_matrix_per_class = []
    for c in range(n_classes-1):  # avoid NA

        samples_distributions_c = samples_distributions[boolean_class_matrix[:, c], :]

        if filter_zero_sensitivity: # filter out elements with zero sensitivity, where the prompt rephrasing has no effect
            zero_sensitivity_mask = entropy(samples_distributions_c, axis=1) / np.log(n_classes) == 0.
            samples_distributions_c = samples_distributions_c[~zero_sensitivity_mask]

        n_samples_c = samples_distributions_c.shape[0]

        if n_samples_c > 0:
            TVD_matrix_c = np.zeros((n_samples_c, n_samples_c))

            for idx in range(n_samples_c):
                TVD_matrix_c[idx, :] = 1. - TVD(
                    samples_distributions_c[idx, :],
                    samples_distributions_c[:, :])

        else:
            TVD_matrix_c = np.zeros((1, 1))

        consistency[c] = TVD_matrix_c.mean()
        consistency_not_averaged.append(TVD_matrix_c.reshape(-1))
        TVD_matrix_per_class.append(TVD_matrix_c)


    consistency_not_averaged = np.concatenate(consistency_not_averaged)
    print(f"Avg consistency over classes: {consistency_not_averaged.mean()},"
          f"Std consistency over classes: {consistency_not_averaged.std()}")

    return TVD_matrix_per_class


def plot_questions_vs_predicted_distribution(sample_ids: List[int],
                                             prompt_types: dict,
                                             llms: List[str],
                                             Qs: List[int],
                                             temp_questions: List[float],
                                             As: List[int],
                                             temp_answers: List[float],
                                             class_labels: List[str],
                                             results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            fig, axes = plt.subplots(Q, n_classes - 1, figsize=(
                            n_classes * 5, Q * 4))  # Create a grid of subplots

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]

                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    total = q_c_distribution.sum()
                                    if total == 0:
                                        total = 1.

                                    axes[q, c].bar(np.arange(n_classes),
                                                   q_c_distribution / total)
                                    axes[q, c].set_ylim([0., 1.])
                                    axes[q, c].set_title(f"Class {class_labels[c]}")
                                    axes[q, c].set_xlabel(f"Predicted Class")
                                    axes[q, c].set_xticks(np.arange(n_classes), class_labels,
                                                          rotation='vertical')
                                    axes[q, c].set_ylabel(f"Question ID {q}")

                            plt.tight_layout()
                            plt.savefig(Path(results_folder,
                                             f'question_vs_trueclass_prediction_distributions_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()


def plot_questions_vs_class_sensitivity(sample_ids: List[int],
                                             prompt_types: dict,
                                             llms: List[str],
                                             Qs: List[int],
                                             temp_questions: List[float],
                                             As: List[int],
                                             temp_answers: List[float],
                                             class_labels: List[str],
                                             results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for prompt_type, prompt in prompt_types.items():
        filename = f"results_{prompt_type}.json"
        print(f'Prompt type: {prompt_type}')

        # Open the JSON file and load its contents into a Python dictionary
        with open(Path(results_folder, filename), 'r') as f:
            data_dict = json.load(f)

        for llm in llms:
            for Q in Qs:
                for A in As:
                    for temp_question in temp_questions:
                        for temp_answer in temp_answers:

                            entropy_matrix = np.zeros((Q, n_classes))

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]
                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    entropy_matrix[q, c] = entropy(q_c_distribution)/np.log(n_classes)

                            ax = sns.heatmap(entropy_matrix, vmax=1.)
                            plt.xlabel('Class ID')
                            plt.ylabel('Question ID')
                            ax.set_xticks(
                                np.arange(len(class_labels)) + 0.5,
                                class_labels, rotation='vertical')

                            plt.tight_layout()
                            plt.savefig(Path(results_folder,
                                             f'question_vs_class_entropy_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                            plt.show()


def print_classification_scores(sample_ids: List[int],
                                prompt_type: str,
                                llm: str,
                                Q: int,
                                temp_question: float,
                                A: int,
                                temp_answer: float,
                                class_labels: List[str],
                                results_folder: Path):
    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    row_headers = [f"Question {i + 1}" for i in range(Q)]
    col_headers = ["Acc", "Micro F1", "Macro F1"]

    table = [None for _ in range(Q)]

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename), 'r') as f:
        data_dict = json.load(f)

    pred_labels = np.zeros(Q * n_samples)
    true_labels = np.zeros(Q * n_samples)

    for q in range(Q):
        pred_labels_q = np.zeros(n_samples)
        true_labels_q = np.zeros(n_samples)

        for idx, s_id in enumerate(sample_ids):
            key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

            experiment = data_dict[key]

            target = experiment['target']
            if target == 'not_entailment':
                target = 'contradiction/neutral'

            target_id = class_labels_to_id[target]
            true_labels_q[idx] += target_id
            true_labels[q * n_samples + idx] += target_id

            pred_id = class_labels_to_id[
                experiment["info_answers"][str(q)][0]]
            pred_labels_q[idx] += pred_id
            pred_labels[q * n_samples + idx] += pred_id

        acc = accuracy_score(true_labels_q, pred_labels_q)
        microf1 = f1_score(true_labels_q, pred_labels_q, average='micro')
        macrof1 = f1_score(true_labels_q, pred_labels_q, average='macro')

        table[q] = [acc, microf1, macrof1]

        # print(f"Question {q}, accuracy: {acc}, micro f1-score: {microf1}, macro f1-score: {macrof1}")

    # print(Table(row_headers, col_headers, table))

    print(
        f"Global scores, accuracy: {accuracy_score(true_labels_q, pred_labels_q)}, micro f1-score: {f1_score(true_labels_q, pred_labels_q, average='micro')}, macro f1-score: {f1_score(true_labels_q, pred_labels_q, average='macro')}")

    std_over_microf1 = np.std([table[q][1] for q in range(Q)])
    print(f'Standard deviation of microf1 score: {std_over_microf1}')



def plot_avg_sensitivity(sample_ids: List[int],
                 prompt_types: dict,
                 llms: List[str],
                 Qs: List[int],
                 temp_questions: List[float],
                 As: List[int],
                 temp_answers: List[float],
                 class_labels: List[str],
                 results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    for llm in llms:
        for Q in Qs:
            for A in As:
                for temp_question in temp_questions:
                    for temp_answer in temp_answers:

                        fig_class = plt.figure()
                        fig_q = plt.figure()

                        for prompt_type, prompt in prompt_types.items():
                            filename = f"results_{prompt_type}.json"

                            # Open the JSON file and load its contents into a Python dictionary
                            with open(Path(results_folder, filename),
                                      'r') as f:
                                data_dict = json.load(f)

                            entropy_matrix = np.zeros((Q, n_classes))

                            for q in range(Q):
                                for c in range(n_classes - 1):
                                    # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                                    q_c_distribution = np.zeros(n_classes)

                                    for idx, s_id in enumerate(sample_ids):
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        experiment = data_dict[key]

                                        target = experiment['target']
                                        if target == 'not_entailment':
                                            target = 'contradiction/neutral'

                                        target_id = class_labels_to_id[target]

                                        pred_id = class_labels_to_id[
                                            experiment["info_answers"][str(q)][0]]

                                        if target_id != c:
                                            continue

                                        q_c_distribution[pred_id] += 1

                                    entropy_matrix[q, c] = entropy(q_c_distribution)/np.log(n_classes)

                            avg_entropy_per_class = entropy_matrix.mean(axis=0)
                            avg_entropy_per_q = entropy_matrix.mean(axis=1)

                            plt.figure(fig_class)

                            #plt.scatter(np.arange(n_classes), avg_entropy_per_class, label=prompt_type)
                            plt.plot(avg_entropy_per_class, label=prompt_type)
                            # Add error bars (standard deviation)
                            # plt.errorbar(np.arange(avg_entropy_per_class.shape[0]), avg_entropy_per_class, yerr=entropy_matrix.std(axis=0), fmt='-o',  solid_capstyle='projecting', capsize=5, label=prompt_type)

                            plt.ylabel('Avg Entropy')
                            plt.xticks(np.arange(len(class_labels)),
                                                class_labels,
                                                rotation='vertical')

                            plt.figure(fig_q)
                            #plt.scatter(np.arange(Q), avg_entropy_per_q, label=prompt_type)
                            # plt.plot(avg_entropy_per_q, label=prompt_type)
                            # Add error bars (standard deviation)
                            plt.errorbar(np.arange(avg_entropy_per_q.shape[0]), avg_entropy_per_q, yerr=entropy_matrix.std(axis=1), fmt='-o',  solid_capstyle='projecting', capsize=5, label=prompt_type)


                            plt.ylabel('Avg Entropy')
                            plt.xticks(np.arange(Q),
                                                [f'Question {i+1}' for i in range(Q)],
                                                rotation='vertical')

                        plt.figure(fig_class)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(Path(results_folder,
                                         f'avg_entropy_per_class_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                        plt.show()

                        plt.figure(fig_q)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(Path(results_folder,
                                         f'avg_entropy_per_question_{prompt_type}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}.pdf'))
                        plt.show()


def print_test_sensitivity_over_samples(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path,
                       noise_amount=0.):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    entropy_matrix = np.zeros(n_samples)

    for idx, s_id in enumerate(sample_ids):
        # compute mean of entropy over samples (over samples) distribution over predicted class labels for a given question and class
        c_distribution = np.zeros(n_classes)

        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        for q in range(Q):

            pred_id = class_labels_to_id[
                experiment["info_answers"][str(q)][0]]

            # Generate a random float between 0 and 1
            random_float = random.random()

            # If the generated float is less than or equal to p, pick a random value from the list
            if random_float < noise_amount:
                pred_id = random.choice([i for i in range(len(class_labels))])

            c_distribution[pred_id] += 1

        entropy_matrix[idx] = entropy(c_distribution)/np.log(n_classes)

    avg_entropy_over_samples = entropy_matrix.mean()
    std_entropy_over_samples = entropy_matrix.std()

    print(f"Avg Entropy over samples: {avg_entropy_over_samples},"
          f"Std over samples: {std_entropy_over_samples}")

    return entropy_matrix

def sensitivity_per_class(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    sensitivity_matrix = [[] for _ in range(n_classes)]

    s_ids = [[] for _ in range(n_classes)]
    for c in range(n_classes):

        for idx, s_id in enumerate(sample_ids):
            key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

            experiment = data_dict[key]
            target = experiment['target']
            if target == 'not_entailment':
                target = 'contradiction/neutral'

            target_id = class_labels_to_id[target]

            if target_id != c:
                continue

            c_distribution = np.zeros(n_classes)
            for q in range(Q):
                # compute aggregated (over samples) distribution over predicted class labels for a given question and class
                pred_id = class_labels_to_id[
                    experiment["info_answers"][str(q)][0]]
                c_distribution[pred_id] += 1

            s_ids[c].append(s_id)
            sensitivity_matrix[c].append(entropy(c_distribution)/np.log(n_classes))


    # print(class_labels)
    # print(list(zip(sensitivity_matrix[class_labels_to_id['Description']], s_ids[class_labels_to_id['Description']])))

    return sensitivity_matrix

def get_predicted_distribution(sample_ids: List[int],
                       prompt_type: str,
                       llm: str,
                       Q: int,
                       temp_question: float,
                       A: int,
                       temp_answer: float,
                       class_labels: List[str],
                       results_folder: Path):

    n_samples = len(sample_ids)
    class_labels_to_id = {c_name: i for i, c_name in enumerate(class_labels)}
    assert "N/A" in class_labels
    n_classes = len(class_labels)  # MUST INCLUDE NA

    filename = f"results_{prompt_type}.json"

    # Open the JSON file and load its contents into a Python dictionary
    with open(Path(results_folder, filename),
              'r') as f:
        data_dict = json.load(f)

    samples_distributions = np.zeros((n_samples, n_classes))

    for idx, s_id in enumerate(sample_ids):
        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

        experiment = data_dict[key]

        samples_distributions[idx] = np.array(experiment["distribution"])

    return samples_distributions

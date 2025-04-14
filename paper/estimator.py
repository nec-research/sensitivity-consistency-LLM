"""
       Sensitivity and Consistency of Large Language Models

  File:     estimator.py
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
import copy
import random
from typing import List, Tuple

import numpy as np

from util import generate_questions, classify_llm


@DeprecationWarning
class MultilabelEstimator:
    """
    Initialize the uncertainty estimator for LLMs in multilabel classification tasks.
    A multilabel classification task is a task where multiple categories have to be predicted,
    and each category represents a binary classification task (present/not present)

    Attributes:
    llm (str): The name or identifier of the language model whose uncertainty is to be estimated.
    temperature_question (float): The temperature parameter controlling the randomness of questions.
    temperature_answer (float): The temperature parameter controlling the randomness of responses.
    num_questions (int): The number of alternative questions to be generated.
    num_answers (int): The number of alternative outputs to be generated.
    llm_rephraser (str, optional): The name or identifier of an optional
        language model rephraser to generate alternative questions. If not specified,
        the parameter llm will be used instead.
    """

    def __init__(
        self,
        llm: str,
        temperature_question: float,
        temperature_answer: float,
        num_questions: int,
        num_answers: int,
        llm_rephraser: str = None,
    ):

        self.llm = llm
        self.temperature_question = temperature_question
        self.temperature_answer = temperature_answer
        self.num_questions = num_questions
        self.num_answers = num_answers
        self.llm_rephraser = llm_rephraser

        assert self.num_answers >= 1 and self.num_questions >= 1
        assert 0.0 <= self.temperature_question <= 2.0
        assert 0.0 <= self.temperature_answer <= 2.0

    def estimate_quantities(
        self,
        prompt: List[Tuple[str]],
        question: str,
        summary: str,
        labels: List[str],
        class_extractor_fun,
        debug=False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Estimates the uncertainty of the LLM in classifying a question.
        To account for situations where the predicted class of the LLM

        :param prompt: the prompt, including the question, as accepted by
            LLMs APIs: a List of tuples containing strings
        :param question: the string corresponding to the question that
            needs to be replaced in the prompt with alternative versions
        :param summary: the summary to be replaced in the prompt
        :param labels: the list of possible labels
        :param class_extractor_fun: a function that, given the LLM response, produces a list of predicted labels (multilabel classification)
        :return: an array of label counts, counting how many times a label was predicted for Q*A total calls to the LLM, and a dictionary containing detailed info about the LLM prediction
        """

        num_labels = len(labels)
        label_counts = np.zeros(num_labels)
        info_dict = {}

        if debug:  # generate fake data

            for q in range(self.num_questions):
                # generate list of potential answers for a specific question

                info_dict[q] = {"question": question, "answers": []}

                for i in range(self.num_answers):
                    probability = 0.5
                    # Generate a list of True/False values based on the probability
                    selection = [random.random() < probability for _ in labels]

                    # Use list comprehension to pick elements with True values
                    pred_labels = [
                        elem
                        for elem, select in zip(labels, selection)
                        if select
                    ]

                    for l, label_name in enumerate(labels):
                        # update label_counts with predicted classes
                        if label_name in pred_labels:
                            label_counts[l] += 1

                    info_dict[q]["answers"].append(pred_labels)

            return label_counts, info_dict

        # generate list of alternative questions
        alt_questions = generate_questions(
            self.llm if not self.llm_rephraser else self.llm_rephraser,
            question,
            self.temperature_question,
            self.num_questions,
        )

        assert len(alt_questions) == self.num_questions, (
            "The number of generated questions is different from "
            "what expected, please check the generate_questions method"
        )

        modified_prompts = []
        for q, alt_question in enumerate(alt_questions):
            # generate list of potential answers for a specific question

            info_dict[q] = {"question": alt_question, "answers": []}

            # generate a new prompt with an alternative version of the original question
            modified_prompt = self._replace_question(
                prompt, question, alt_question
            )
            modified_prompts.append(modified_prompt)

        answers = classify_llm(
            self.llm,
            modified_prompts,
            summary,
            self.temperature_answer,
            no_parallel_calls=self.num_answers,
        )

        for q, alt_question in enumerate(alt_questions):
            for k in range(self.num_answers):

                answer = answers[f"{str(q)}_{str(k)}"].content

                # extract list of classes from LLM answer
                pred_labels = class_extractor_fun(answer)

                for l, label_name in enumerate(labels):
                    # update label_counts with predicted classes
                    if label_name in pred_labels:
                        label_counts[l] += 1

                info_dict[q]["answers"].append(pred_labels)

        return label_counts, info_dict

    def _replace_question(
        self, prompt: List[List[str]], question: str, alternative_question: str
    ):
        prompt_copy = copy.deepcopy(prompt)

        for i, (_, txt) in enumerate(prompt_copy):
            prompt_copy[i][1] = txt.replace(question, alternative_question)

        return prompt_copy

    def compute_entropy(
        self, label_counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions' entropy as a form of uncertainty.

        :param label_counts: the number of partial predictions for each label,
            assuming a multilabel classification scenario. The parameter counts
            the number of time a label has been included in a prediction,
             regardless of the ground truth.
        :return: tuple of numpy arrays, the first containing the entropy for
            each label and the second containing the probability that a label
             was predicted over all queries to the LLM
        """
        label_distribution = label_counts / (
            self.num_questions * self.num_answers
        )
        label_entropy = self._compute_entropy(label_distribution)

        return label_entropy, label_distribution

    def _compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute the entropy of each Bernoulli distribution specified by the given probabilities.

        :param probabilities (numpy.ndarray): Array of probabilities
            (parameters) for Bernoulli distributions.

        :return: array containing the entropy of each Bernoulli distribution.
        """
        # Ensure probabilities are within valid range [0, 1]
        probabilities = np.clip(probabilities, 0.00001, 1.0)  # Avoid log(0)

        # Compute entropy for each Bernoulli distribution
        entropy_values = -probabilities * np.log2(probabilities) - (
            1 - probabilities
        ) * np.log2(1 - probabilities)

        return entropy_values


class MulticlassEstimator:
    """
    Initialize the uncertainty estimator for LLMs in classification tasks.

    Attributes:
    llm (str): The name or identifier of the language model whose uncertainty is to be estimated.
    temperature_question (float): The temperature parameter controlling the randomness of questions.
    num_questions (int): The number of alternative questions to be generated.
    llm_rephraser (str, optional): The name or identifier of an optional
        language model rephraser to generate alternative questions. If not specified,
        the parameter llm will be used instead.
    """

    def __init__(
        self,
        llm: str,
        temperature_question: float,
        num_questions: int,
        llm_rephraser: str = None,
    ):

        self.llm = llm
        self.temperature_question = temperature_question
        self.num_questions = num_questions
        self.llm_rephraser = llm_rephraser

        assert self.num_questions >= 1
        assert 0.0 <= self.temperature_question

    def generate_questions(self, question: str) -> List[str]:
        """
        Generates all possible alternative questions given a question.
        :param question: the question to be generated
        :return: a list of all possible alternative questions, with the first
            question being the original one.
        """
        alt_questions = generate_questions(
            self.llm if not self.llm_rephraser else self.llm_rephraser,
            question,
            self.temperature_question,
            self.num_questions,
        )

        assert len(alt_questions) == self.num_questions, (
            "The number of generated questions is different from "
            "what expected, please check the generate_questions method"
        )

        return alt_questions

    def generate_answers(
        self,
        modified_prompts: List[str],
        summary: str,
        temperature_answer: float,
        num_answers: int,
    ) -> dict:
        """
        Asks the LLM to generate one answer for each different prompt, where
        the difference lies in the question that has been rewritten. The
        prompts expect an extra input text, called summary.
        :param modified_prompts: the list of prompts with the modified question
        :param summary:
        :param temperature_answer (float): The temperature parameter controlling the randomness of responses.
        :param num_answers (int): The number of alternative outputs to be generated.
        :return: A dictionary where the keys are the id of the question
         and the id of the answer (both starting from 0)
        """
        answers = classify_llm(
            self.llm,
            modified_prompts,
            summary,
            temperature_answer,
            no_parallel_calls=num_answers,
        )
        return answers

    def estimate_quantities(
        self, answers: dict, labels: List[str], class_extractor_fun
    ) -> Tuple[np.ndarray, dict]:
        """
        Estimates the uncertainty of the LLM in classifying a question.
        To account for situations where the predicted class of the LLM
        :param answers: the answers produced by the generate_answers method
        :param labels: the list of possible labels
        :param class_extractor_fun: a function that, given the LLM response, produces a list of predicted labels (multiclass classification)
        :return: an array of label counts, counting how many times a label was predicted for Q*A total calls to the LLM, and a dictionary containing detailed info about the LLM prediction
        """
        num_labels = len(labels)
        label_counts = np.zeros(num_labels)
        info_dict = {}
        num_answers = len(answers.keys()) // self.num_questions

        for q_id in range(self.num_questions):
            info_dict[q_id] = []

            for k in range(num_answers):

                answer = answers[f"{str(q_id)}_{str(k)}"].content

                # extract list of classes from LLM answer
                pred_labels = class_extractor_fun(answer)

                for l, label_name in enumerate(labels):
                    # update label_counts with predicted classes
                    if label_name in pred_labels:
                        label_counts[l] += 1

                info_dict[q_id].append(pred_labels)

        return label_counts, info_dict

    def replace_question(
        self, prompt: List[List[str]], question: str, alternative_question: str
    ):
        prompt_copy = copy.deepcopy(prompt)

        for i, (_, txt) in enumerate(prompt_copy):
            prompt_copy[i][1] = txt.replace(question, alternative_question)

        return prompt_copy

    def compute_entropy(
        self, label_counts: np.ndarray, num_answers: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions' entropy as a form of uncertainty.

        :param label_counts: the number of partial predictions for each label,
            assuming a multilabel classification scenario. The parameter counts
            the number of time a label has been included in a prediction,
             regardless of the ground truth.
        :param num_answers (int): The number of alternative outputs
        :return: tuple of numpy arrays, the first containing the entropy for
            each label and the second containing the probability that a label
             was predicted over all queries to the LLM
        """
        label_distribution = label_counts / (self.num_questions * num_answers)
        label_entropy = self._compute_entropy(label_distribution)

        return label_entropy, label_distribution

    def _compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Computes the entropy of the multiclass distribution specified by the given probabilities.

        :param probabilities (numpy.ndarray): Array of probabilities
            (parameters) for Bernoulli distributions.

        :return: array containing the entropy of each Bernoulli distribution.
        """
        # Ensure probabilities are within valid range [0, 1]
        probabilities = np.clip(probabilities, 0.00001, 1.0)  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))

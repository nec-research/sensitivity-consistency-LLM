"""
       Sensitivity and Consistency of Large Language Models

  File:     evaluator.py
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
import os
from pathlib import Path
from typing import List, Callable

import ray

from estimator import MulticlassEstimator


def run_grid_search(
    samples: List[dict],
    llms: List[str],
    prompt_types: dict,
    Qs: List[int],
    temp_questions: List[float],
    question_to_rewrite: str,
    As: List[int],
    temp_answers: List[float],
    class_labels: List[str],
    class_extractor_fun: Callable,
    results_folder: Path,
):
    """
    Run Grid Search on the given samples.
    :param samples: List of dictionaries containing sample information. Each
        dictionary should have the following keys: input, class
    :param llms: list of llms to use
    :param prompt_types: dictionary with different types of prompts
    :param Qs: number of rephrasings of the question to rewrite
    :param temp_questions: temperature of the llm rephraser (same as llm)
    :param question_to_rewrite: self-descriptive
    :param As: number of calls to the llm that classifies inputs
    :param temp_answers: temperature of the llm that classifies inputs
    :param class_labels: ordered list of labels
    :param class_extractor_fun: function to extract class from the llm answer
    :param results_folder: folder where results will be stored
    """
    for prompt_type, prompt_original in prompt_types.items():
        data_dict = {}
        filename = f"results_{prompt_type}.json"

        not_done = True
        while not_done:
            try:

                ray.shutdown()
                ray.init()

                if os.path.exists(Path(results_folder, filename)):
                    # Open the JSON file and load its contents into a Python dictionary
                    with open(Path(results_folder, filename), "r") as f:
                        data_dict = json.load(f)

                for llm in llms:

                    if "mixtral" in llm:  # move "system" text to "user"

                        prompt = prompt_original[1:]
                        prompt[0][
                            1
                        ] = f"{prompt_original[0][1]} {prompt[0][1]}"
                        print(prompt)
                    else:
                        prompt = prompt_original

                    for Q in Qs:
                        for temp_question in temp_questions:
                            key_questions = f"{llm}_{Q}_{temp_question}"

                            # if alternative versions have already been generated in the simple prompt method,
                            # reuse them to ensure consistency across experiments
                            simple_filename = f"results_simple.json"

                            # Initialize LLM and ask generate alternative questions for the task once for all samples
                            llm_quantities_estimator = MulticlassEstimator(
                                llm=llm,
                                temperature_question=temp_question,
                                num_questions=Q,
                                llm_rephraser=None,
                            )

                            if (
                                prompt_type != "simple"
                                and os.path.exists(
                                    Path(results_folder, simple_filename)
                                )
                                and not key_questions in data_dict
                            ):
                                print(
                                    'Reusing same prompt variants from "simple" approach...'
                                )
                                with open(
                                    Path(results_folder, simple_filename), "r"
                                ) as f:
                                    simple_data_dict = json.load(f)
                                    alt_questions = simple_data_dict[
                                        key_questions
                                    ]

                                    data_dict[key_questions] = alt_questions

                                    json_data = json.dumps(data_dict)
                                    with open(
                                        Path(results_folder, filename), "w"
                                    ) as f:
                                        f.write(json_data)
                            else:
                                if not key_questions in data_dict:

                                    # Generate different prompts to be reused acrosed all samples
                                    alt_questions = llm_quantities_estimator.generate_questions(
                                        question=question_to_rewrite
                                    )

                                    data_dict[key_questions] = alt_questions

                                    json_data = json.dumps(data_dict)
                                    with open(
                                        Path(results_folder, filename), "w"
                                    ) as f:
                                        f.write(json_data)

                                        # else do not recompute
                                else:
                                    alt_questions = data_dict[key_questions]

                            modified_prompts = [
                                llm_quantities_estimator.replace_question(
                                    prompt, question_to_rewrite, alt_question
                                )
                                for alt_question in alt_questions
                            ]

                            for A in As:
                                for temp_answer in temp_answers:
                                    futures = []

                                    @ray.remote
                                    def task(
                                        task_key, task_input, task_target
                                    ):
                                        done = False
                                        while not done:
                                            try:
                                                answers = llm_quantities_estimator.generate_answers(
                                                    modified_prompts,
                                                    task_input,
                                                    temp_answer,
                                                    A,
                                                )
                                            except Exception as e:
                                                print(e)
                                                continue
                                            done = True
                                        label_counts, info_dict = (
                                            llm_quantities_estimator.estimate_quantities(
                                                answers,
                                                labels=class_labels,
                                                class_extractor_fun=class_extractor_fun,
                                            )
                                        )
                                        label_entropy, label_distribution = (
                                            llm_quantities_estimator.compute_entropy(
                                                label_counts, A
                                            )
                                        )

                                        return (
                                            task_key,
                                            label_counts,
                                            label_entropy,
                                            label_distribution,
                                            info_dict,
                                            task_target,
                                        )

                                    # Parallelize across samples
                                    for sample in samples:

                                        s_id = sample["id"]

                                        # Unique ID of the experiment
                                        key = f"{s_id}_{llm}_{Q}_{A}_{temp_question}_{temp_answer}"

                                        if (
                                            key in data_dict
                                        ):  # do not recompute the same row
                                            print(f"Skipping test {key}")
                                            continue

                                        futures.append(
                                            task.remote(
                                                key,
                                                sample["input"],
                                                sample["class"],
                                            )
                                        )
                                        print(f"Job {key} submitted")

                                    waiting = futures
                                    while waiting:
                                        completed, waiting = ray.wait(waiting)
                                        for future in completed:
                                            (
                                                task_key,
                                                label_counts,
                                                label_entropy,
                                                label_distribution,
                                                info_dict,
                                                target,
                                            ) = ray.get(future)
                                            (
                                                s_id,
                                                llm,
                                                num_q,
                                                num_a,
                                                tmp_q,
                                                tmp_a,
                                            ) = task_key.split("_")

                                            num_labels = len(class_labels)

                                            data_dict[task_key] = dict(
                                                id=s_id,
                                                llm=llm,
                                                Q=int(num_q),
                                                A=int(num_a),
                                                target=target,
                                                temp_question=float(tmp_q),
                                                temp_answer=float(tmp_a),
                                                entropy=float(label_entropy),
                                                counts=list(label_counts),
                                                distribution=list(
                                                    label_distribution
                                                ),
                                                info_answers=info_dict,
                                            )

                                        # Serialize the dictionary to a JSON string
                                        json_data = json.dumps(data_dict)

                                        # Write the JSON string to a file
                                        with open(
                                            Path(results_folder, filename), "w"
                                        ) as f:
                                            f.write(json_data)

                not_done = False
            except Exception as e:
                print(e)

    ray.shutdown()

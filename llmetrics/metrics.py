# Sensitivity and Consistency of Large Language Models
#
#     File: metrics.py
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


import numpy as np
from typing import List, Tuple

def _compute_entropy( probabilities: np.ndarray) -> np.ndarray:
        """
        Computes the entropy of the multiclass distribution specified by the given probabilities.

        :param probabilities (numpy.ndarray): Array of probabilities
            (parameters) for Bernoulli distributions.

        :return: array containing the entropy of each Bernoulli distribution.
        """
        # Ensure probabilities are within valid range [0, 1]
        probabilities = np.clip(probabilities, 0.00001, 1.0)  # Avoid log(0)
        return -np.sum(probabilities * np.log2(probabilities))

def compute_entropy(
    num_promt_versions: int,
    label_counts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the predictions' entropy as a form of uncertainty.
        
        :param num_promt_versions: The number of alternative questions
        :param label_counts: the number of partial predictions for each label,
            assuming a multilabel classification scenario. The parameter counts
            the number of time a label has been included in a prediction,
             regardless of the ground truth.
        :return: tuple of numpy arrays, the first containing the entropy for
            each label and the second containing the probability that a label
             was predicted over all queries to the LLM
        """
        label_distribution = label_counts / (num_promt_versions)
        label_entropy = _compute_entropy(label_distribution)/len(label_counts)

        return label_entropy, label_distribution


def TVD(distribution1, distributions):
    return 0.5*np.abs(np.expand_dims(distribution1, 0) - distributions).sum(1)

def consistency_matrix(distributions: np.ndarray):
    _consistency_matrix = np.zeros((distributions.shape[0], distributions.shape[0])) # shape: (n_samples, n_samples) 
    for idx, row in enumerate(distributions):
        _consistency_matrix[idx, :] = 1. - TVD(row, distributions)
    return _consistency_matrix

def compute_consistency(TVD_matrix: np.ndarray) -> float:
    consistency = 0
    if TVD_matrix.shape[0] == 1 and TVD_matrix.shape[1] == 1:
        return np.nan
        # return TVD_matrix[0,0] # TODO: check if this is correct
    for idx, row in enumerate(TVD_matrix):
        _row = np.concatenate((row[:idx], row[idx+1:]))
        consistency += _row.sum()
    consistency = consistency / (TVD_matrix.shape[0]*TVD_matrix.shape[1] - TVD_matrix.shape[0])
    return consistency


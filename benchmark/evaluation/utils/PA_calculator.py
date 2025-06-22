"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import regex as re


def calculate_parsing_accuracy(dataset, groundtruth_df, parsedresult_df, filter_templates=None):

    # parsedresult_df = pd.read_csv(parsedresult)
    # groundtruth_df = pd.read_csv(groundtruth)
    if filter_templates is not None:
        groundtruth_df = groundtruth_df[groundtruth_df['EventTemplate'].isin(filter_templates)]
        parsedresult_df = parsedresult_df.loc[groundtruth_df.index]
    correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()

    # parsedresult_df["groundtruth"] = groundtruth_df["EventTemplate"]
    # df = parsedresult_df[parsedresult_df['EventTemplate'] != parsedresult_df['groundtruth']]
    # df.to_csv(f"/Users/wuyifan/PycharmProjects/LogPub-main/result/bad_case_{dataset}.csv", columns=["Content", "EventTemplate", "groundtruth"], index=False)
    total_messages = len(parsedresult_df[['Content']])

    PA = float(correctly_parsed_messages) / total_messages

    # similarities = []
    # for index in range(len(groundtruth_df)):
    #     similarities.append(calculate_similarity(groundtruth_df['EventTemplate'][index], parsedresult_df['EventTemplate'][index]))
    # SA = sum(similarities) / len(similarities)
    # print('Parsing_Accuracy (PA): {:.4f}, Similarity_Accuracy (SA): {:.4f}'.format(PA, SA))
    print('Parsing_Accuracy (PA): {:.4f}'.format(PA))
    return PA

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

import os
import time
import csv
from evaluation.utils.GA_calculator import calculate_group_accuracy
from evaluation.utils.PA_calculator import calculate_parsing_accuracy
from evaluation.utils.template_level_analysis import evaluate_template_level
import pandas as pd


def prepare_results(output_dir):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary.csv'
    with open(os.path.join(output_dir, result_file), 'w') as csv_file:
        fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['Dataset', 'parse_time', 'identified_templates',
                     'ground_templates', 'GA', 'PA', 'FGA', 'PTA', 'RTA', 'FTA'])

    return result_file


def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return len(content) == 0


def evaluator(
        dataset,
        input_dir,
        output_dir,
        log_file,
        LogParser,
        param_dict,
        result_file
):
    """
    Unit function to run the evaluation for a specific configuration.

    """

    print('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    groundtruth = os.path.join(indir, log_file_basename + '_structured.csv')
    parsedresult = os.path.join(output_dir, log_file_basename + '_structured.csv')
    # identify templates using Drain
    start_time = time.time()
    if LogParser is not None:
        print("start parsing.")
        parser = LogParser(**param_dict)
        print(param_dict)
        parser.parse(log_file_basename)
        print("end parsing.")
        parse_time = time.time() - start_time  # end_time is the wall-clock time in seconds
    else:
        parse_time = -1
    print("parsing time: ", parse_time)

    if not os.path.exists(parsedresult) or is_file_empty(parsedresult):
        print("No output file generated.")
        result = dataset + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + '\n'

        with open(os.path.join(output_dir, result_file), 'a') as summary_file:
            summary_file.write(result)
        return

    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    groundtruth = pd.read_csv(groundtruth, dtype=str)

    print("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = calculate_group_accuracy(groundtruth, parsedresult)

    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))

    # calculate parsing accuracy
    start_time = time.time()
    PA = calculate_parsing_accuracy(dataset, groundtruth, parsedresult)
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(dataset, groundtruth, parsedresult)
    TA_end_time = time.time() - start_time
    print('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    result = dataset + ',' + \
             "{:.2f}".format(parse_time) + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             "{:.1f}".format(GA * 100) + ',' + \
             "{:.1f}".format(PA * 100) + ',' + \
             "{:.1f}".format(FGA * 100) + ',' + \
             "{:.1f}".format(PTA * 100) + ',' + \
             "{:.1f}".format(RTA * 100) + ',' + \
             "{:.1f}".format(FTA * 100) + '\n'

    with open(os.path.join(output_dir, result_file), 'a') as summary_file:
        summary_file.write(result)

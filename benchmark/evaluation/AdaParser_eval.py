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
import argparse
import sys
import os

sys.setrecursionlimit(3000)
sys.path.append('../')

from logparser.AdaParser import LogParser
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average

datasets = [
    "Hadoop",
    "HDFS",
    "OpenStack",
    "Spark",
    "Zookeeper",
    "BGL",
    "HPC",
    "Thunderbird",
    "Linux",
    "Mac",
    "Apache",
    "OpenSSH",
    "HealthApp",
    "Proxifier"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default="gpt-3.5-turbo-0125")
    parser.add_argument('--log_ratio', type=str,
                        default="20")
    args = parser.parse_args()

    input_dir = f"../../full_dataset/"
    output_dir = f"../../result/result_AdaParser_full"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = prepare_results(
        output_dir=output_dir
    )

    for dataset in datasets:
        log_file = f"{dataset}/{dataset}_full.log"
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        if os.path.exists(os.path.join(output_dir, f"{dataset}_full.log_structured.csv")):
            parser = None
            print("parseing result exist.")
        else:
            parser = LogParser
        # run evaluator for a dataset
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            LogParser=parser,
            param_dict={
                'indir': indir, 'outdir': output_dir, 'model': args.model, 'log_ratio': args.log_ratio
            },
            result_file=result_file
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, "AdaParser")

# Log Parsing using LLMs with Self-Generated In-Context Learning and Self-Correction

This is the replication package for Log Parsing using LLMs with Self-Generated In-Context Learning and Self-Correction.

## Repository Organization 

```
├── benchmark/
│   ├── evaluation/ # the evaluation code of AdaParser
│   └── logparser/ # the implementation code of AdaParser
├── full_dataset/ # Please download and unzip full datasets into this directory
│   └── sampled_examples # Our saved sampled candidates for different ratios of available logs
├── openai_key.txt # the OpenAI api key and address
├── README.md
└── requirements.txt
```

## Quick Start

### Datasets

Please first download the large-scale datasets for log parsing in Loghub-2.0 from [Zenodo](https://zenodo.org/record/8275861) and unzip these datasets into the directory of `full_dataset`.

###  Installation

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```

### Execution

Please first add an OpenAI API key (`sk-xxxx`) into the first line of openai_key.txt.

```bash
cd benchmark/evaluation
python AdaParser_eval.py --model [model] --log_ratio [log_ratio]
```

The parsed results and evaluation results will be saved in the `result/` directory.
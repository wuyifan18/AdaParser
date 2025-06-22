import json
import logging
import re
import string
import time

import openai
from dateutil import parser
from openai import OpenAI

from .post_process import correct_single_template

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def get_openai_key(file_path):
    with open(file_path, 'r') as file:
        api_key = file.readline().strip()
        base_url = file.readline().strip()
    return api_key, base_url

api_key, base_url = get_openai_key('../../openai_key.txt')
print(api_key)
print(base_url)

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    max_retries=0
)

regs_common = []
with open("../logparser/AdaParser/common.json") as fr:
    dic = json.load(fr)
patterns = dic['COMMON']['regex']
for pattern in patterns:
    regs_common.append(re.compile(pattern))

compiled_pattern1 = re.compile(r"\{\w+}")
punc = "!\"#$%&'()+,-./:;=?[]^_`{|}~@"
splitters = "\\s\\" + "\\".join(punc)
compiled_pattern2 = re.compile(r"([{}])".format(splitters))
complied_pattern3 = re.compile(r"^[A-Z_]+$")
excluded_str = {'=', '|', '(', ')', ':', '/'}
translation_table = str.maketrans('', '', ''.join(set(punc) - excluded_str))
lower_camel = re.compile(r'^[a-z]+([A-Z][a-z]*)*$')


def post_process_template(template):
    template = compiled_pattern1.sub("<*>", template)
    for reg in regs_common:
        template = reg.sub("<*>", template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            template = template.replace("<*>", "{variables}")
            return template, True
    # print("Get a too general template. Error.")
    template = template.replace("<*>", "{variables}")
    return template, False


def is_datetime_string(s):
    try:
        parser.parse(s)
        return True
    except (ValueError, OverflowError):
        return False


def is_camel_case(s):
    return bool(lower_camel.match(s))


def read_json_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_dict = json.loads(line)
            data.append(json_dict)
    return data


def gpt_call(message, model="gpt-3.5-turbo-0125", max_retries=100, temperature=0.0):
    for i in range(max_retries):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                seed=0
            )
            prompt_tokens = result.usage.prompt_tokens
            completion_tokens = result.usage.completion_tokens
            res = result.choices[0].message.content
            return res, prompt_tokens, completion_tokens
        except (openai.APITimeoutError, openai.InternalServerError, openai.APIConnectionError, openai.APIStatusError, TypeError) as e:
            logging.warning(f"Retry {i + 1}/{max_retries}: {e}")
            time.sleep(1)
    logging.error("Exceeded maximum retry number")
    return None


def custom_key(key):
    return (1, key) if key.startswith('<') else (0, key)


def post_process_tokens(tokens):
    processed_tokens = []
    for token in tokens:
        if "<*>" in token:
            if processed_tokens and processed_tokens[-1] == "<*>":
                continue
            processed_tokens.append("<*>")
        else:
            new_str = token.translate(translation_table)
            if new_str:
                processed_tokens.append(new_str)
    return processed_tokens


def message_split(message):
    tokens = string_split(message)
    tokens = post_process_tokens(tokens)
    return tokens


def string_split(tokens):
    tokens = compiled_pattern1.sub("<*>", tokens)
    tokens = compiled_pattern2.split(tokens)
    tokens = [token.strip() for token in tokens if token.strip()]
    return tokens


def LCS_similarity(t1, t2):
    t1 = string_split(t1)
    t2 = string_split(t2)

    def lcs(X, Y):
        m = len(X)
        n = len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if X[i - 1] == Y[j - 1]:
                lcs.insert(0, X[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return lcs

    similarity = 2 * len(lcs(t1, t2)) / (len(t1) + len(t2))
    return similarity

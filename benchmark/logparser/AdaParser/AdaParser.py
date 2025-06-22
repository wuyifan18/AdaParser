import hashlib
import os

import pandas as pd
import regex as re

from .Trie import Trie
from .utils import gpt_call, read_json_file, LCS_similarity, is_datetime_string, post_process_template

parsing_prompt = "I want you to act like an expert of log parsing. I will give you a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {variables} and output a static log template. Print the input log's template delimited by backticks."
constant_refine_prompt = '''The token {} may not be dynamic variables and do not need to be abstracted. Please provide a revised log template.'''
post_refine_prompt = "The log template can not match the log message via regular expression. There may extra {{variables}}, punctuations or spaces. If there are typos, do not fix it. Please provide a revised log template."

pattern1 = re.compile(r"\{\w+}")
pattern2 = re.compile(r"[^ ]+")
pattern3 = re.compile(r"^[A-Za-z\s/]*$")
pattern4 = re.compile(r"(.*?:)")
pattern5 = re.compile(r"/(<\*>|\w)+/?")


class LogParser:
    def __init__(self, indir, outdir, model, log_ratio):
        self.indir = indir
        self.outdir = outdir
        self.model = model
        self.log_ratio = log_ratio
        self.df_log = None
        self.trie = Trie()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def query_template_from_ChatGPT(self, logMessage, messages, temperature, msg):
        pred_template, prompt_tokens, completion_tokens = gpt_call(messages, model=self.model, temperature=temperature)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        pred_template = pred_template.replace("Log template:", "")
        pred_template = pred_template.replace("{non-variable}", "{variables}")
        start_index = pred_template.find('`') + 1
        end_index = pred_template.rfind('`')
        if start_index != 0 and end_index != -1 and start_index < end_index:
            pred_template = pred_template[start_index:end_index].strip("`").strip()
        pred_template, flag = post_process_template(pred_template)
        if not flag:
            pred_template, flag = post_process_template(logMessage)
        print(f"{msg} ({temperature}): {pred_template}")
        return pred_template

    def example_select(self, examples, logMessage, candidate_num=3):
        for example in examples:
            example["sim"] = LCS_similarity(logMessage, example["query"])
        examples.sort(key=lambda item: item['sim'])
        return examples[-candidate_num:]

    def match_template(self, pred_template, logMessage):
        rex = pattern1.sub("WILDCARD", pred_template)
        rex = pattern2.sub(lambda x: re.escape(x.group(0)), rex)
        rex = rex.replace("WILDCARD", "(\\S.*){0,1}")
        return re.fullmatch(rex, logMessage)

    def post_process_nomatch(self, pred_template, logMessage, history_messages, temperature):
        if not self.match_template(pred_template, logMessage):
            if set(pred_template.split()) - set(logMessage.split()) == {"{variables}"}:
                pred_template = pred_template.replace("{variables}", "").strip()
                print(f"match: {pred_template}")
            else:
                messages = history_messages + ([{"role": "assistant", "content": f"Log template: `{pred_template}`"},
                                                {"role": "user", "content": post_refine_prompt.format(pred_template, logMessage)}])
                pred_template = self.query_template_from_ChatGPT(logMessage, messages, temperature, msg="match")

            if not self.match_template(pred_template, logMessage):
                return pred_template, False
        return pred_template, True

    def post_process_constant(self, pred_template, logMessage, history_messages, temperature):

        def get_constants(pred_template):
            if pred_template == logMessage:
                return ""
            rex = pattern1.sub("WILDCARD", pred_template)
            rex = pattern2.sub(lambda x: re.escape(x.group(0)), rex)
            rex = rex.replace("WILDCARD", "(.*)")
            match = re.findall(rex, logMessage)
            constants = []
            for tokens in match:
                tokens = tokens if isinstance(tokens, tuple) else [tokens]
                for token in tokens:
                    if is_datetime_string(token):
                        continue
                    if (re.search(r'(?<!Exception)(?<!interrupt)(?<!interrupted)(?<!thrown)(?<!failure)(?<!Error)(?<!read)(?<!Kickstart)(?<!install)(?<!because)'
                                  r'(?<!address)(?<!died)(?<!failed)(?<!Reason)(?<!Diagnostics)(?<!job)(?<!Responder)(?!<disconnected)(?<!answers)(?<!tftp)(?<!:)'
                                  r'([=:])\s*' + re.escape(token), logMessage) and
                            all(excluded_str not in token for excluded_str in ["exception", "Exception", "killed", "failed", "Failed", "connected", "Connection"])):
                        continue
                    tmp = []
                    subtoken = token.split()
                    for t in subtoken:
                        if pattern3.match(t) and not t.isupper() and not pattern5.match(t):
                            tmp.append(t)
                    if tmp:
                        if tmp == subtoken:
                            tmp = pattern4.split(' '.join(subtoken))
                            tmp = list(filter(lambda x: x != "", tmp))
                        constants.extend(f"\"{t.strip()}\"" for t in tmp)
            return constants

        constants = get_constants(pred_template)
        last_constant = None
        max_retry = 3
        while constants and max_retry > 0:
            if constants[0] == last_constant:
                break
            last_constant = constants[0]
            messages = history_messages + ([{"role": "assistant", "content": f"Log template: `{pred_template}`"},
                                            {"role": "user", "content": constant_refine_prompt.format(constants[0])}])
            pred_template = self.query_template_from_ChatGPT(logMessage, messages, temperature, msg="constant")
            constants = get_constants(pred_template)
            max_retry -= 1

        if get_constants(pred_template) or not self.match_template(pred_template, logMessage):
            return pred_template, False
        else:
            return pred_template, True

    def parse(self, logName):
        file_path = os.path.join(self.indir, logName)
        print("Parsing file: " + file_path)

        dataset_name = logName.split('_')[0]
        candidates = read_json_file(f"../../full_dataset/sampled_examples_{self.log_ratio}%/{dataset_name}/32shot.json")
        for d in candidates:
            self.trie.insert(d["answer"])

        self.load_data(file_path)
        query_num = 0
        total_lines = len(self.df_log)

        for logID, logMessage in enumerate(self.df_log["Content"], start=1):
            logMessage = logMessage.strip()
            stop_node, flag = self.trie.search(logMessage)
            if flag:
                stop_node.logIDs.append(logID)
            else:
                query_num += 1
                print(f"{logID}/{total_lines}: {logMessage} (query times: {query_num})")
                messages = [
                    {"role": "system", "content": "You are an expert of log parsing, and now you will help to do log parsing."},
                    {"role": "user", "content": parsing_prompt},
                    {"role": "assistant", "content": "Sure, I can help you with log parsing."}
                ]
                examples = self.example_select(candidates, logMessage, candidate_num=3)
                for example in examples:
                    messages.append({"role": "user", "content": f"Log message: `{example['query']}`"})
                    messages.append({"role": "assistant", "content": f"Log template: `{example['answer']}`"})
                messages.append({"role": "user", "content": f"Log message: `{logMessage}`"})

                count = 0
                flag1, flag2 = False, False
                pred_template = ""
                while count < 3 and not (flag1 and flag2):
                    temperature = count * 0.5
                    pred_template = self.query_template_from_ChatGPT(logMessage, messages, temperature, msg="pred")
                    pred_template, flag1 = self.post_process_nomatch(pred_template, logMessage, messages, temperature)
                    pred_template, flag2 = self.post_process_constant(pred_template, logMessage, messages, temperature)
                    count += 1
                if not flag1:
                    print(f"not match!!! {pred_template}")
                else:
                    candidates.append({"query": logMessage, "answer": pred_template})

                self.trie.update(pred_template, stop_node, logID)

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.outputResult(logName)
        print(f"dataset: {logName}, total_prompt_tokens: {self.total_prompt_tokens}, total_completion_tokens: {self.total_completion_tokens}")

    def outputResult(self, logName):
        log_templateids = [""] * self.df_log.shape[0]
        log_templates = [""] * self.df_log.shape[0]
        df_events = []

        def dfs(node):
            if node.is_end_of_token:
                template_id = hashlib.md5(node.tokens.encode('utf-8')).hexdigest()[0:8]
                for logID in node.logIDs:
                    logID -= 1
                    log_templateids[logID] = template_id
                    log_templates[logID] = re.sub(r"\{\w+}", "<*>", node.tokens)
                df_events.append([template_id, node.tokens, len(node.logIDs)])
            for child in node.children.values():
                dfs(child)

        dfs(self.trie.root)
        df_events = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        self.df_log.to_csv(os.path.join(self.outdir, logName + '_structured.csv'), index=False)
        df_events.to_csv(os.path.join(self.outdir, logName + '_templates.csv'), index=False)

    def load_data(self, file_path):
        csv_path = os.path.join(file_path + '_structured.csv')
        if os.path.exists(csv_path):
            self.df_log = pd.read_csv(csv_path)

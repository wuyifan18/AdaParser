import re
from collections import defaultdict

from sortedcontainers import SortedDict

from .utils import LCS_similarity, custom_key, message_split, post_process_template, is_camel_case


class TrieNode:
    def __init__(self, token=None):
        self.children = SortedDict(custom_key)
        self.token = token
        self.is_end_of_token = False
        self.logIDs = []
        self.tokens = ""


class Trie:
    def __init__(self):
        self.root = TrieNode("RootTrieNode")

    def merge_templates(self, similarity, group_templates, event_template):
        template_length = len(event_template.split())
        if not all(len(template.split()) == template_length for template in group_templates):
            return ""

        template_lists = [re.split(r"([ ,():\[\]])", template) for template in group_templates + [event_template]]
        merged_template_tokens, merged_tokens = [], set()

        for tokens in zip(*template_lists):
            if len(set(tokens)) == 1:
                merged_template_tokens.append(tokens[0])
            else:
                if similarity > 0.8 and all(t.isalpha() and len(t) != 1 for t in tokens) and len(tokens) < 5:
                    return ""
                if all("{variables}" in t for t in tokens):
                    return ""
                merged_template_tokens.append("{variables}")
                merged_tokens.add(tokens)

        merged_template = "".join(merged_template_tokens)
        if "{variables}{variables}" in merged_template:
            return ""
        merged_template, flag = post_process_template(merged_template)
        if not flag or "{variables} {variables}" in merged_template:
            return ""
        if similarity <= 0.8:
            if any(len(t) < 5 for t in merged_tokens) or all(is_camel_case(s) for t in merged_tokens for s in t):
                return ""

        return merged_template

    def update(self, event_template, stop_node, logID):
        logIDs = [logID]
        clusters = defaultdict(list)
        if (event_template.count("{variables}") + 1) / len(message_split(event_template)) <= 0.5:
            relevant_templates = self.get_related_templates(stop_node, event_template)
            for template in relevant_templates:
                clusters[template["sim"]].append(template["template"])
            for similarity, group_templates in clusters.items():
                merged_template = self.merge_templates(similarity, group_templates, event_template)
                if merged_template:
                    print(similarity, group_templates)
                    print(f"=> {merged_template}")
                    logIDs += [log_id for template in group_templates for log_id in self.delete(template)]
                    event_template = merged_template

        self.insert(event_template, logIDs)

    def insert(self, event_template, logID=None):
        node = self.root
        for token in message_split(event_template):
            if token not in node.children:
                node.children[token] = TrieNode(token)
            node = node.children[token]
        node.is_end_of_token = True
        node.tokens = event_template
        if logID:
            node.logIDs.extend(logID if isinstance(logID, list) else [logID])

    def delete(self, event_template):
        node, parents = self.root, []
        tokens = message_split(event_template)
        for token in tokens:
            parents.append((token, node))
            node = node.children[token]

        node.is_end_of_token = False
        logIDs = node.logIDs
        for token, parent in reversed(parents):
            if node.children or node.is_end_of_token:
                break
            del parent.children[token]
            node = parent

        return logIDs

    def search(self, logMessage):
        message_tokens = message_split(logMessage)
        message_length = len(message_tokens)

        def dfs(node, index):
            if index == message_length:
                if node.is_end_of_token:
                    return node, node.is_end_of_token
                elif "<*>" in node.children:
                    return node.children["<*>"], node.children["<*>"].is_end_of_token
                else:
                    return node, node.is_end_of_token

            token = message_tokens[index]
            if token in node.children:
                matched_node, is_complete = dfs(node.children[token], index + 1)
                if is_complete:
                    return matched_node, is_complete

            if "<*>" in node.children:
                flag = True
                for skip in range(index, message_length + 1):
                    if (skip + 2 < message_length
                            and node.token == "="
                            and message_tokens[skip].isalpha()
                            and message_tokens[skip + 1] == node.token
                            and message_tokens[skip] not in node.children["<*>"].children):
                        if not message_tokens[skip + 2].isdigit():
                            flag = False
                        elif flag and message_tokens[skip + 2].isdigit():
                            break
                    matched_node, is_complete = dfs(node.children["<*>"], skip)
                    if is_complete:
                        return matched_node, is_complete
            return node, False

        return dfs(self.root, 0)

    def get_related_templates(self, node, pred_templates):
        relevant_templates = []
        similarity = LCS_similarity(pred_templates, node.tokens)
        if node.is_end_of_token:
            relevant_templates.append({
                "template": node.tokens,
                "sim": similarity
            })
        for child in node.children.values():
            templates = self.get_related_templates(child, pred_templates)
            relevant_templates.extend(templates)
        return relevant_templates

    def print_trie(self, node=None, path=None):
        if node is None:
            node = self.root
            path = []

        if node != self.root:
            path.append(node.token)

        if node.is_end_of_token:
            print(' -> '.join(path), node.logIDs)

        for child in node.children.values():
            self.print_trie(child, list(path))

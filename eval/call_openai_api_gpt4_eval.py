import argparse
import fcntl
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from time import sleep

import jsonlines
import numpy as np
import requests
from openai import OpenAI
from tenacity import (RetryError, retry, stop_after_attempt,
                      wait_random_exponential)

role_consist_prompt = {
    "zh":
    "你是一个擅长判别对话者角色身份的模型，如下是发生在用户和一个角色之间的对话，其中用户的指令的开头用[指令]注释，而角色的回复的开头使用[回复]注释。\n\n[指令]\n[开始]\n{query}\n[结束]\n\n[回复]\n[开始]\n{answer}\n[结束]\n\n如下是该角色可能的身份：\n{cands}\n\n请根据回复的内容从上述可能的角色身份中选择一个最符合当前对话中回复者的身份的角色。请先对问题进行一步一步的分析，最后以\"最符合的角色为[角色名]\"的句式回答。",
    "en":
    "You are a model that excels at discerning the identities of dialogue participants. Here is a conversation between a user and a character, where the user's commands are annotated with [Instruction], and the character's replies are annotated with [Response].\n\n[Instruction]\n[Begin]\n{query}\n[End]\n\n[Response]\n[Begin]\n{answer}\n[End]\n\nThe possible identities for the character are as follows:{cands}\n\nPlease analyze the content of the replies step by step and select the character identity that best fits the responder in the current conversation. Please conclude your answer by stating, \"The most suitable role is [character name].\""
}

rejection_judgement_prompt = {
    "zh":
    "如下是用户与角色{label}的对话。其中用户的指令的开头用[指令]注释，而角色的回复的开头使用[回复]注释。\n\n[指令]\n[开始]\n{query}\n[结束]\n\n[回复]\n[开始]\n{answer}\n[结束]\n\n请判断该角色是否因为自身知识有限或其他原因拒绝回答问题。如果拒答则回复1，否则回复0。请不要回复除了1和0外的其他内容。",
    "en":
    "The following is a conversation between the user and the role {label}. Instructions from the user are annotated with [Instruction], while replies from the role are annotated with [Response].\n\n[Instruction]\n[begin]\n{query}\n[End]\n\n[Response]\n[Begin]\n{answer}\n[End]\n\nPlease determine whether the role refuses to answer the question due to limited knowledge or other reasons. Reply with 1 if the role refuses, otherwise reply with 0. Please do not reply with any content other than 1 and 0."
}

eval_prompt = {
    "en":
    """You are an expert for checking the quality of the answer.
During the conversation, the assistant is playing the role of {label}. Here is the profile of {label}: {summary}.

Please act as an impartial judge and evaluate the correctness of the response provided by an AI assistant to the user question displayed below.
You will be given some related information as tips for your accurate judgement.

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]

[Begin of Tips]
{evidence}
[End of Tips]

Please provide a comprehensive analysis of the judgement in the first line.
In the subsequent line, please output a single line containing only a single value indicating the score for answer.
Please do not generate other information but a single number for the score. The score ranges from 1 to 10, with higher being better and smaller being worse.""",
    "zh":
    """你是一个检查人工智能助手回复质量的专家。
在对话过程中，助手需要扮演{label}的角色。以下是{label}的简介：{summary}。
请充当一个公正的评委，并评估AI助手在扮演{label}的前提下对用户问题给出的回答的正确性。
为了准确判断，如下提供了一些相关信息作为参考。

[问题]
{question}

[人工智能助手回复开始]
{answer}
[人工智能助手回复结束]

[提示的开始部分]
{evidence}
[提示的结束部分]

请在回复的第一行提供对评分的全面分析。
在接下来的一行中，请输出一个只包含一个数值的单独行，表示对人工智能助手回复的评分。
请不要生成其他信息，只输出一个分数。分数范围从1到10，分数越高表示越好，分数越低表示越差。
"""
}

MAX_API_RETRY = 10
LLM_MIT_RETRY_SLEEP = 5

client = OpenAI()


@retry(wait=wait_random_exponential(min=5, max=60),
       stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(
            '-------------------------------------------------------------------------------------'
        )
        print(e)
        print("kwargs", kwargs)
        print(
            '-------------------------------------------------------------------------------------'
        )
        raise e


def get_consist_score(single_conv, cands, lang, label, api_config):
    prompt = role_consist_prompt[lang].format(query=single_conv[0]['content'],
                                              answer=single_conv[1]['content'],
                                              cands=cands)
    res = completion_with_backoff(messages=[{
        "role": "user",
        "content": prompt
    }],
                                  **api_config)
    responses = [each.message.content for each in res.choices]
    choices = [
        re.search(
            r"最符合的角色为\[(.*)\]"
            if lang == 'zh' else r"most suitable role is \[(.*)\]", response)
        for response in responses
    ]
    score = np.mean([c.group(1) == label for c in choices if c is not None])
    if np.isnan(score):
        score = 0
    return score


def get_reject_score(single_conv, lang, label, api_config):
    prompt = rejection_judgement_prompt[lang].format(
        query=single_conv[0]['content'],
        answer=single_conv[1]['content'],
        label=label)
    res = completion_with_backoff(messages=[{
        "role": "user",
        "content": prompt
    }],
                                  **api_config)
    responses = [each.message.content for each in res.choices]
    choices = [
        int(response) for response in responses if response in ('0', '1')
    ]
    label = Counter(choices).most_common(1)[0][0]
    return label


def get_knowledge_score(single_conv, lang, label, summary, evidence,
                        api_config):
    prompt = eval_prompt[lang].format(question=single_conv[0]['content'],
                                      answer=single_conv[1]['content'],
                                      label=label,
                                      summary=summary,
                                      evidence=evidence)
    res = completion_with_backoff(messages=[{
        "role": "user",
        "content": prompt
    }],
                                  **api_config)
    response = np.mean([
        int(each.message.content.split("\n")[-1].replace("Score: ", ""))
        for each in res.choices
    ])
    return response


def call_api(item: dict, model: str, api_config=None):
    lang = item['meta']['lang']
    label = item['meta']['label']
    summary = item['meta']['summary']
    evidence = item['meta']['evidence']
    cands = "\n".join(
        ["- " + cand for cand in item['meta']['role_consist_cands']])

    conv = item['messages'][1:]
    consist_scores, reject_scores, knowledge_scores = [], [], []
    for i in range(0, len(conv), 2):
        single_conv = conv[i:i + 2]
        consist_score = get_consist_score(single_conv, cands, lang, label,
                                          api_config)
        reject_score = get_reject_score(single_conv, lang, label, api_config)
        knowledge_score = get_knowledge_score(single_conv, lang, label,
                                              summary, evidence[i // 2],
                                              api_config)
        consist_scores.append(consist_score)
        reject_scores.append(reject_score)
        knowledge_scores.append(knowledge_score)

    item['consist_scores'] = consist_scores
    item['reject_scores'] = reject_scores
    item['knowledge_scores'] = knowledge_scores
    return item


def process(input_items, model, output_path, fail_path, requests_per_minute,
            api_config):
    with ProcessPoolExecutor(max_workers=requests_per_minute) as executor:
        for item_index, item in enumerate(input_items):
            executor.submit(call_api_and_save,
                            item=item,
                            model=model,
                            output_path=output_path,
                            fail_path=fail_path,
                            api_config=api_config)
            sleep(1 / requests_per_minute * 60)


def call_api_and_save(item: dict, model: str, output_path: str, fail_path: str,
                      api_config):
    try:

        output_item = call_api(item, model, api_config)
        success = True
    except Exception as e:
        print(e)
        raise e
        success = False

    if success:
        output_line = json.dumps(output_item, ensure_ascii=False)
        with open(output_path, "a") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            output_file.write(output_line + "\n")
            fcntl.flock(output_file, fcntl.LOCK_UN)
    else:
        fail_line = json.dumps(item, ensure_ascii=False)
        with open(fail_path, "a") as fail_file:
            fcntl.flock(fail_file, fcntl.LOCK_EX)
            fail_file.write(fail_line + "\n")
            fcntl.flock(fail_file, fcntl.LOCK_UN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--topp", type=float, default=1)
    parser.add_argument("--call_per_minute", type=int, default=60)
    args = parser.parse_args()

    api_config = {
        'temperature': args.temperature,
        'n': args.n,
        'max_tokens': 2048,
        'model': args.model
    }

    with open(args.input_file) as f:
        input_items = [json.loads(line) for line in f.readlines()]
    if args.limit != 0:
        input_items = input_items[:args.limit]

    process(
        input_items,
        args.model,
        os.path.join("output", f"{args.task}_output.json"),
        os.path.join("output", f"{args.task}_fail.json"),
        args.call_per_minute,
        api_config,
    )

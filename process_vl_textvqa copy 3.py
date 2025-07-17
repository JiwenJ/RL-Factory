# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile
import random
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
# DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant. "
# instruction_following = (
#     "You are a math expert. Answer the given question. You must conduct reasoning inside <think> and </think>. "
#     "After reasoning, if you can not get the anwser, the format for action is <tool_call>{\"name\":\"image_edit\", \"arguments\": {\"instruction\": \"top|down|left|right\"}}</tool_call>. "
#     "For example,  <think> I think I need to crop the image </think> <tool_call>{\"name\":\"image_flip\", \"arguments\": {\"instruction\": \"top\"}}</tool_call>. Question:"
# )
# instruction_following = (
#     r"You FIRST think about the reasoning process as an internal monologue and make necessary tool call, then provide the final answer. "
#     r"The reasoning process MUST BE enclosed within <think> </think> tags. The tool call MUST BE enclosed within <tool_call> </tool_call> tags. The final answer MUST BE enclosed within <answer> </answer> tags. "
# )

# instruction_following = (
#     r"Given a question and an image, you must answer the question based on the image. Follow these steps strictly: "
#     r"First, analyze the question and image inside <think>...</think> tags. "
#     r"If you find the image is rotated or unclear or upside down, you MUST call the `image_flip` tool to rotate the image ONCE per turn using this exact format: <tool_call> {\"name\":\"image_flip\",\"arguments\":{\"instruction\":\" <degree>\"}} </tool_call>, the degree is int number, means the rorate degree.  "
#     r"You MUST wait the tool result before providing the final answer. If no tool is needed, provide only the final answer inside <answer>...</answer> "
#     r"Key rules: Never call multiple tools in one response; Never include explanations in <answer>; Call Function inside the <tool_call> and </tool_call> XML tags; Always reason in <think> before acting. Should not answer directly when waiting the tool result. "
# )
# instruction_following = (
#     r"Given a question and an image, you must answer the question based on the image. Follow these steps strictly: "
#     r"First, analyze the question and image inside <think>...</think> tags. "
#     r"If you find the image is rotated or unclear or upside down, you MUST call the `rotate` tool to rotate the image ONCE per turn using this exact format: <tool_call> {\"name\":\"rotate\",\"arguments\":{\"degree\":\" <number>\"}} </tool_call>, set the number from 0 to 360.  "
#     r"You MUST wait the tool result before providing the final answer. If no tool is needed, provide only the final answer inside <answer>...</answer> "
#     r"Key rules: Never call multiple tools in one response; Never include explanations in <answer>; Call Function inside the <tool_call> and </tool_call> XML tags; Always reason in <think> before acting. Should not answer directly when waiting the tool result. "
# )
instruction_following = (
    r"Given a question and an image, answer the question based strictly on visual content. Examine the image carefully and identify any recognizable entities, such as faces, objects, locations, events, logos, or text. "
    r"Determine whether the image is not upside down and you are confident recognize the main visual element and answer the user's question. If so, first explain your reasoning, then provide a clear and direct answer. If you find the image is upside down and are unable to confidently identify the visual element, stop and invoke the image tool by appending the string <tool_call><function json parameter></tool_call> at the end of your response. You must include your reasoning inside <think>...</think> before taking any action, whether it is calling the image tool, generating a text search query, or providing a final answer. The reasoning may involve analysis of the original image and question, interpretation of search results, or logical steps leading to the final answer. "
    r"All tool results will be placed inside <tool_response> and </tool_response> and returned to you. When you are ready to answer the question, wrap your final answer between <answer> and </answer>, without detailed illustrations. For example: <answer>Titanic</answer>. "
)
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Data/textvqav20")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/hiyouga/geometry3k"
    data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/lmms-lab/textvqa/data/train-00000-of-00020.parquet"
    data_source1 = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/lmms-lab/textvqa/data/validation-00000-of-00003.parquet"
    # dataset = datasets.load_dataset(data_source)
    train_dataset = datasets.load_dataset("parquet",data_files = data_source)["train"]
    test_dataset = datasets.load_dataset("parquet",data_files = data_source1)["train"]
    # breakpoint()
    

    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    # )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("question")
            # prompt = problem + " " + instruction_following
            prompt = instruction_following + "Question:\n" + "<image>" + problem
            answer = example.pop("answers")
            images = example.pop("image")
            images = images.resize((1024,1024))
            angle = random.choice([90, 180, 270])
            # breakpoint()
            # Rotate the image
            images1 = [images.rotate(angle, expand=True)]
            # breakpoint()
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Every turn you can call one function at most among the following functions to assist with the user query. "
                            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
                            '{"type": "function", "function": {"name": "rotate", "description": "Rotate a Pillow image by specified degrees", "parameters": {"type": "object", "properties": {"degree": {"type": "integer", "description": "Rotation angle in degrees"}}, "required": ["degree"]}}}\n</tools>\n'
                            'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n'
                            '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\nThe tool response will wrapped with <tool_response></tool_response> XML tags. '
                            
                            
)
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images1,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
            }
            # breakpoint()
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=10)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=10)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

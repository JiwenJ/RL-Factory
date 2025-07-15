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
instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and make necessary tool call, then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. The tool call MUST BE enclosed within <tool_call> </tool_call> tags. The final answer MUST BE enclosed within <answer> </answer> tags. "
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
    parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Data/textvqav1")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/hiyouga/geometry3k"
    data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/lmms-lab/textvqa/data/train-00000-of-00020.parquet"
    data_source1 = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/lmms-lab/textvqa/data/test-00001-of-00004.parquet"
    # dataset = datasets.load_dataset(data_source)
    train_dataset = datasets.load_dataset("parquet",data_files = data_source)["train"]
    test_dataset = datasets.load_dataset("parquet",data_files = data_source)["train"]
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
            problem = example.pop("problem")
            # prompt = problem + " " + instruction_following
            prompt = problem + " \n"
            answer = example.pop("answer")
            images = example.pop("images")
            angle = random.choice([90, 180, 270])
            # breakpoint()
            # Rotate the image
            images1 = [images[0].rotate(angle, expand=True)]
            # breakpoint()
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. Given a question and an image, you must answer the question based on the image. Follow these steps strictly: "
                            "First, analyze the question and image inside <think>...</think> tags. "
                            "If the image is rotated/unclear, you may call the image_flip tool ONCE per turn using this exact format: <tool_call> {\"name\":\"image_flip\",\"arguments\":{\"instruction\":\"rotate degree from 0 to 360\"}} </tool_call> "
                            "If no tool is needed, provide only the final answer inside <answer>...</answer> (e.g., <answer>12</answer>). "
                            "Key rules: Never call multiple tools in one response; Never include explanations in <answer>; Always reason in <think> before acting. "
                            "Question: "
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
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=1)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=1)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

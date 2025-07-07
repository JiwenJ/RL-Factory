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
    parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/Data/geo3kv12")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "/mnt/dolphinfs/hdd_pool/docker/share/jjw/visual_tool/huggingface.co/datasets/hiyouga/geometry3k"

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

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
            angle = random.choice([0, 90, 180, 270])
            # breakpoint()
            # Rotate the image
            images = [images[0].rotate(angle, expand=True)]
            # breakpoint()
            

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and a image, you need to answer the question based on the image information. You must conduct reasoning inside <think> and </think> first every time you get new information. "
                            "After reasoning, if you find the image is somehow ratoted, you can call a image tool inside <tool_call> and </tool_call> to see the image clearly, and the tool will return the wanted image and some tool execution information. "
                            "This time, you have the `image_flip` tool, which can help you rotate the image. The valid format is <tool_call> {\"name\":\"image_flip\", \"arguments\": {\"instruction\": \"top or down or left or right\"}} </tool_call>. "
                            "Each time you can call tool at most once. You MUST NOT call two tools in one turn. If you find no tool needed, you can directly provive the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> 12 </answer>. "
                            "Question: "
)
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
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

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=20)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=20)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

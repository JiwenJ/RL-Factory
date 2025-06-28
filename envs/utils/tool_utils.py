import torch
import itertools
import numpy as np
from verl import DataProto
import torch.distributed as dist
from tensordict import TensorDict
from typing import List
from PIL import Image


class ToolUtils:
    def __init__(self, tokenizer, processor, meta_info, config, env_object):
        self.tokenizer = tokenizer  
        self.processor = processor
        self.final_str = config.stop[-1] if config.stop else ''
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length
        self.stop_id = self.tokenizer.encode(config.stop[0], add_special_tokens=False)[0]
        self.max_turns = config.max_turns
        self.max_prompt_length = config.prompt_length

        
        pad_token_id = meta_info.get('pad_token_id')
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            eos_token_id = meta_info.get('eos_token_id')
            if isinstance(eos_token_id, (list, tuple)):
                self.pad_token_id = eos_token_id[-1]
            else:
                self.pad_token_id = eos_token_id
                
        eos_token_id = meta_info.get('eos_token_id')
        if isinstance(eos_token_id, (list, tuple)):
            self.eos_token_id = eos_token_id[0]
        else:
            self.eos_token_id = eos_token_id
        
        self.meta_info = meta_info
        self.loop_cnt = 0

        self.env_object = env_object
        
        # Qwen2.5VL mrope specific parameters
        self.vision_start_token_id = getattr(tokenizer, 'vision_start_token_id', 151652)
        self.image_token_id = getattr(tokenizer, 'image_token_id', 151655) 
        self.video_token_id = getattr(tokenizer, 'video_token_id', 151656)
        self.spatial_merge_size = 2
        self.tokens_per_second = 2

    def _calculate_mrope_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None, second_per_grid_ts=None):
        """
        Calculate proper 3D position IDs for Qwen2.5VL mrope.
        This function handles the temporal, height, and width dimensions properly.
        """
        if image_grid_thw is None and video_grid_thw is None:
            # Pure text case
            position_ids_2d = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
            # Replicate for 3 dimensions (temporal, height, width)
            position_ids = position_ids_2d.unsqueeze(0).expand(3, -1, -1)
            return position_ids
            
        batch_size, seq_len = input_ids.shape
        position_ids = torch.ones(3, batch_size, seq_len, dtype=input_ids.dtype, device=input_ids.device)
        
        # Process each sequence in the batch
        for i in range(batch_size):
            seq_input_ids = input_ids[i][attention_mask[i] == 1]
            seq_tokens = seq_input_ids.tolist()
            
            # Find vision tokens
            vision_start_indices = torch.argwhere(seq_input_ids == self.vision_start_token_id).squeeze(-1)
            if vision_start_indices.numel() == 0:
                # No vision tokens, use standard text position encoding
                seq_len_valid = attention_mask[i].sum()
                text_pos_ids = torch.arange(seq_len_valid, device=input_ids.device)
                position_ids[:, i, attention_mask[i] == 1] = text_pos_ids.unsqueeze(0).expand(3, -1)
                continue
                
            vision_tokens = seq_input_ids[vision_start_indices + 1] if vision_start_indices.numel() > 0 else torch.tensor([], device=input_ids.device)
            image_nums = (vision_tokens == self.image_token_id).sum().item()
            video_nums = (vision_tokens == self.video_token_id).sum().item()
            
            # Calculate position IDs with vision awareness
            pos_ids_list = []
            st = 0
            image_index, video_index = 0, 0
            remain_images, remain_videos = image_nums, video_nums
            
            for _ in range(image_nums + video_nums):
                # Find next vision token
                ed_image = len(seq_tokens) + 1
                ed_video = len(seq_tokens) + 1
                
                if self.image_token_id in seq_tokens[st:] and remain_images > 0:
                    ed_image = seq_tokens.index(self.image_token_id, st)
                if self.video_token_id in seq_tokens[st:] and remain_videos > 0:
                    ed_video = seq_tokens.index(self.video_token_id, st)
                    
                if ed_image < ed_video:
                    # Process image
                    if image_grid_thw is not None and image_index < len(image_grid_thw):
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                    else:
                        t, h, w = 1, 1, 1
                        second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    # Process video
                    if video_grid_thw is not None and video_index < len(video_grid_thw):
                        t, h, w = video_grid_thw[video_index]
                        if second_per_grid_ts is not None and video_index < len(second_per_grid_ts):
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                    else:
                        t, h, w = 1, 1, 1
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                
                # Add text tokens before vision
                text_len = ed - st
                if text_len > 0:
                    st_idx = pos_ids_list[-1].max() + 1 if len(pos_ids_list) > 0 else 0
                    text_pos_ids = torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    pos_ids_list.append(text_pos_ids)
                
                # Add vision tokens with proper mrope encoding
                llm_grid_t = t.item() if hasattr(t, 'item') else t
                llm_grid_h = (h.item() if hasattr(h, 'item') else h) // self.spatial_merge_size
                llm_grid_w = (w.item() if hasattr(w, 'item') else w) // self.spatial_merge_size
                
                # Calculate temporal, height, width position IDs
                vision_seq_len = llm_grid_t * llm_grid_h * llm_grid_w
                if vision_seq_len > 0:
                    # Temporal dimension
                    t_indices = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1, 1).expand(-1, llm_grid_h, llm_grid_w)
                    t_indices = (t_indices * second_per_grid_t * self.tokens_per_second).long().flatten()
                    
                    # Height dimension  
                    h_indices = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    
                    # Width dimension
                    w_indices = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    
                    vision_offset = pos_ids_list[-1].max() + 1 if len(pos_ids_list) > 0 else 0
                    vision_pos_ids = torch.stack([t_indices, h_indices, w_indices]) + vision_offset + text_len
                    pos_ids_list.append(vision_pos_ids)
                
                st = ed + vision_seq_len
            
            # Add remaining text tokens
            if st < len(seq_tokens):
                remaining_text_len = len(seq_tokens) - st
                st_idx = pos_ids_list[-1].max() + 1 if len(pos_ids_list) > 0 else 0
                text_pos_ids = torch.arange(remaining_text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                pos_ids_list.append(text_pos_ids)
            
            # Concatenate and assign to position_ids
            if pos_ids_list:
                seq_position_ids = torch.cat(pos_ids_list, dim=1)
                valid_len = attention_mask[i].sum().item()
                if seq_position_ids.shape[1] >= valid_len:
                    position_ids[:, i, attention_mask[i] == 1] = seq_position_ids[:, :valid_len]
                else:
                    # Fallback to standard position encoding if something went wrong
                    text_pos_ids = torch.arange(valid_len, device=input_ids.device)
                    position_ids[:, i, attention_mask[i] == 1] = text_pos_ids.unsqueeze(0).expand(3, -1)
        
        return position_ids

    def _extract_multimodal_grid_info(self, multi_modal_data, batch_idx=0):
        """
        Extract grid information for mrope calculation from multimodal data.
        This is a helper method that can be extended based on specific environment needs.
        """
        image_grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None
        
        if multi_modal_data is not None and len(multi_modal_data) > batch_idx:
            # Try to extract grid information from the processor or environment
            # This would need to be customized based on how your environment stores this info
            sample_data = multi_modal_data[batch_idx]
            if isinstance(sample_data, dict) and "image" in sample_data:
                images = sample_data["image"]
                if isinstance(images, list) and len(images) > 0:
                    # For now, use default grid dimensions
                    # In practice, you'd want to extract these from the actual image processing
                    image_grid_thw = []
                    for img in images:
                        if isinstance(img, Image.Image):
                            # Default grid for images - these should be extracted from actual processing
                            h, w = img.size
                            # This is a simplified calculation - actual grid should come from processor
                            grid_h = max(1, h // 224)  # Assuming 224x224 patches
                            grid_w = max(1, w // 224)
                            image_grid_thw.append(torch.tensor([1, grid_h, grid_w]))
                    
                    if image_grid_thw:
                        image_grid_thw = torch.stack(image_grid_thw)
        
        return image_grid_thw, video_grid_thw, second_per_grid_ts

    def postprocess_output_tp(self, output: DataProto, image_data: List[List[Image.Image]], step: int=2):
        '''output: cpu'''

        # init loop responses token
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.tool_use = [[] for _ in range(self.batch_size)]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.end_flags = [False for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]  

            batch_idxs = list(range(self.batch_size))
            # Remove breakpoint for production use
            # breakpoint()
            for idx in range(self.batch_size):
                prompt_token = self.init_prompt_token[idx]
                prompt_token_list = torch.tensor(prompt_token)[torch.tensor(prompt_token) != self.pad_token_id].tolist()
                self.loop_responses_token[idx].append(prompt_token_list)
        else:
            batch_idxs = output.meta_info['index']

        responses = output.batch.get('responses')

        process_response = []
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            if self.env_object.use_process_reward:
            # assure last token is stop token （add or change）
                if response_token_list[-1] != self.stop_id:
                    if len(response_token_list) != self.config_response_length:
                        response_token_list.append(self.stop_id)
                    else:
                        response_token_list[-1] = self.stop_id
            self.loop_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)
        
        # decode responses for env step (detect tool call)
        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )

        infos_str, dones, _, _ = self.env_object.step(
            responses=responses_str, tokenizer=self.tokenizer, image_data=image_data
        )
        
        #if not use_process_reward will be 0
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str)
        else:
            step_scores = [0] * len(responses_str)
        
        # encode infos for next prompt
        info_tokens = self.tokenizer(infos_str).input_ids
        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []
        for idx, batch_idx in enumerate(batch_idxs):
            # 只在第一次未结束时添加response_token_list
            if not self.end_flags[batch_idx]:
                response_token = responses[idx]
                response_token_list = response_token[response_token != self.pad_token_id].tolist()
                self.loop_responses_token[batch_idx].append(response_token_list)
                # get process reward
                self.tool_use[batch_idx].append(step_scores[idx])

            # 如果done了，设置end_flag
            if dones[idx]:
                self.end_flags[batch_idx] = True

            # info_token_list只在未done时添加
            if not dones[idx] and not self.end_flags[batch_idx]:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)

            next_sample_idx.append(batch_idx)
            promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
            next_prompt_token.append(promt_token)
            next_prompt_length.append(len(promt_token))
        
        # left pad
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        # position_ids = (torch.cumsum(next_attention_mask, dim=1) - 1) * next_attention_mask
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask
        
        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:].cpu().share_memory_(),
                'position_ids': position_ids[:, -max_len:].cpu().share_memory_(),
                'attention_mask': next_attention_mask[:, -max_len:].to(dtype=torch.int64).cpu().share_memory_()
            },
            batch_size=next_input_ids.shape[0]
        ).share_memory_()
        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        # raw_prompt_ids[:] = [np.array(x[-max_len:]) for x in next_prompt_token]
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1

        return next_data
    
    def postprocess_output(self, output: DataProto, step: int=2):
        '''output: cpu'''
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            # self.mo = [[] for _ in range(self.batch_size)]
            self.loop_raw_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            self.raw_prompt_id = output.non_tensor_batch.get('raw_prompt_ids')
            # self.init_prompt_token = output.non_tensor_batch.get('raw_prompt_ids')

            self.multi_modal_inputs = [[] for _ in range(self.batch_size)]
            # self.init_prompt_ids = output.batch.get('prompts')
            # self.init_prompt_token = output.non_tensor_batch.get('raw_prompt_ids')
            self.image_list = [i["image"] for i in output.non_tensor_batch["multi_modal_data"]]
            # self.rollout_n = self.batch_size // len(self.image_data)
            self.tool_use = [[] for _ in range(self.batch_size)]
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:,:prompt_length]

            batch_idxs = list(range(self.batch_size))
            
            for idx in range(self.batch_size):
                self.multi_modal_inputs[idx].append(output.non_tensor_batch["multi_modal_inputs"][idx])
                prompt_token = self.init_prompt_token[idx]
                prompt_token = torch.tensor(prompt_token)
                assert isinstance(prompt_token, torch.Tensor)
                prompt_token_list = torch.tensor(prompt_token[torch.tensor(prompt_token) != self.pad_token_id]).tolist()
                self.loop_responses_token[idx].append(prompt_token_list)
                self.loop_raw_responses_token[idx].append(self.raw_prompt_id[idx])

        else:
            batch_idxs = output.meta_info['index']
        responses = output.batch.get('responses')

        process_response = []
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            self.loop_responses_token[batch_idx].append(response_token_list)
            self.loop_raw_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)

        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )
        
        infos_str, dones, _, _, new_image_data, raw_prompt, multi_modal_data = self.env_object.step(
            responses=responses_str, tokenizer=self.tokenizer, image_data=self.image_list, processor=self.processor
        )
        # breakpoint()
        for idx, batch_idx in enumerate(batch_idxs):
            if multi_modal_data[idx] is not None:
                self.multi_modal_inputs[batch_idx].append(multi_modal_data[idx])

        step_scores = [0] * len(responses_str)


        def tokenize_infos(infos_str):
            # if self.processor:
            #     info_tokens = self.processor(text=infos_str, images=new_image_data, return_tensors="pt").input_ids
            # else:
            info_tokens = self.tokenizer(infos_str).input_ids
            return info_tokens

        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []
        next_image_data = []
        next_raw_prompt_token = []
        next_raw_prompt_length = []
        # next_multi_modal_data = []
        # breakpoint()
        for idx, batch_idx in enumerate(batch_idxs):
            if not dones[idx]:
                info_token_list = tokenize_infos(infos_str[idx])
                raw_prompt_list = tokenize_infos(raw_prompt[idx])
                self.loop_responses_token[batch_idx].append(info_token_list)
                self.loop_raw_responses_token[batch_idx].append(raw_prompt_list)
                next_sample_idx.append(batch_idx)
                promt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
                raw_prompt_token = list(itertools.chain.from_iterable(self.loop_raw_responses_token[batch_idx]))
                next_prompt_token.append(promt_token)
                next_raw_prompt_token.append(raw_prompt_token)
                next_prompt_length.append(len(promt_token))
                next_raw_prompt_length.append(len(raw_prompt_token))
                # get process reward 
                self.tool_use[batch_idx].append(step_scores[idx])
                # append the new image from tool call
                if new_image_data[idx] is not None:
                    self.image_list[idx].append(new_image_data[idx])
                    next_image_data.append(self.image_list[idx])
        # breakpoint()
        if len(next_prompt_token) == 0:
            return 
        
        # left pad
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask
        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:].cpu().share_memory_(),
                'position_ids': position_ids[:, -max_len:].cpu().share_memory_(),
                'attention_mask': next_attention_mask[:, -max_len:].to(dtype=torch.int64).cpu().share_memory_()
            },
            batch_size=next_input_ids.shape[0]
        ).share_memory_()
        
        raw_prompt_ids = np.empty(len(next_raw_prompt_token), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_raw_prompt_token]

        next_image_data = np.array([{"image": img} for img in next_image_data], dtype=object)
        # next_multi_modal_data = np.array(next_multi_modal_data, dtype=object)
        # breakpoint()
        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids, 'multi_modal_data': next_image_data, })
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False # step > 0 does not do sample
        self.loop_cnt += 1

        return next_data

    def compose_final_output(self, step) -> DataProto:
        # breakpoint()
        """Compose final generation output."""
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        raw_prompt_ids_list = []
        
        for idx, responses in enumerate(self.loop_responses_token):
            # loss_mask = [0]*len(responses[0]) # init_prompt loss
            loss_mask = []
            prompts_list = list(itertools.chain.from_iterable(responses[1:]))
            # responses_token: [prompt_token, reponse_token_1, info_token_1, response_token_2....]
            for turn_idx in range(len(responses[1:])): 
                length = len(responses[turn_idx + 1])
                loss_mask.extend([(turn_idx + 1) % 2] * length)
            input_ids_list.append(prompts_list)
            loss_mask_list.append(loss_mask)
            length_list.append(len(prompts_list))
        # breakpoint()
        # max_len = max(max(length_list), self.config_response_length)
        max_response_length = torch.tensor([max(length_list)], device=torch.cuda.current_device())
        dist.all_reduce(max_response_length, op=dist.ReduceOp.MAX)
        max_len = int(max_response_length)
        
        # right pad
        input_ids = []
        loss_mask = []
        for idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            loss_mask = loss_mask_list[idx] + [0] * (max_len - len(loss_mask_list[idx]))
            input_ids_list[idx] = input_ids
            loss_mask_list[idx] = loss_mask[0:max_len]

        response_token = torch.tensor(input_ids_list, dtype=torch.int64)[:,:max_len]
        response_loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)
        response_attention_mask = (response_token != self.pad_token_id).long()

        # get the max length of the process rewards
        max_tool_use_len = self.max_turns
        for tool_use_item in self.tool_use:
            max_tool_use_len = max(max_tool_use_len, len(tool_use_item))
        tool_use_tensor = []

        # Pad tool_use to have consistent dimensions
        for idx in range(len(self.tool_use)):
            if not self.tool_use[idx]:
                padded_tool_use = [torch.nan] * max_tool_use_len
            else:
                padded_tool_use = self.tool_use[idx] + [torch.nan] * (max_tool_use_len - len(self.tool_use[idx]))
            tool_use_tensor.append(padded_tool_use)

        tool_use_score = torch.tensor(tool_use_tensor)
        # breakpoint()
        multi_modal_inputs = np.array(self.merge_multi_modal_inputs(self.multi_modal_inputs))
        # breakpoint()
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        
        if self.processor is not None and self.processor.image_processor._processor_class== "Qwen2_5_VLProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [] 
            for idx, input_id in enumerate(input_ids):
                position_id = get_rope_index(
                        self.processor,
                        input_ids=input_id,
                        image_grid_thw=multi_modal_inputs[idx][0].get("image_grid_thw"),
                        video_grid_thw=multi_modal_inputs[idx][0].get("video_grid_thw"),
                        attention_mask=attention_mask[idx],
                    )
                
                position_ids.append(position_id)
            
            # Stack the 2D position_ids into a 3D tensor
            position_ids = torch.stack(position_ids, dim=0)
            
        else:
            position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
        with open('/root/autodl-tmp/RL-Factory/tmp/tensor_debug.txt', 'w') as f: f.write(str(position_ids))
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask, dtype=torch.float32), response_loss_mask], dim=-1)
        # breakpoint()
        final_batch = TensorDict(
            {
                'prompts': self.init_prompt_token,
                'responses': response_token,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'loss_mask': loss_mask,
                'tool_use_scores': tool_use_score
            },
            batch_size=self.batch_size,
        )  
        # breakpoint()
        image_list = np.array([{"image": img} for img in self.image_list], dtype=object)
        temp = []
        for i in range(len(multi_modal_inputs)):
            temp.append(multi_modal_inputs[i][0])
        # breakpoint()
        final_output = DataProto(batch=final_batch,non_tensor_batch={'multi_modal_data': image_list, "multi_modal_inputs":np.array(temp)})
        print("dsds")
        return final_output



    
    def merge_tensor_dicts(self, list_of_dicts):

        if not list_of_dicts:
            return {}

        keys = list_of_dicts[0].keys()
        merged_dict = {}

        for key in keys:
            tensors_to_concat = [d[key] for d in list_of_dicts]
            merged_dict[key] = torch.cat(tensors_to_concat, dim=0)

        return merged_dict

    def merge_multi_modal_inputs(self,multi_modal_inputs):
        """
        Merges dictionaries within each sublist of multi_modal_inputs.
        """
        merged_inputs = []
        for sublist in multi_modal_inputs:
            merged_dict = self.merge_tensor_dicts(sublist)
            merged_inputs.append([merged_dict])
        return merged_inputs

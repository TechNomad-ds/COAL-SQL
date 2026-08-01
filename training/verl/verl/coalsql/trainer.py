# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.utils.torch_functional import get_response_mask

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
class CoalSQLPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("MyRayPPOTrainer initialized!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset 
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        from verl.trainer.main_ppo import create_rl_sampler

        #breakpoint()

        if train_dataset is None:
            train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         config=self.config.data)
        
        if val_dataset is None:
            val_dataset = RLHFDataset(data_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       config=self.config.data)
        
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _validate(self):
        print("Validation: Generation Begin.")
        
        reward_tensor_lst = []
        data_source_lst = []
        length_lst = []
        acc_lst = []  # store the acc value of each batch

        for test_data in tqdm(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch, validation_or_not=True)

            # obtain response length
            def obtain_reponse_length(output_batch):
                prompt_length = output_batch.batch['prompts'].shape[-1]
                response_length = output_batch.batch['attention_mask'][:,prompt_length:].sum(1).numpy()
                return response_length
            
            length_lst.append(obtain_reponse_length(test_output_gen_batch))
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            # Extract and accumulate acc values (if the reward manager provides acc)
            if 'acc' in test_batch.batch:
                # acc is a tensor, convert it to numpy
                acc_lst.append(test_batch.batch['acc'].cpu().numpy())

        print('Validation: Generation end.')

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        lengths = np.concatenate(length_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_response_lengths = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            if data_source not in data_source_response_lengths:
                data_source_response_lengths[data_source] = []
            data_source_response_lengths[data_source].append(lengths[i])

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        for data_source, lengths in data_source_response_lengths.items():
            metric_dict[f'val/test_length/{data_source}'] = np.mean(lengths)

        # Compute and add the acc metric (if present)
        if acc_lst:
            acc_tensor = np.concatenate(acc_lst, axis=0)  # (total_samples,)
            data_source_acc = {}
            for i in range(acc_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_acc:
                    data_source_acc[data_source] = []
                data_source_acc[data_source].append(acc_tensor[i])

            for data_source, accs in data_source_acc.items():
                metric_dict[f'val/test_acc/{data_source}'] = np.mean(accs)

        return metric_dict

    def _validate_with_save(self):
        print("Validation: Generation Begin.")
    
        # Declare lists to store information for all samples
        all_sample_data = []
        
        # Track the sample index within the loader
        sample_index_counter = 0
    
        for test_data in tqdm(self.val_dataloader):
            test_batch = DataProto.from_single_dict(test_data)
            
            # Only validate for rule-based reward models
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}
    
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch_repeated = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch_repeated.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
    
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    
            test_batch_union = test_batch_repeated.union(test_output_gen_batch)
    
            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch_union, validation_or_not=True)
    
            # Get the generated text responses
            prompt_length = test_output_gen_batch.batch['prompts'].shape[-1]
            response_ids = test_output_gen_batch.batch['responses'][:, prompt_length:]
            # responses = self.tokenizer.decode(response_ids) # may need changes depending on the tokenizer's batch decode
            # If tokenizer.decode cannot handle the whole batch, iterate instead
            responses = [self.tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in response_ids]
    
    
            # Get the validation result and response for each sample
            batch_size = test_batch.batch['input_ids'].shape[0]
            for i in range(batch_size):
                sample_data = {
                    'sample_index': sample_index_counter + i,
                    'responses': responses[i*n_val_samples : (i+1)*n_val_samples],
                    'rewards': reward_tensor[i*n_val_samples : (i+1)*n_val_samples].sum(-1).tolist(),
                    'acc': reward_tensor[i*n_val_samples : (i+1)*n_val_samples].sum().item() / n_val_samples,
                    'response_lengths': test_output_gen_batch.batch['attention_mask'][i*n_val_samples : (i+1)*n_val_samples, prompt_length:].sum(1).tolist()
                }
                all_sample_data.append(sample_data)
            
            sample_index_counter += batch_size
    
        print('Validation: Generation and data collection end.')
    
        # Save as a parquet file using pandas and pyarrow
        if all_sample_data:
            import pandas as pd
            df = pd.DataFrame(all_sample_data)
            df.to_parquet('validation_results.parquet')
            print("Validation results saved to validation_results.parquet.")
    
        return None

    def _log_failed_samples(self, batch, solve_none_uids, unique_uids, reward_tensor, n_samples, global_steps):
        """
        Log failed samples (all n attempts failed) to a JSONL file for error analysis.

        Saves for each failed sample:
        - dataset_index: original index in the dataset (for tracing back to source)
        - uid: runtime unique identifier
        - prompt: the raw text prompt (decoded from tokens)
        - responses: list of n sampled responses (decoded text)
        - rewards: list of n rewards
        - total_reward: sum of rewards
        - ground_truth: the correct answer
        - global_step: training step when this was logged
        - timestamp: when this was logged
        """
        error_log_dir = self.config.trainer.get("error_log_dir", None)
        if not error_log_dir or not solve_none_uids:
            return

        # Create directory if not exists
        os.makedirs(error_log_dir, exist_ok=True)
        log_file = os.path.join(error_log_dir, "error_samples.jsonl")

        uids = batch.non_tensor_batch['uid']

        failed_samples = []
        for uid in solve_none_uids:
            # Get all n samples for this uid
            uid_mask = uids == uid
            uid_indices = np.where(uid_mask)[0]

            # Get prompt (same for all n samples, take first)
            prompt_idx = uid_indices[0]
            prompt_tokens = batch.batch['prompts'][prompt_idx]
            # Remove padding tokens
            prompt_attention_mask = batch.batch['attention_mask'][prompt_idx]
            prompt_length = prompt_attention_mask[:batch.batch['prompts'].shape[1]].sum().item()
            prompt_tokens = prompt_tokens[-prompt_length:]  # prompts are left-padded
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            # Get all n responses
            responses_text = []
            responses_rewards = []
            for idx in uid_indices:
                response_tokens = batch.batch['responses'][idx]
                # Get response mask to remove padding
                response_attention_mask = batch.batch['attention_mask'][idx]
                response_length = batch.batch['prompts'].shape[1]
                response_mask = response_attention_mask[response_length:]
                valid_length = response_mask.sum().item()
                response_tokens = response_tokens[:valid_length]
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                responses_text.append(response_text)
                responses_rewards.append(reward_tensor[idx].sum().item())

            # Get ground truth if available
            ground_truth = None
            if 'tgt_input_ids' in batch.batch:
                tgt_tokens = batch.batch['tgt_input_ids'][prompt_idx]
                # Remove padding
                tgt_valid_mask = tgt_tokens != self.tokenizer.pad_token_id
                tgt_valid_tokens = tgt_tokens[tgt_valid_mask]
                if len(tgt_valid_tokens) > 0:
                    ground_truth = self.tokenizer.decode(tgt_valid_tokens, skip_special_tokens=True)

            # Get original dataset index if available
            dataset_index = None
            if 'sample_id' in batch.non_tensor_batch:
                dataset_index = batch.non_tensor_batch['sample_id'][prompt_idx]
                # Convert numpy type to Python int if needed
                if hasattr(dataset_index, 'item'):
                    dataset_index = dataset_index.item()

            # Get extra_info if available (may contain additional metadata)
            extra_info = None
            if 'extra_info' in batch.non_tensor_batch:
                extra_info = batch.non_tensor_batch['extra_info'][prompt_idx]

            sample_data = {
                'dataset_index': dataset_index,  # Original index in dataset (train.parquet)
                'uid': uid,  # Runtime unique identifier
                'global_step': global_steps,
                'prompt': prompt_text,
                'responses': responses_text,
                'rewards': responses_rewards,
                'total_reward': sum(responses_rewards),
                'ground_truth': ground_truth,
                'extra_info': extra_info,
                'timestamp': datetime.now().isoformat()
            }
            failed_samples.append(sample_data)

        # Append to JSONL file
        with open(log_file, 'a', encoding='utf-8') as f:
            for sample in failed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"[Error Log] Logged {len(failed_samples)} failed samples to {log_file}")

    def replace_response_in_batch(self, batch):
        """
            Replace generated responses with ground truth tgt_input_ids in batch.
        """

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        batch = batch.batch
        prompts = batch['prompts'] # b x p_l
        tgt_input_ids = batch['tgt_input_ids'].clone() # b x r_l
        
        # === DEBUG ===
        print(f"\n[DEBUG replace_response_in_batch]")
        print(f"  prompts: {prompts.shape}")
        print(f"  tgt_input_ids: {tgt_input_ids.shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  position_ids: {batch['position_ids'].shape}")
        assert batch['attention_mask'].size(1) >= prompts.size(1), \
            f"attention_mask({batch['attention_mask'].shape}) < prompts({prompts.shape})"
        tgt_lengths = (tgt_input_ids != pad_token_id).sum(dim=1)
        empty_count = (tgt_lengths == 0).sum().item()
        if empty_count > 0:
            print(f"  [WARNING] {empty_count}/{len(tgt_lengths)} samples have empty targets!")
        # === END DEBUG ===
        
        # add eos token
        seq_len = tgt_input_ids.shape[1]
        tgt_lengths = (tgt_input_ids != pad_token_id).sum(dim=1)
        replace_positions = torch.where(tgt_lengths < seq_len, tgt_lengths, seq_len - 1)
        tgt_input_ids[torch.arange(len(tgt_input_ids)), replace_positions] = eos_token_id

        original_attention_mask = batch['attention_mask'][:, :prompts.size(1)] # b x p_l
        original_position_ids = batch['position_ids'][:, :prompts.size(1)] # b x p_l
        
        device = prompts.device
        batch_size = prompts.size(0)
        response_length = tgt_input_ids.size(1) # r_l
        
        # replace responses
        batch['responses'] = tgt_input_ids
        
        # replace input_ids (prompt + tgt)
        batch['input_ids'] = torch.cat([prompts, tgt_input_ids], dim=-1)
        
        # find eos_token of tgt_input_ids, and mask the tokens before eos_token and eos_token
        response_attention_mask = get_response_mask(tgt_input_ids, eos_token_id, 
                                                dtype=original_attention_mask.dtype)
        
        batch['attention_mask'] = torch.cat(
            [original_attention_mask, response_attention_mask], dim=-1)
        
        delta_pos = torch.arange(1, response_length+1, device=device)\
                    .expand(batch_size, response_length)
        last_prompt_pos = original_position_ids[:, -1].unsqueeze(-1)
        response_position_ids = last_prompt_pos + delta_pos
        
        batch['position_ids'] = torch.cat(
            [original_position_ids, response_position_ids], dim=-1)

        return

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            if self.config.trainer.get("val_before_train_with_save", False):
                val_metrics = self._validate_with_save()
            else:
                val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        n_samples = self.config.actor_rollout_ref.rollout.n
        sft_data_size = self.config.actor_rollout_ref.actor.sft.sft_data_size
        sft_buffer_batch = None

        coalsql_skip_steps = self.config.trainer.get("coalsql_skip_steps", 1)

        # ═══ Initialize retrieval augmentation (lightweight, no embedding model) ═══
        retriever = None
        retrieval_cfg = self.config.trainer.get("retrieval_augmentation", None)
        if retrieval_cfg and retrieval_cfg.get("enable", False):
            from .epoch_retrieval import EpochRetriever
            retriever = EpochRetriever(
                faiss_index_path=retrieval_cfg.faiss_index_path,
                metadata_path=retrieval_cfg.metadata_path,
                question_bank_parquet=retrieval_cfg.question_bank_parquet,
                train_embeddings_path=retrieval_cfg.train_embeddings_path,
                train_idx_mapping_path=retrieval_cfg.train_idx_mapping_path,
                qb_embeddings_path=retrieval_cfg.qb_embeddings_path,
                top_k=retrieval_cfg.get("top_k", 5),
                num_per_epoch=retrieval_cfg.get("num_per_epoch", 2000),
                rank_weights=list(retrieval_cfg.get("rank_weights", [0.40, 0.25, 0.20, 0.10, 0.05])),
                seed=retrieval_cfg.get("seed", 42),
                random_ratio=retrieval_cfg.get("random_ratio", 0.0),
                error_aware_config=retrieval_cfg.get("error_aware", None),
            )
            print(f"[Retrieval] EpochRetriever initialized. top_k={retriever.top_k}, "
                  f"num_per_epoch={retriever.num_per_epoch}, random_ratio={retriever.random_ratio}, "
                  f"error_aware={'enabled' if retriever.error_aware_retriever else 'disabled'}")

        for epoch in range(self.config.trainer.total_epochs):
            # Collect failed samples across all batches in this epoch for retrieval
            epoch_failed_samples = []  # list of (split, index)
            epoch_failed_sample_details = []  # list of dicts with context info for error-aware

            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    uids = batch.non_tensor_batch['uid']
                    seen = set()
                    unique_uids = [uid for uid in uids if uid not in seen and not seen.add(uid)]

                    fail_threshold = 4
                    solve_value = 4

                    solve_none = 0
                    solve_all = 0

                    solve_none_uids = []
                    uid2solve_num = {}
                    for uid in unique_uids:
                        uid_mask = uids == uid
                        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                        
                        # Check if all rewards are <4 (solve_none) or all are ==4 (solve_all) for this uid
                        if (uid_rewards < fail_threshold).all():
                            solve_none_uids.append(uid)
                            solve_none += 1
                        elif (uid_rewards == solve_value).all():
                            solve_all += 1

                        uid2solve_num[uid] = uid_rewards.sum().item()

                    # Log to metrics
                    metrics['batch/solve_none'] = solve_none
                    metrics['batch/solve_all'] = solve_all

                    metrics['batch/solved'] = (reward_tensor.sum(-1) == solve_value).sum().item() / len(uids)
                    metrics['batch/failed'] = (reward_tensor.sum(-1) < fail_threshold).sum().item() / len(uids)

                    # ═══ Collect failed samples for epoch-level retrieval ═══
                    if retriever is not None and solve_none_uids:
                        for uid in solve_none_uids:
                            uid_mask = uids == uid
                            first_idx = np.where(uid_mask)[0][0]
                            extra = batch.non_tensor_batch.get('extra_info', None)
                            if extra is not None:
                                sample_extra = extra[first_idx]
                                if isinstance(sample_extra, dict):
                                    split = sample_extra.get('split', 'train')
                                    index = sample_extra.get('index', None)
                                    if index is not None:
                                        epoch_failed_samples.append((split, index))

                                        # Collect error context for error-aware retrieval
                                        if retriever.error_aware_retriever is not None:
                                            # Get prompt text (context)
                                            prompt_tokens = batch.batch['prompts'][first_idx]
                                            prompt_attention_mask = batch.batch['attention_mask'][first_idx]
                                            prompt_length = prompt_attention_mask[:batch.batch['prompts'].shape[1]].sum().item()
                                            prompt_tokens_valid = prompt_tokens[-int(prompt_length):]
                                            context = self.tokenizer.decode(prompt_tokens_valid, skip_special_tokens=True)

                                            # Get correct SQL from tgt_input_ids or extra_info
                                            correct_sql = ""
                                            if 'tgt_input_ids' in batch.batch:
                                                tgt_tokens = batch.batch['tgt_input_ids'][first_idx]
                                                tgt_valid_mask = tgt_tokens != self.tokenizer.pad_token_id
                                                tgt_valid_tokens = tgt_tokens[tgt_valid_mask]
                                                if len(tgt_valid_tokens) > 0:
                                                    correct_sql = self.tokenizer.decode(tgt_valid_tokens, skip_special_tokens=True)

                                            # Get first error response
                                            error_response = ""
                                            response_tokens = batch.batch['responses'][first_idx]
                                            response_mask = batch.batch['attention_mask'][first_idx]
                                            resp_start = batch.batch['prompts'].shape[1]
                                            resp_mask = response_mask[resp_start:]
                                            valid_len = resp_mask.sum().item()
                                            if valid_len > 0:
                                                error_response = self.tokenizer.decode(
                                                    response_tokens[:int(valid_len)],
                                                    skip_special_tokens=True
                                                )

                                            epoch_failed_sample_details.append({
                                                'split': split,
                                                'index': index,
                                                'context': context,
                                                'correct_sql': correct_sql,
                                                'error_response': error_response,
                                            })

                    # Log failed samples for error analysis
                    if self.config.trainer.get("error_log_dir", None) and solve_none_uids:
                        self._log_failed_samples(
                            batch=batch,
                            solve_none_uids=solve_none_uids,
                            unique_uids=unique_uids,
                            reward_tensor=reward_tensor,
                            n_samples=n_samples,
                            global_steps=self.global_steps
                        )

                    # how to buffer samples for subsequent SFT
                    sft_buffer_uids = solve_none_uids
                    
                    # create buffer batch
                    buffer_indexes = []
                    uids = batch.non_tensor_batch['uid']
                    for i, uid in enumerate(unique_uids):
                        if uid in sft_buffer_uids:
                            indices = np.where(uids == uid)[0]
                            indice = indices[0]
                            buffer_indexes.append(indice)

                    # update sft_buffer_batch
                    if sft_data_size != -1 and buffer_indexes and self.global_steps >= coalsql_skip_steps:
                        buffer_batch = batch.select_idxs(buffer_indexes)
                        
                        if sft_buffer_batch is not None:
                            sft_buffer_batch = DataProto.concat([sft_buffer_batch, buffer_batch])
                        else:
                            sft_buffer_batch = buffer_batch

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # SFT update using hard-batch
                    if sft_data_size != -1 and self.global_steps >= coalsql_skip_steps and sft_buffer_batch is not None and len(sft_buffer_batch) >= sft_data_size:
                        with _timer('sft_update_actor', timing_raw):
                            print("SFT")
                            
                            sft_buffer_batch.to('cpu')
                            sft_buffer_batch.batch.to('cpu')
                            
                            sft_train_batch = sft_buffer_batch.slice(0,sft_data_size)
                            
                            # replace on-policy with off-policy
                            self.replace_response_in_batch(sft_train_batch)
                            
                            if len(sft_buffer_batch) == sft_data_size:
                                sft_buffer_batch = None
                            else:
                                sft_buffer_batch = sft_buffer_batch.slice(sft_data_size, len(sft_buffer_batch))

                            self._balance_batch(sft_train_batch, metrics=metrics)
                            sft_output = self.actor_rollout_wg.sft_update_actor(sft_train_batch)
                            sft_output_metrics = reduce_metrics(sft_output.meta_info['metrics'])
                            metrics.update(sft_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

            # ═══ End of epoch: Retrieval augmentation ═══════════════════════
            # After processing all batches in this epoch, retrieve similar
            # questions from the question bank for failed samples and either
            # augment the dataset for next epoch RL (extra_rl) or directly
            # perform SFT on retrieved samples (extra_sft).
            if retriever is not None and epoch < self.config.trainer.total_epochs - 1:
                if epoch_failed_samples:
                    print(f"\n[Retrieval] Epoch {epoch}: {len(epoch_failed_samples)} failed samples collected. "
                          f"Retrieving new training data...")
                    new_samples_df = retriever.retrieve_for_failed(
                        failed_samples=epoch_failed_samples,
                        failed_sample_details=epoch_failed_sample_details if epoch_failed_sample_details else None,
                    )
                    if len(new_samples_df) > 0:
                        retrieval_mode = self.config.trainer.retrieval_augmentation.get("mode", "extra_rl")
                        
                        if retrieval_mode == "extra_sft":
                            # ── Extra SFT: directly SFT on retrieved samples ──
                            self._run_extra_sft(new_samples_df, epoch, logger)
                        else:
                            # ── Extra RL: augment dataset for next epoch ──
                            self._augment_dataset_and_rebuild(new_samples_df)
                            # Update progress bar total since steps per epoch changed
                            new_steps_per_epoch = len(self.train_dataloader)
                            remaining_epochs = self.config.trainer.total_epochs - epoch - 1
                            new_total = self.global_steps - 1 + new_steps_per_epoch * remaining_epochs
                            self.total_training_steps = max(self.total_training_steps, new_total)
                            progress_bar.total = self.total_training_steps
                            progress_bar.refresh()
                            print(f"[Retrieval] Epoch {epoch}: added {len(new_samples_df)} samples. "
                                  f"Dataset: {len(self.train_dataset)} total. "
                                  f"Steps/epoch: {new_steps_per_epoch}. "
                                  f"Estimated total steps: {self.total_training_steps}")

                    # Log retrieval stats
                    retrieval_stats = retriever.get_stats()
                    logger.log(data=retrieval_stats, step=self.global_steps - 1)
                else:
                    print(f"\n[Retrieval] Epoch {epoch}: no failed samples, skipping retrieval.")
                
    def _run_extra_sft(self, new_samples_df: 'pd.DataFrame', epoch: int, logger):
        """
        Run extra SFT on retrieved samples at the end of an epoch.

        Instead of augmenting the dataset for future RL (like _augment_dataset_and_rebuild),
        this method directly performs SFT using the ground truth (tgt_input_ids) from
        the retrieved question bank samples.

        Flow:
        1. (Optional) Run validation before extra SFT
        2. Construct SFT data: tokenize retrieved samples via RLHFDatasetWithTarget,
           then use replace_response_in_batch to build the SFT batch
        3. Execute SFT via sft_update_actor
        4. (Optional) Run validation after extra SFT

        Args:
            new_samples_df: DataFrame from question bank with retrieved samples (has prompt & target columns)
            epoch: Current epoch number
            logger: Tracking logger for metrics
        """
        import tempfile

        extra_sft_cfg = self.config.trainer.retrieval_augmentation.get("extra_sft", {})
        eval_before = extra_sft_cfg.get("eval_before", False)
        eval_after = extra_sft_cfg.get("eval_after", True)
        extra_sft_batch_size = extra_sft_cfg.get("batch_size", 64)
        extra_sft_epochs = extra_sft_cfg.get("epochs", 1)  # outer epoch: how many times to iterate over the whole DataLoader

        print(f"\n[Extra SFT] Epoch {epoch}: Starting extra SFT with {len(new_samples_df)} retrieved samples")

        # ═══ Step 1: Optional pre-SFT evaluation ═══════════════════════
        if eval_before and self.val_reward_fn is not None:
            print(f"[Extra SFT] Epoch {epoch}: Running pre-SFT evaluation...")
            pre_val_metrics = self._validate()
            # Add prefix to distinguish from regular validation
            prefixed_metrics = {f"extra_sft_pre/{k}": v for k, v in pre_val_metrics.items()}
            if 'avg_score' not in pre_val_metrics:
                pre_val_metrics['avg_score'] = np.mean([v for k, v in pre_val_metrics.items() if k.startswith('val/test_score/')])
            prefixed_metrics['extra_sft_pre/avg_score'] = pre_val_metrics['avg_score']
            logger.log(data=prefixed_metrics, step=self.global_steps - 1)
            self.maybe_save_best_hf(pre_val_metrics)
            print(f"[Extra SFT] Epoch {epoch}: Pre-SFT avg_score = {pre_val_metrics['avg_score']:.4f}")

        # ═══ Step 2: Construct SFT data from retrieved samples ══════════
        # Write to temp parquet for RLHFDatasetWithTarget to load
        tmp_dir = self.config.trainer.get("augmented_data_dir", None)
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"extra_sft_epoch_{epoch}.parquet")
        else:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
            os.close(tmp_fd)

        new_samples_df.to_parquet(tmp_path, index=False)

        # Create a temporary dataset to tokenize retrieved samples
        from .rl_dataset_with_target import RLHFDatasetWithTarget, collate_fn as target_collate_fn

        extra_dataset = RLHFDatasetWithTarget(
            parquet_files=tmp_path,
            tokenizer=self.tokenizer,
            config=self.config.data,
        )

        print(f"[Extra SFT] Epoch {epoch}: Loaded {len(extra_dataset)} samples for extra SFT")

        # Create DataLoader for the extra SFT data
        from torch.utils.data import DataLoader

        extra_dataloader = DataLoader(
            dataset=extra_dataset,
            batch_size=extra_sft_batch_size,
            num_workers=0,  # Use 0 workers for simplicity in temp dataset
            shuffle=True,
            drop_last=True,
            collate_fn=target_collate_fn,
        )

        # ═══ Step 3: Execute SFT on each batch ═══════════════════════════
        # Read extra_sft_epochs for the internal per-batch epoch (passed via meta_info)
        extra_sft_internal_epochs = extra_sft_cfg.get("internal_epochs", None)  # None = use default sft_epochs

        extra_sft_metrics_all = {}
        n_batches = 0

        for ext_ep in range(extra_sft_epochs):
            for batch_dict in extra_dataloader:
                sft_batch = DataProto.from_single_dict(batch_dict)

                # The batch from RLHFDatasetWithTarget has:
                #   input_ids (prompt tokens), attention_mask, position_ids, tgt_input_ids
                # We need to set input_ids as "prompts" so replace_response_in_batch works
                sft_batch.batch['prompts'] = sft_batch.batch['input_ids'].clone()

                # Replace response with ground truth (tgt_input_ids)
                # This constructs: input_ids = [prompts, tgt], attention_mask, position_ids
                self.replace_response_in_batch(sft_batch)

                # Balance batch across DP ranks
                extra_metrics = {}
                self._balance_batch(sft_batch, metrics=extra_metrics)

                # Set required meta_info for extra_sft_update_actor
                sft_batch.meta_info["global_token_num"] = torch.sum(sft_batch.batch["attention_mask"], dim=-1).tolist()
                sft_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                if extra_sft_internal_epochs is not None:
                    sft_batch.meta_info["extra_sft_epochs"] = extra_sft_internal_epochs

                # Execute SFT update using extra_sft_update_actor (separate optimizer)
                sft_output = self.actor_rollout_wg.extra_sft_update_actor(sft_batch)
                sft_output_metrics = reduce_metrics(sft_output.meta_info['metrics'])

                # Accumulate metrics
                for k, v in sft_output_metrics.items():
                    key = f"extra_sft/{k}"
                    if key not in extra_sft_metrics_all:
                        extra_sft_metrics_all[key] = []
                    extra_sft_metrics_all[key].append(v)
                
                n_batches += 1

        # Average metrics across batches
        avg_extra_sft_metrics = {k: np.mean(v) for k, v in extra_sft_metrics_all.items()}
        avg_extra_sft_metrics["extra_sft/n_samples"] = len(new_samples_df)
        avg_extra_sft_metrics["extra_sft/n_batches"] = n_batches
        avg_extra_sft_metrics["extra_sft/n_outer_epochs"] = extra_sft_epochs
        logger.log(data=avg_extra_sft_metrics, step=self.global_steps - 1)

        print(f"[Extra SFT] Epoch {epoch}: Completed extra SFT. {n_batches} batches processed over {extra_sft_epochs} outer epochs.")

        # ═══ Step 4: Optional post-SFT evaluation ══════════════════════
        if eval_after and self.val_reward_fn is not None:
            print(f"[Extra SFT] Epoch {epoch}: Running post-SFT evaluation...")
            post_val_metrics = self._validate()
            # Add prefix to distinguish from regular validation
            prefixed_metrics = {f"extra_sft_post/{k}": v for k, v in post_val_metrics.items()}
            if 'avg_score' not in post_val_metrics:
                post_val_metrics['avg_score'] = np.mean([v for k, v in post_val_metrics.items() if k.startswith('val/test_score/')])
            prefixed_metrics['extra_sft_post/avg_score'] = post_val_metrics['avg_score']
            logger.log(data=prefixed_metrics, step=self.global_steps - 1)
            self.maybe_save_best_hf(post_val_metrics)
            print(f"[Extra SFT] Epoch {epoch}: Post-SFT avg_score = {post_val_metrics['avg_score']:.4f}")

        # ═══ Step 5: Clean up temporary file ═══════════════════════════
        if not self.config.trainer.get("augmented_data_dir", None):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        print(f"[Extra SFT] Epoch {epoch}: Extra SFT complete.\n")

    def _augment_dataset_and_rebuild(self, new_samples_df: 'pd.DataFrame'):
        """
        Append new samples (from retrieval) to the training dataset and
        rebuild the dataloader for the next epoch.

        Strategy:
        1. Convert the new DataFrame to a temporary parquet file.
        2. Load it as a HuggingFace Dataset.
        3. Concatenate with the existing dataset.
        4. Rebuild the sampler and dataloader.

        This approach is minimally invasive: we reuse the existing
        RLHFDatasetWithTarget by directly manipulating its underlying
        HuggingFace Dataset object (self.train_dataset.dataframe).
        """
        import tempfile
        import datasets

        print(f"[Retrieval] Augmenting dataset: current={len(self.train_dataset)}, "
              f"adding={len(new_samples_df)} samples")

        # ── Step 1: Write new samples to a temporary parquet file ───────
        tmp_dir = self.config.trainer.get("augmented_data_dir", None)
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"augmented_epoch_{self.global_steps}.parquet")
        else:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
            os.close(tmp_fd)

        new_samples_df.to_parquet(tmp_path, index=False)

        # ── Step 2: Load as HuggingFace Dataset ────────────────────────
        new_hf_dataset = datasets.load_dataset("parquet", data_files=tmp_path)["train"]

        # Add sample_id for the new samples (continuing from the existing max)
        existing_max_id = len(self.train_dataset) - 1
        new_hf_dataset = new_hf_dataset.map(
            lambda example, idx: {"sample_id": existing_max_id + 1 + idx},
            with_indices=True
        )

        # ── Step 3: Concatenate with existing dataset ──────────────────
        self.train_dataset.dataframe = datasets.concatenate_datasets([
            self.train_dataset.dataframe,
            new_hf_dataset,
        ])

        print(f"[Retrieval] Dataset augmented: {len(self.train_dataset)} total samples")

        # ── Step 4: Rebuild sampler and dataloader ─────────────────────
        from verl.trainer.main_ppo import create_rl_sampler
        from .rl_dataset_with_target import collate_fn as target_collate_fn

        new_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=target_collate_fn,
            sampler=new_sampler,
        )

        print(f"[Retrieval] Dataloader rebuilt: {len(self.train_dataloader)} batches/epoch")

        # ── Step 5: Clean up temporary file (if not saved for debugging) ─
        if not self.config.trainer.get("augmented_data_dir", None):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
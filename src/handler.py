#!/usr/bin/env python
""" Contains the handler function that will be called by the serverless worker. """

# Start the vLLM serving layer on our RunPod worker.
from typing import Generator, List, Optional, Tuple, Union
import json


from transformers import (
    ExponentialDecayLengthPenalty,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    ExponentialDecayLengthPenalty,
    LogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
)
from metrics import vllm_log_system_stats
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.utils import random_uuid
import runpod
import os
import torch


class ProcessorCompatabilityLayer:
    def __init__(self, logits_processor: LogitsProcessor):
        self.logits_processor = logits_processor

    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        input_ids_tensor = torch.LongTensor(input_ids).to("cuda").view(1, -1)
        scores = scores.view(1, -1)
        return self.logits_processor(input_ids_tensor, scores).view(-1)

# Prepare the model and tokenizer
MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME is not defined")
USE_FULL_METRICS = os.environ.get("USE_FULL_METRICS", True)


# Tensor parallelism
try:
    NUM_GPU_SHARD = int(os.environ.get("NUM_GPU_SHARD", 1))
except ValueError:
    print("Error: NUM_GPU_SHARD should be an integer. Using default value of 1.")
    NUM_GPU_SHARD = 1

# Prepare the engine's arguments
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    quantization="awq",
    tokenizer_mode="auto",
    tensor_parallel_size=NUM_GPU_SHARD,
    dtype="half",
    seed=0,
    max_num_batched_tokens=4096,
    disable_log_stats=False,
    # max_num_seqs=256,
)

# Create the vLLM asynchronous engine
llm = AsyncLLMEngine.from_engine_args(engine_args)
print("vLLM engine created successfully.")

# Collect banned token id's
tokenizer = llm.engine.tokenizer

vocab = tokenizer.get_vocab()
banned_token_ids = []
for id in vocab.values():
    token = tokenizer.decode(id)
    for char in [
        "(",
        ")",
        "*",
        '"',
        "[",
        "]",
        ":",
        ";",
        "\n",
        "\t",
        "\v",
        "\f",
        "\\",
        "\r",
    ]:
        if char in token:
            banned_token_ids.append([id])
            break


def get_banned_emojis_list():
    with open(os.path.join(os.path.dirname(__file__), "emojis.json"), "r") as f:
        emojis = json.load(f)["emojis"]

    banned_first_emoji_tokens_set = set()
    for emoji in emojis:
        tokens = tokenizer.encode(emoji)
        if tokens[0] == 1:
            tokens = tokens[1:]
        else:
            print(tokens)

        if tokens[0] == 29871:
            tokens = tokens[1:]
        else:
            print(tokens)

        decode = tokenizer.decode(tokens[0])
        if decode.isascii():
            continue
        banned_first_emoji_tokens_set.add(tokens[0])

    print("banned emoji tokens: ", banned_first_emoji_tokens_set)
    banned_first_emoji_tokens_list = []
    for tok in banned_first_emoji_tokens_set:
        decode = tokenizer.decode(tok)
        if decode.isascii():
            continue
        banned_first_emoji_tokens_list.append([tok])
        print(tok, decode, decode.encode("unicode_escape"))
    print(banned_first_emoji_tokens_list)
    return banned_first_emoji_tokens_list


banned_first_emoji_tokens_list = get_banned_emojis_list()


# Incorporate metrics tracking
llm.engine._log_system_stats = lambda x, y: vllm_log_system_stats(llm.engine, x, y)


def concurrency_controller() -> bool:
    # Calculate pending sequences
    total_pending_sequences = len(llm.engine.scheduler.waiting) + len(
        llm.engine.scheduler.swapped
    )
    print("Total pending sequences in vLLM queue: {}".format(total_pending_sequences))

    # Enable auto-scaling if pending sequences exist
    return total_pending_sequences > 30


def prepare_metrics() -> dict:
    # The vLLM metrics are updated every 5 seconds, see metrics.py for the _LOGGING_INTERVAL_SEC field.
    if hasattr(llm.engine, "metrics"):
        return llm.engine.metrics
    else:
        return {}


# Validation
def get_sampling_params(sampling_params, input_ids_length):
    if not sampling_params:
        return SamplingParams()

    def validate_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def validate_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def validate_bool(value, default):
        if isinstance(value, bool):
            return value
        return default

    n = validate_int(sampling_params.get("n"), None)
    best_of = validate_int(sampling_params.get("best_of"), None)
    presence_penalty = validate_float(sampling_params.get("presence_penalty"), None)
    frequency_penalty = validate_float(sampling_params.get("frequency_penalty"), None)
    temperature = validate_float(sampling_params.get("temperature"), None)
    top_p = validate_float(sampling_params.get("top_p"), None)
    top_k = validate_int(sampling_params.get("top_k"), None)
    use_beam_search = validate_bool(sampling_params.get("use_beam_search"), None)
    stop = sampling_params.get("stop", None)
    ignore_eos = validate_bool(sampling_params.get("ignore_eos"), None)
    max_tokens = validate_int(sampling_params.get("max_tokens"), 256)
    logprobs = validate_float(sampling_params.get("logprobs"), None)
    no_repeat_ngram_size = validate_int(
        sampling_params.get("no_repeat_ngram_size"), None
    )
    ban_non_speakable_tokens = validate_bool(
        sampling_params.get("ban_non_speakable_tokens"), False
    )
    min_length = validate_int(sampling_params.get("min_length"), None)
    repetition_penalty = validate_float(sampling_params.get("repetition_penalty"), None)
    exponential_decay_length_penalty = sampling_params.get(
        "exponential_decay_length_penalty"
    )
    logits_processors = LogitsProcessorList()
    ban_emojis = validate_bool(sampling_params.get("ban_emojis"), False)

    if no_repeat_ngram_size is not None:
        logits_processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if ban_non_speakable_tokens:
        logits_processors.append(
            NoBadWordsLogitsProcessor(banned_token_ids, tokenizer.eos_token_id)
        )
    if ban_emojis:
        logits_processors.append(
            NoBadWordsLogitsProcessor(
                banned_first_emoji_tokens_list, tokenizer.eos_token_id
            )
        )
    if repetition_penalty is not None:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if exponential_decay_length_penalty is not None:
        start_index = int(exponential_decay_length_penalty.split(",")[0])
        decay_factor = float(exponential_decay_length_penalty.split(",")[1])
        logits_processors.append(
            ExponentialDecayLengthPenalty(
                (start_index, decay_factor), tokenizer.eos_token_id, input_ids_length
            )
        )

    if min_length is not None:
        logits_processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_length, min_length, tokenizer.eos_token_id
            )
        )

    logits_processors = [ProcessorCompatabilityLayer(p) for p in logits_processors]

    params = {
        "n": n,
        "best_of": best_of,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "use_beam_search": use_beam_search,
        "stop": stop,
        "ignore_eos": ignore_eos,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "logits_processors": logits_processors,
    }

    # Remove None values
    params = {k: v for k, v in params.items() if v is not None}
    return SamplingParams(**params)


# def validate_and_set_sampling_params(sampling_params):
#     """
#     Validates the given sampling parameters and creates a SamplingParams object.
#     If no sampling parameters are provided, defaults are used.
#     """
#     if sampling_params:
#         validated_params = validate_sampling_params(sampling_params)
#         # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L7
#         return SamplingParams(**validated_params)
#     return SamplingParams()


async def handler(job: dict) -> dict[str, list]:
    """
    This is the handler function that will be called by the serverless worker.
    """
    print("Job received by handler: {}".format(job))

    # Retrieve the job input.
    job_input = job["input"]

    # Create the prompt using the template.
    prompt = job_input["prompt"]

    prompt_token_ids = tokenizer.encode(prompt)

    # Validate and set sampling parameters
    sampling_params = get_sampling_params(
        job_input.get("sampling_params", None), len(prompt_token_ids)
    )

    # Print job input and sampling parameters
    print("Job Input:", job_input)
    print("Sampling Parameters:", sampling_params)

    # Send request to VLLM
    request_id = random_uuid()
    results_generator = llm.generate(
        None, sampling_params, request_id, prompt_token_ids=prompt_token_ids
    )

    # Get the final generated output
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    # Extract prompt and text outputs
    prompt = final_output.prompt
    text_outputs = [output.text for output in final_output.outputs]
    token_ids_outputs = [output.token_ids for output in final_output.outputs]

    for token_ids_output in token_ids_outputs:
        print("Token IDs:")
        for token_id in token_ids_output:
            print(token_id, ":", tokenizer.decode(token_id))

    # Number of generated sequences
    num_seqs = sampling_params.n

    # Prepare metrics if full metrics are enabled
    runpod_metrics = prepare_metrics() if USE_FULL_METRICS else {}

    # Record job input and token counts
    # runpod_metrics['job_input'] = job_input

    runpod_metrics["input_tokens"] = len(final_output.prompt_token_ids) * num_seqs
    runpod_metrics["output_tokens"] = sum(
        [len(output.token_ids) for output in final_output.outputs]
    )

    # Store the scenario type
    runpod_metrics["scenario"] = "batch"

    # Include metrics for the job.
    runpod.serverless.modules.rp_metrics.metrics_collector.push_metrics_internal(
        job_id=job["id"], metrics=runpod_metrics
    )

    ret = {
        "text": text_outputs,
        "input_tokens": runpod_metrics["input_tokens"],
        "output_tokens": runpod_metrics["output_tokens"],
    }
    return ret


# Start the serverless worker with appropriate settings
# if STREAMING:
#     print("Starting the vLLM serverless worker with streaming enabled.")
#     runpod.serverless.start({
#         "handler": handler_streaming,
#         "concurrency_controller": concurrency_controller,
#         "return_aggregate_stream": True
#     })
# else:
print("Starting the vLLM serverless worker with streaming disabled.")
runpod.serverless.start(
    {"handler": handler, "concurrency_controller": concurrency_controller}
)

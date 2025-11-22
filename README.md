# `comparator`

LLM head-to-head evaluations using a range of different datasets and scorers.
Can be used to compare performance between two different models, or the same 
model at different checkpoints to validate training.

### Prerequisites

This program downloads parquet datasets from HuggingFace, using Apache Arrow.
Install it on your machine [here.](https://arrow.apache.org/install/)

Afterwards, use `cmake` to build the program:

```bash
cmake --build <build-dir> --target comparator -j 8
```


### Usage example
```text
[comparator] INFO - Running scoring task: dataset=MRPC, model=babbage-002, endpoint=https://api.openai.com/v1/completions, config=default, split=train, scoring metric: F1, workers=5
[comparator] INFO - Running scoring task: dataset=MRPC, model=gpt-3.5-turbo-instruct, endpoint=https://api.openai.com/v1/completions, config=default, split=train, scoring metric: F1, workers=5
[comparator] INFO - model gpt-3.5-turbo-instruct got F1 score on dataset MRPC: 0.695652
[comparator] INFO - model babbage-002 got F1 score on dataset MRPC: 0.484848
```

### CLI args
```text
./comparator help
comparator [evaluate/compare] [ARGS]
base args (applies to both 'compare' and 'evaluate'):
--max-tokens -> max tokens for model responses
--num-logprobs -> num logprobs to return from model responses
--workers -> number of workers to use per dataset scoring
--scorer -> metric to score with e.g. accuracy, f1, etc
comparator evaluate args:
--model -> model id for inference
--endpoint -> endpoint uri
--dataset -> hf dataset id to use, e.g. SetFit/mrpc
--config -> hf dataset id config to use (e.g. default, cola, etc)
--split -> hf dataset split to use (e.g. train, test, etc)
comparator compare args:
--model_a -> model id for model A
--endpoint_a -> endpoint uri for model A
--model_b -> model id for model B
--endpoint_b -> endpoint uri for model B
--dataset -> hf dataset id to use, e.g. SetFit/mrpc
--config -> hf dataset id config to use (e.g. default, cola, etc)
--split -> hf dataset split to use (e.g. train, test, etc)
```
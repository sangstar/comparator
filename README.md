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
./comparator compare --model_a gpt-3.5-turbo-instruct --model_b babbage-002 --dataset MRPC --endpoint_a https://api.openai.com/v1/completions --endpoint_b https://api.openai.com/v1/completions --scorer f1
[comparator] INFO - Running scoring task: dataset=MRPC, model=gpt-3.5-turbo-instruct, endpoint=https://api.openai.com/v1/completions, config=default, split=train, scoring metric: F1, workers=5
[comparator] INFO - Running scoring task: dataset=MRPC, model=babbage-002, endpoint=https://api.openai.com/v1/completions, config=default, split=train, scoring metric: F1, workers=5
[comparator] INFO - REPORT RESULT: model gpt-3.5-turbo-instruct for endpoint https://api.openai.com/v1/completions got F1 score on dataset MRPC: 0.727273
[comparator] INFO - REPORT RESULT: model babbage-002 for endpoint https://api.openai.com/v1/completions got F1 score on dataset MRPC: 0.387097
```
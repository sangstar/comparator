//
// Created by Sanger Steel on 11/19/25.
//

#include "scorer.h"

BatchedResult ScoreResult_vector_to_BatchedResult(const std::vector<ScoreResult>& scores) {
    size_t correct = 0;
    auto rows = scores.size();
    for (const auto& score: scores) {
        correct += score.passed;
    }
    return BatchedResult{rows, correct, (float) correct / (float) rows};
}

BatchedResult score_batch(const char* url, const char* model, const Dataset& dataset, ParquetBatch& batch) {
    std::vector<ScoreResult> scores;
    for (int64_t i = 0; i < batch.num_rows; ++i) {
        auto row = batch.get_row(i);
        std::string prompt = dataset.prompt_creation_fn(&row);

        int max_tokens = 10;
        std::string resp = send_openai_query(url, model, prompt.c_str(), max_tokens, 5);
        if (!is_valid_resp(resp)) throw std::runtime_error("invalid resp: " + resp);
        auto compls = get_responses_from_json(resp.c_str());
        scores.emplace_back(
            ScoreResult{prompt, model, dataset.response_scorer_fn(compls, &row)}
        );
    }
    return ScoreResult_vector_to_BatchedResult(scores);
}

void score_dataset(int idx, const char* url, const char* model, const Dataset& dataset, ParquetBatch batch,
    std::vector<BatchedResult>& results) {
    auto scores = score_batch(url, model, dataset, batch);
    {
        std::lock_guard<std::mutex> lock(score_mu);
        results.emplace_back(scores);
    }
}

void RunScoringTask(OverallScore& score, DatasetIds dataset_id, const char* url, const char* model, const char* config,
    const char* split, int num_threads) {
    Dataset mrpc = CreateDataset(dataset_id, config, split);
    std::thread threads[num_threads];
    std::vector<BatchedResult> results;

    // Give each thread an equally sized chunk per batch
    mrpc.data.reader.set_chunksize(mrpc.data.table->num_rows() / num_threads);

    for (int i = 0; i < num_threads; ++i) {
        const auto batch = mrpc.data.get_batch();
        if (batch.has_value()) {
            threads[i] = std::thread([i, url, model, mrpc, batch, &results] {
                score_dataset(i, url, model, mrpc, batch.value(), results);
            });
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (threads[i].joinable()) threads[i].join();
    }
    fprintf(stdout, "finished\n");
    size_t corrects = 0;
    size_t rows_scored = 0;
    for (const auto& result: results) {
        corrects += result.num_correct;
        rows_scored += result.num_rows;
    }
    score.model = model;
    score.url = url;
    score.dataset_id = dataset_id;
    score.pct = (float) corrects / (float) rows_scored;
}

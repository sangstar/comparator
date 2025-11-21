//
// Created by Sanger Steel on 11/19/25.
//

#include "scorer.h"

void OverallScore::report() const {
    fprintf(stdout,
        "REPORT RESULT: model %s for endpoint %s got %s score on dataset %s: %f\n",
        model, url, score_str.c_str(), DatasetIdStrs[(size_t)dataset_id],  score
        );
}

ScoreStrategies scorer_strategy_from_str(const char* str) {
    if (strcmp(str, "accuracy") == 0) return ScoreStrategies::ACCURACY;
    if (strcmp(str, "f1") == 0) return ScoreStrategies::F1;
    throw std::runtime_error("invalid str for score strategy");
}

BatchedResult ScoreResult_vector_to_BatchedResult(const std::vector<QAResponse>& scores) {
    size_t correct = 0, tps = 0, fps = 0, tns = 0, fns = 0;
    // TODO: This ignores the yes/no logprobs and stuff
    auto rows = scores.size();
    for (const auto& score: scores) {
        correct += score.passed;
        tps += score.tp;
        fps += score.fp;
        tns += score.tn;
        fns += score.fn;
    }
    return BatchedResult{rows, correct, tps, fps, tns, fns, (float) correct / (float) rows};
}

BatchedResult score_batch(ScoreConfig& config, const Dataset& dataset, ParquetBatch& batch) {
    std::vector<QAResponse> scores;
    int its = batch.num_rows;
#ifdef DEBUG
    its = 5;
#endif
    for (int64_t i = 0; i < its; ++i) {
        auto row = batch.get_row(i);
        std::string prompt = dataset.prompt_creation_fn(&row);
        std::string resp = send_openai_query(config.endpoint, config.model, prompt.c_str(), config.max_tokens, config.num_logprobs);
        if (!is_valid_resp(resp)) throw std::runtime_error("invalid resp: " + resp);
        auto compls = get_responses_from_json(resp.c_str());
        scores.emplace_back(
            dataset.response_scorer_fn(compls, &row)
        );
    }
    return ScoreResult_vector_to_BatchedResult(scores);
}

void score_dataset(int idx, ScoreConfig config, const Dataset& dataset, ParquetBatch batch,
    std::vector<BatchedResult>& results) {
    auto scores = score_batch(config, dataset, batch);
    {
        std::lock_guard<std::mutex> lock(score_mu);
        results.emplace_back(scores);
    }
}

void RunScoringTask(OverallScore& score, ScoreConfig config, const Scorer& scorer, int num_threads) {
    const char* score_str = ScoreStrategiesToStr[(size_t)scorer.strategy];
    score.score_str = std::string(score_str);

    fprintf(stdout,
        "Running scoring task: dataset=%s, model=%s, endpoint=%s, config=%s, split=%s, scoring metric: %s, workers=%i\n",
        DatasetIdStrs[(size_t)config.dataset_id], config.model, config.endpoint, config.config, config.split, score_str, num_threads);
    Dataset mrpc = CreateDataset(config.dataset_id, config.config, config.split);
    std::thread threads[num_threads];
    std::vector<BatchedResult> results;

    // Give each thread an equally sized chunk per batch
    mrpc.data.reader.set_chunksize(mrpc.data.table->num_rows() / num_threads);

    for (int i = 0; i < num_threads; ++i) {
        const auto batch = mrpc.data.get_batch();
        if (batch.has_value()) {
            threads[i] = std::thread([i, config, mrpc, batch, &results] {
                score_dataset(i, config, mrpc, batch.value(), results);
            });
        }
    }
    for (int i = 0; i < num_threads; ++i) {
        if (threads[i].joinable()) threads[i].join();
    }
    float scored = scorer.fn(results);
    score.model = config.model;
    score.url = config.endpoint;
    score.dataset_id = config.dataset_id;
    score.score = scored;
}

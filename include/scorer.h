//
// Created by Sanger Steel on 11/19/25.
//

#ifndef COMPARATOR_SCORER_H
#define COMPARATOR_SCORER_H

#include "response_parser.h"
#include "dataset.h"


static std::mutex score_mu;

struct BatchedResult {
    size_t num_rows;
    size_t num_correct;
    float pct;
};

struct OverallScore {
    const char* model;
    const char* url;
    DatasetIds dataset_id;
    float score;

    void report() const;
};

using ScoringFn = std::function<float(const std::vector<BatchedResult>&)>;


enum class ScoreStrategies {
    ACCURACY = 0,
    F1 = 1,
};

constexpr const char* ScoreStrategiesToStr[] = {
    "accuracy",
    "f1"
};

ScoreStrategies scorer_strategy_from_str(const char* str);



struct Scorer {
    ScoringFn fn;
    ScoreStrategies strategy;
    explicit Scorer(enum ScoreStrategies strategy_) : strategy(strategy_) {
        switch (strategy) {
            case ScoreStrategies::ACCURACY:
                fn = Scorer::accuracy_fn; break;
            case ScoreStrategies::F1:
                throw std::runtime_error("F1 scorer not implemented yet");
            default: throw std::runtime_error("no valid score fn given");
        }
    }
    static float accuracy_fn(const std::vector<BatchedResult>& batches) {
        size_t corrects = 0;
        size_t rows_scored = 0;
        for (const auto& batch: batches) {
            corrects += batch.num_correct;
            rows_scored += batch.num_rows;
        }
        return (float) corrects / float(rows_scored);
    }
};


struct ScoreConfig {
    DatasetIds dataset_id;
    const char* endpoint;
    const char* model;
    const char* config;
    const char* split;
    int max_tokens;
    int num_logprobs;
};

BatchedResult ScoreResult_vector_to_BatchedResult(const std::vector<QAResponse>& scores);

BatchedResult score_batch(ScoreConfig& config, const Dataset& dataset, ParquetBatch& batch);

void score_dataset(int idx, ScoreConfig config, const Dataset& dataset, ParquetBatch batch,
                   std::vector<BatchedResult>& results);

void RunScoringTask(OverallScore& score, ScoreConfig config, const Scorer& scorer, int num_threads);


#endif //COMPARATOR_SCORER_H

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
    size_t count_tp;
    size_t count_fp;
    size_t count_tn;
    size_t count_fn;
    float pct;
};

struct OverallScore {
    const char* model;
    const char* url;
    dataset_types::DatasetIds dataset_id;
    float score;
    std::string score_str;

    void report() const;
};

using ScoringFn = std::function<float(const std::vector<BatchedResult>&)>;

namespace scoring_enums {
    DECLARE_ENUM(ScoreStrategies, Accuracy, F1);
}



struct Scorer {
    ScoringFn fn;
    scoring_enums::ScoreStrategies strategy;
    explicit Scorer(scoring_enums::ScoreStrategies strategy_) : strategy(strategy_) {
        switch (strategy) {
            case scoring_enums::Accuracy:
                fn = Scorer::accuracy_fn; break;
            case scoring_enums::F1:
                fn = Scorer::f1_score_fn; break;
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
    static float f1_score_fn(const std::vector<BatchedResult>& batches) {
        float tps = 0, fps = 0, tns = 0, fns = 0;
        size_t rows_scored = 0;
        for (const auto& batch: batches) {
            tps += (float) batch.count_tp;
            fps += (float) batch.count_fp;
            tns += (float) batch.count_tn;
            fns += (float) batch.count_fn;
            rows_scored += batch.num_rows;
        }
        float precision = tps / (tps + fps);
        float recall = tps / (tps + fns);
        return (2 * precision * recall) / (precision + recall);
    }
};


struct ScoreConfig {
    dataset_types::DatasetIds dataset_id;
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

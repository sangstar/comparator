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


struct ScoreResult {
    std::string prompt;
    std::string model;
    bool passed;
};

struct OverallScore {
    const char* model;
    const char* url;
    DatasetIds dataset_id;
    float pct;
};

BatchedResult ScoreResult_vector_to_BatchedResult(const std::vector<ScoreResult>& scores);

BatchedResult score_batch(const char* url, const char* model, const Dataset& dataset, ParquetBatch& batch);


void score_dataset(int idx, const char* url, const char* model, const Dataset& dataset, ParquetBatch batch,
                   std::vector<BatchedResult>& results);

void RunScoringTask(OverallScore& score, DatasetIds dataset_id, const char* url, const char* model, const char* config,
    const char* split, int num_threads);


#endif //COMPARATOR_SCORER_H

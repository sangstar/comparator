//
// Created by Sanger Steel on 11/19/25.
//

#ifndef COMPARATOR_DATASET_H
#define COMPARATOR_DATASET_H

#include "parquet_handler.h"
#include "response_parser.h"

enum class DatasetIds{
    NONE,
    MRPC,
};

constexpr const char* DatasetIdStrs[] = {
    "NONE",
    "MPRC",
};

DatasetIds dataset_id_from_str(const char* str);

struct QAResponse {
    bool yes;
    bool passed;
    std::optional<float> yes_logprob;
    std::optional<float> no_logprob;
};

struct Dataset {
    ParquetTableViewer data;
    std::string (*prompt_creation_fn)(const ParquetRow*);
    QAResponse (*response_scorer_fn)(StreamedCompletions&, const ParquetRow*);
};

Dataset CreateMRPCDataset(const char* config, const char* split);

Dataset CreateDataset(DatasetIds type, const char* config, const char* split);

#endif //COMPARATOR_DATASET_H
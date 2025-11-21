//
// Created by Sanger Steel on 11/19/25.
//

#ifndef COMPARATOR_DATASET_H
#define COMPARATOR_DATASET_H

#include "parquet_handler.h"
#include "response_parser.h"
#include "helpers.h"

DECLARE_ENUM(DatasetIds, NONE, MRPC, COLA);

struct QAResponse {
    bool tp;
    bool fp;
    bool tn;
    bool fn;
    bool passed;
    std::optional<float> yes_logprob;
    std::optional<float> no_logprob;
};

struct Dataset {
    ParquetTableViewer data;
    std::string (*prompt_creation_fn)(const ParquetRow*);
    bool (*label_accessor_fn)(const ParquetRow*);
    QAResponse (*response_scorer_fn)(StreamedCompletions&, const ParquetRow*, bool(*)(const ParquetRow*));
};

Dataset CreateMRPCDataset(const char* config, const char* split);

Dataset CreateColaDataset(const char* split);

Dataset CreateDataset(comparator_enums::DatasetIds type, const char* config, const char* split);

QAResponse yesno_response_scorer(StreamedCompletions& resps, const ParquetRow* row, bool (*label_accessor_fn)(const ParquetRow*));

#endif //COMPARATOR_DATASET_H
//
// Created by Sanger Steel on 11/18/25.
//

#include <catch2/catch_test_macros.hpp>

#include "response_parser.h"
#include "../include/dataset.h"


TEST_CASE("Test mrpc dataset") {
    Dataset mrpc = CreateDataset(DatasetIds::MRPC, "default", "train");
    auto batch = mrpc.data.get_batch().value();
    auto row = batch.get_row(0);
    std::string prompt = mrpc.prompt_creation_fn(&row);

    const char* url = "https://api.openai.com/v1/completions";
    const char* model = "gpt-3.5-turbo-instruct";
    int max_tokens = 1;
    std::string resp = send_openai_query(url, model, prompt.c_str(), max_tokens, 5);
    auto compls = get_responses_from_json(resp.c_str());
    bool found = mrpc.response_scorer_fn(compls, &row);
    if (found) {
        fprintf(stdout, "answered yes");
    }
    return;
}

//
// Created by Sanger Steel on 11/19/25.
//

//
// Created by Sanger Steel on 11/18/25.
//

#include <catch2/catch_test_macros.hpp>

#include "scorer.h"

constexpr int workers = 2;

TEST_CASE("Test scoring") {

    int num_threads_per_task = 5;

    std::thread threads[workers];
    OverallScore scores[workers];

    threads[0] = std::thread([&scores, num_threads_per_task] {
        RunScoringTask(scores[0], DatasetIds::MRPC, "https://api.openai.com/v1/completions",  "babbage-002", "default", "train", num_threads_per_task);
    });
    threads[1] = std::thread([&scores, num_threads_per_task] {
    RunScoringTask(scores[1], DatasetIds::MRPC, "https://api.openai.com/v1/completions",  "gpt-3.5-turbo-instruct", "default", "train", num_threads_per_task);
    });

    for (int i = 0; i < workers; ++i) {
        if (threads[i].joinable()) threads[i].join();
    }
    if (scores[0].pct >= scores[1].pct) {
        std::cout << scores[0].model << " > " << scores[1].model;
    } else {
        std::cout << scores[1].model << " > " << scores[0].model;
    }
    return;
}

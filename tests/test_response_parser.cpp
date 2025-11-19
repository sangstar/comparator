//
// Created by Sanger Steel on 11/17/25.
//

#include <iostream>

#include "../include/response_parser.h"
#include "catch2/catch_test_macros.hpp"


TEST_CASE("Test get compl") {
    const char* url = "https://api.openai.com/v1/completions";
    const char* model = "gpt-3.5-turbo-instruct";
    const char* prompt = "Write me a song";
    int max_tokens = 50;
    std::string resp = send_openai_query(url, model, prompt, max_tokens, 5);
    printf("%s\n", resp.c_str());
    auto compls = get_responses_from_json(resp.c_str());
    printf("Done");
}

//
// Created by Sanger Steel on 11/18/25.
//

#ifndef COMPARATOR_RESPONSE_PARSER_H
#define COMPARATOR_RESPONSE_PARSER_H

#include "curl_handler.h"
#include <nlohmann/json.hpp>
#include <unordered_map>

#define MAX_CHOICES 10
#define MAX_TOKENS 10
#define MAX_LOGPROBS 100
#define MAX_RESPONSES 100
#define PARSER_BUF_SIZE (1024 * 3)

using json = nlohmann::json;


enum class ParseStates {
    NONE,
    PARSING,
};


struct logprob {
    std::string token;
    float prob;
};


struct OAIResponseChoice {
    std::string text;
    std::vector<std::string> tokens;
    std::vector<float> token_logprobs;

    OAIResponseChoice() = default;
    OAIResponseChoice(const OAIResponseChoice&) = delete;
    OAIResponseChoice(OAIResponseChoice&&) noexcept = default;
    OAIResponseChoice& operator=(OAIResponseChoice&&) noexcept = default;

    std::vector<logprob> top_logprobs;

    std::string finish_reason;
    unsigned int index{};
};

struct OAIResponse {
    std::string id;
    std::string object;
    unsigned int created;


    std::vector<OAIResponseChoice> choices;

    std::string model;

    static OAIResponse from_json(json j);
};

using StreamedCompletions = std::vector<OAIResponse>;

bool is_valid_resp(std::string resp);

StreamedCompletions get_responses_from_json(const char* json_str);

#endif //COMPARATOR_RESPONSE_PARSER_H

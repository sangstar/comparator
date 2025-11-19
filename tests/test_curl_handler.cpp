//
// Created by Sanger Steel on 11/17/25.
//

#include <catch2/catch_test_macros.hpp>
#include <curl/curl.h>
#include "../include/curl_handler.h"


TEST_CASE("Test add header") {
    CurlHandler ch = CurlHandler::make_from_url("https://api.openai.com/v1/completions");
    ch = ch.add_header("foo bar");
    REQUIRE(
        strcmp(ch.headers[ch.num_headers - 1], "foo bar") == 0
    );
}

TEST_CASE("Test set payload") {
    CurlHandler ch = CurlHandler::make_from_url("https://api.openai.com/v1/completions");

    char payload[256] = {};
    snprintf(payload,
             sizeof(payload),
             "{"
             "\"model\": \"%s\","
             "\"prompt\": \"%s\","
             "\"max_tokens\": %d,"
             "\"stream\": true"
             "}",
             "gpt-3.5-turbo-instruct",
             "Hello!",
             10);

    ch = ch.set_payload(payload);
    REQUIRE(
        strcmp(ch.payload, payload) == 0
    );
}

TEST_CASE("Test perform curl") {
    const char* url = "https://api.openai.com/v1/completions";
    const char* model = "gpt-3.5-turbo-instruct";
    const char* prompt = "Write me a song";
    int max_tokens = 50;
    fprintf(stdout, "%s\n", send_openai_query(url, model, prompt, max_tokens, 20).c_str());
}

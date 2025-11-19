//
// Created by Sanger Steel on 11/17/25.
//

#ifndef COMPARATOR_CURL_HANDLER_H
#define COMPARATOR_CURL_HANDLER_H

#include <string>
#include <curl/curl.h>

#define MAX_RESPONSE_BUFFER_SIZE (1024 * 100)

struct CurlResponseBuf {
    char data[MAX_RESPONSE_BUFFER_SIZE];
    size_t size;
    int finished;
};

struct CurlHandler {
    const char* url;
    const char* payload;
    const char* headers[8];
    size_t num_headers;

    CurlHandler add_header(const char* header) const;

    static CurlHandler make_from_url(const char* url);

    CurlHandler set_payload(const char* payload) const;

    static size_t default_write_callback(void* contents, size_t size, size_t nmemb, void* userp);

    [[nodiscard]] std::string perform() const;
};

std::string send_openai_query(const char* url, const char* model, const char* prompt, int max_tokens, int logprobs);
#endif //COMPARATOR_CURL_HANDLER_H

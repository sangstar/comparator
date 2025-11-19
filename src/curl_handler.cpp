//
// Created by Sanger Steel on 11/17/25.
//

#include "curl_handler.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>


CurlHandler CurlHandler::add_header(const char* header) const {
    CurlHandler out = *this;
    out.headers[out.num_headers++] = header;
    return out;
}

CurlHandler CurlHandler::make_from_url(const char* url) {
    CurlHandler ch = CurlHandler{
        .url = url,
        .payload = nullptr,
        .headers = {},
        .num_headers = 0,
    };
    return ch;
}

CurlHandler CurlHandler::set_payload(const char* payload) const {
    CurlHandler out = *this;
    out.payload = payload;
    return out;
}

size_t CurlHandler::default_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    auto buf = static_cast<CurlResponseBuf *>(userp);
    if (buf->size + total >= MAX_RESPONSE_BUFFER_SIZE) {
        throw std::runtime_error("exceeded max response buf size");
    }
    memcpy(buf->data + buf->size, static_cast<const char *>(contents), total);
    buf->size += total;
    if (strstr(buf->data, "[DONE]") != NULL) {
        buf->finished = 1;
    }
    return total;
}

std::string CurlHandler::perform() const {


    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();
    if (!curl) exit(1);

    struct curl_slist* c_headers = NULL;
    if (num_headers > 0) {
        for (int i = 0; i < num_headers; ++i) {
            c_headers = curl_slist_append(c_headers, headers[i]);
        }
    }

    CurlResponseBuf buf = {};

    curl_easy_setopt(curl, CURLOPT_URL, url);
    fprintf(stdout, "curling %s\n", url);
    if (payload != nullptr) curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, c_headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlHandler::default_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);


    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(c_headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK) {
        std::exit(1);
    }
    return {buf.data};
}

std::string escape_str(std::string_view str) {
    std::string out;
    out.reserve(str.size() * 2);
    for (char c : str) {
        switch (c) {
            case '\\': out.append("\\\\"); break;
            case '\"': out.append("\\\""); break;
            case '\n': out.append("\\n"); break;
            case '\r': out.append("\\r"); break;
            case '\t': out.append("\\t"); break;
            default: out += c; break;
        }
    }
    return out;
}

std::string send_openai_query(const char* url, const char* model, const char* prompt, int max_tokens, int logprobs) {
    CurlHandler ch = CurlHandler::make_from_url(url);
    ch = ch.add_header("Content-Type: application/json");

    char* api_key = getenv("OPENAI_API_KEY");
    if (!api_key) {
        exit(1);
    }

    char api_key_buf[256];
    snprintf(api_key_buf, sizeof(api_key_buf), "Authorization: Bearer %s", api_key);

    ch = ch.add_header(api_key_buf);

    std::string payload;
    payload.append(
        "{"
        "\"model\": \""
        );
    payload.append(model);
    payload.append(
        "\","
        "\"prompt\": \""
        );
    payload.append(escape_str(prompt));
    payload.append(
        "\","
        "\"max_tokens\": "
        );
    payload.append(std::to_string(max_tokens));
    payload.append(
    ","
    "\"logprobs\": "
    );
    payload.append(std::to_string(logprobs));
    payload.append(
        ","
        "\"stream\": true"
        "}"
    );
    curl_global_init(CURL_GLOBAL_ALL);
    CURL* curl = curl_easy_init();
    if (!curl) exit(1);

    struct curl_slist* headers = NULL;
    for (int i = 0; i < ch.num_headers; ++i) {
        headers = curl_slist_append(headers, ch.headers[i]);
    }

    ch = ch.set_payload(payload.c_str());

    CurlResponseBuf buf = {};

    curl_easy_setopt(curl, CURLOPT_URL, ch.url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, ch.payload);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CurlHandler::default_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, std::strlen(ch.payload));


    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK) {
        std::exit(1);
    }
    return {buf.data};
}

//
// Created by Sanger Steel on 11/18/25.
//

#include "response_parser.h"

OAIResponse OAIResponse::from_json(json j) {
    OAIResponse resp = {};

    resp.id = j["id"];

    resp.object = j["object"];

    resp.model = j["model"];

    resp.created = j["created"];

    for (auto& choice: j["choices"]) {
        OAIResponseChoice c;
        for (auto& [k, v]: choice.items()) {
            if (k == "text") {
                std::string str = v;
                c.text = str;
            }
            if (k == "index") {
                c.index = v;
            }
            if (k == "finish_reason") {
                std::string str;
                if (v.is_null()) {
                    str = "null";
                } else {
                    str = v;
                }
                c.finish_reason = str;
            }
            if (k == "logprobs") {
                for (auto& [k_, v_]: choice["logprobs"].items()) {
                    if (k_ == "tokens") {
                        for (auto& token: v_) {
                            std::string str = token;
                            c.tokens.emplace_back(str);
                        }
                    }
                    if (k_ == "token_logprobs") {
                        for (auto& logprob: v_) {
                            c.token_logprobs.emplace_back(logprob);
                        }
                    }
                    if (k_ == "top_logprobs") {
                        for (auto& top_logprob: v_) {
                            std::unordered_map<std::string, float> map = top_logprob;
                            for (auto& [name, prob]: map) {
                                logprob lb;
                                lb.prob = prob;
                                lb.token = name;
                                c.top_logprobs.emplace_back(lb);
                            }
                        }
                    }
                }
            }
        }
        resp.choices.emplace_back(std::move(c));
    }


    return resp;
}

bool is_valid_resp(std::string resp) {
    char buf[20] = {};
    memcpy(buf, resp.c_str(), 20);
    return strstr(buf, "error") == nullptr;
}

StreamedCompletions get_responses_from_json(const char* json_str) {
    StreamedCompletions resps;
    ParseStates state = ParseStates::NONE;

    struct _buf {
        char buf[PARSER_BUF_SIZE];
        size_t len;

        [[nodiscard]] bool finished_json_chunk() const {
            return len >= 6 &&
                   buf[len - 6] == '\n' &&
                   buf[len - 5] == '\n' &&
                   buf[len - 4] == 'd' &&
                   buf[len - 3] == 'a' &&
                   buf[len - 2] == 't' &&
                   buf[len - 1] == 'a';
        }

        void append(char c) {
            if (len == PARSER_BUF_SIZE) throw std::runtime_error("max buf limit reached");
            buf[len] = c;
            buf[len + 1] = '\0';
            len++;
        }

        _buf copy(size_t start, size_t stop) {
            if (stop >= PARSER_BUF_SIZE) throw std::runtime_error("invalid stop size");
            _buf b = {};
            memset(b.buf, 0, PARSER_BUF_SIZE);
            memcpy(b.buf, this->buf + start, stop);
            b.len = stop - start;
            return b;
        }

        static _buf reset() {
            _buf b = {};
            memset(b.buf, 0, PARSER_BUF_SIZE);
            b.len = 0;
            return b;
        }
    };

    _buf buf = {};
    memset(buf.buf, 0, PARSER_BUF_SIZE);
    buf.len = 0;

    size_t json_len = strlen(json_str);
    for (int i = 0; i < json_len; ++i) {
        char c = json_str[i];
        if (state == ParseStates::NONE && c == '{') state = ParseStates::PARSING;
        if (state == ParseStates::PARSING) {
            buf.append(c);
        }
        if (buf.finished_json_chunk()) {
            state = ParseStates::NONE;
            _buf json_b = buf.copy(0, buf.len - 6);
            try {
                resps.emplace_back(OAIResponse::from_json(json::parse(json_b.buf)));
            } catch (const json::parse_error& e) {
                fprintf(stderr, "failed to process due to invaid json: %s", json_b.buf);
            }
            buf = _buf::reset();
        }
    }
    return resps;
}

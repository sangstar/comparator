//
// Created by Sanger Steel on 11/19/25.
//

#include "dataset.h"

#include <cfloat>

std::string normalize(std::string_view sv) {
    std::string out;
    out.reserve(sv.size());
    for (char c : sv)
        if (c != ' ') out.push_back(std::tolower((unsigned char)c));
    return out;
}

bool search_yes(std::string_view s) {
    std::string str = normalize(s);
    return str == "yes" ||
           str == "yep" ||
           str == "ye"  ||
           str == "y";
}

bool search_no(std::string_view s) {
    std::string str = normalize(s);
    return str =="nope" ||
           str == "no"  ||
           str == "n";
}

static auto str_contains = [](std::string_view str, std::string_view cmp) -> bool {
    return strstr(str.data(), cmp.data()) != NULL;
};


const char* id_to_str(DatasetIds id) {
    switch (id) {
        case DatasetIds::NONE: return "NONE";
        case DatasetIds::MRPC: return "MRPC";
        default:
            return "NONE";
    }
}

DatasetIds dataset_id_from_str(const char* str) {
    if (strstr(str, "mrpc") != nullptr) return DatasetIds::MRPC;
    return DatasetIds::NONE;
}

std::optional<float> maybe_find_logprob(const std::function<bool(std::string&)>& search_fn, OAIResponseChoice& choice) {
    float best_logprob = -INFINITY;
    bool found = false;
    for (auto& lp : choice.top_logprobs) {
        if (search_fn(lp.token)) {
            // In case the logprobs have both, e.g., a "yes" and " yes"
            // separately, pick the best logprob out of them
            found = true;
            if (lp.prob > best_logprob) best_logprob = lp.prob;
        }
    }
    return found ? std::optional<float>(best_logprob) : std::nullopt;
}


QAResponse yesno_response_scorer(StreamedCompletions& resps, const ParquetRow* row) {
    QAResponse resp{false, false, false, false};

    size_t size = resps.size();
    bool label = static_cast<bool>(std::get<uint64_t>((*row)[2]));

    for (size_t i = 0; i < size; ++i) {
        for (auto& choice : resps[i].choices) {
            if (choice.text.empty()) continue;
            auto yes_logprob = maybe_find_logprob(search_yes, choice);
            auto no_logprob = maybe_find_logprob(search_no, choice);
            bool is_yes = yes_logprob.has_value() && no_logprob.has_value() && yes_logprob.value() > no_logprob.value();
            bool is_no = yes_logprob.has_value() && no_logprob.has_value() && yes_logprob.value() < no_logprob.value();

            if (is_yes) {
                resp.yes_logprob = yes_logprob;
                resp.no_logprob = no_logprob;
                if (label && is_yes) {
                    resp.passed = true;
                    resp.tp = true;
                } else {
                    resp.passed = false;
                    resp.fp = true;
                }
                return resp;
            }
            if (is_no) {
                resp.no_logprob = no_logprob;
                resp.yes_logprob = yes_logprob;
                if (!label && is_no) {
                    resp.passed = true;
                    resp.tn = true;
                } else {
                    resp.passed = false;
                    resp.fn = true;
                }
                return resp;
            }
        }
    }
    resp.passed = false;
    label ? resp.fn = true : resp.fp = true;
    return resp;
}

Dataset CreateMRPCDataset(const char* config, const char* split) {
    ParquetTableViewer data = get_hf_dataset("SetFit/mrpc", config, split);
    auto mrpc_prompt_creation_fn = [](const ParquetRow* row) -> std::string {
        std::string s;
        std::string_view text1 = std::get<std::string_view>((*row)[0]);
        std::string_view text2 = std::get<std::string_view>((*row)[1]);
        s.append("Is the following sentence pair semantically equivalent? Yes or no.\\nSentence 1: \\n");
        s.append(text1);
        s.append("\\n\\n");
        s.append("Sentence 2: \\n");
        s.append(text2);
        s.append("\\n\\nAnswer: ");
        return s;
    };
    return Dataset{data,mrpc_prompt_creation_fn, yesno_response_scorer };
}

Dataset CreateDataset(DatasetIds type, const char* config, const char* split) {
    switch (type) {
        case DatasetIds::MRPC: return CreateMRPCDataset(config, split);
        default:
            throw std::runtime_error("Cannot make dataset for this type");
    }
}

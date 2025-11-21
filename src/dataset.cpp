//
// Created by Sanger Steel on 11/19/25.
//

#include "dataset.h"

inline std::string to_lower_copy(std::string_view sv) {
    std::string out;
    out.reserve(sv.size());
    for (char c : sv)
        out.push_back(std::tolower((unsigned char)c));
    return out;
}

inline bool contains(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string::npos;
}


inline bool search_yes(std::string_view s) {
    auto lower = to_lower_copy(s);
    return contains(lower, "yes") ||
           contains(lower, "yep") ||
           contains(lower, "ye")  ||
           contains(lower, "y");
}

inline bool search_no(std::string_view s) {
    auto lower = to_lower_copy(s);
    return contains(lower, "nope") ||
           contains(lower, "no")   ||
           contains(lower, "n");
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

std::optional<float> maybe_find_other_logprob(const std::function<bool(std::string&)>& search_fn, OAIResponseChoice& choice) {
    for (auto& lp : choice.top_logprobs) {
        if (search_fn(lp.token)) {
            return lp.prob;
        }
    }
    return std::nullopt;
}


QAResponse default_response_scorer(StreamedCompletions& resps, const ParquetRow* row) {
    QAResponse resp{false, false};

    size_t size = resps.size();

    for (size_t i = 0; i < size; ++i) {
        for (auto& choice : resps[i].choices) {
            if (choice.text.empty()) continue;
            bool label = static_cast<bool>(std::get<uint64_t>((*row)[2]));
            bool is_yes = search_yes(choice.text);
            bool is_no = search_no(choice.text);
            if (is_yes) {
                resp.yes = true;
                resp.yes_logprob = choice.token_logprobs[0];
                resp.no_logprob = maybe_find_other_logprob(search_no, choice);
                if (label && is_yes) resp.passed = true;
            }
            else if (is_no) {
                resp.yes = false;
                resp.no_logprob = choice.token_logprobs[0];
                resp.yes_logprob = maybe_find_other_logprob(search_yes, choice);
                if (label && is_no) resp.passed = true;
            }
        }
    }
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
    return Dataset{data,mrpc_prompt_creation_fn, default_response_scorer };
}

Dataset CreateDataset(DatasetIds type, const char* config, const char* split) {
    switch (type) {
        case DatasetIds::MRPC: return CreateMRPCDataset(config, split);
        default:
            throw std::runtime_error("Cannot make dataset for this type");
    }
}

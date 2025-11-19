//
// Created by Sanger Steel on 11/19/25.
//

#include "dataset.h"

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
    auto mrpc_scorer_fn = [](StreamedCompletions& resps, const ParquetRow* row) -> bool {
        std::string text;

        size_t tolerance = 3; // Allow this many tokens as grace to give a parseable answer
        size_t size = resps.size();
        size_t its = size <= tolerance ? size : tolerance;
        for (size_t i = 0; i < its; ++i) {
            for (auto& choice : resps[i].choices) {
                text.append(choice.text);
            }
        }


        auto str_contains = [](std::string_view str, std::string_view cmp) -> bool {
            return strstr(str.data(), cmp.data()) != NULL;
        };

        bool label = static_cast<bool>(std::get<uint64_t>((*row)[2]));
        bool yes_conds = str_contains(text,"yes") || str_contains(text, "Yes") || str_contains(text, "y") || str_contains(text, "Y") || str_contains(text, "ye");
        return yes_conds && label;
    };
    return Dataset{data,mrpc_prompt_creation_fn, mrpc_scorer_fn };
}

Dataset CreateDataset(DatasetIds type, const char* config, const char* split) {
    switch (type) {
        case DatasetIds::MRPC: return CreateMRPCDataset(config, split);
        default:
            throw std::runtime_error("Cannot make dataset for this type");
    }
}

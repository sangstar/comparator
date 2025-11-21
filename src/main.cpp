//
// Created by Sanger Steel on 11/17/25.
//

#include <string>
#include <vector>
#include "scorer.h"

struct Arg {
    std::string arg;
    std::string value;
    const char* desc;
    const char* default_value;
};

struct ArgSet {
    std::vector<Arg*> args;
    void validate() {
        for (auto& arg : args) {
            if (arg->value.empty()) {
                if (arg->default_value) {
                    arg->value = arg->default_value;
                } else throw std::runtime_error("required arg " + arg->arg + " unset");
            }

        }
    }
    void parse_args(int argc, char** argv) const {
        for (int i = 2; i < argc; ++i) {
            std::string_view arg = argv[i];
            for (auto& a: args) {
                if (a->arg == arg) {
                    a->value = argv[i+1];
                }
            }
        }
    }
    const char* get(const char* name) const {
        for (const auto& arg_ : args) {
            if (arg_->arg == name) {
                return arg_->value.data();
            }
        }
        return "null";
    }
};

int main(int argc, char** argv) {
    if (argc <= 1) {
        throw std::runtime_error("no args provided");
    }
    std::string_view task = argv[1];

    Arg max_tokens {"--max-tokens", "", "max tokens for model responses", "5"};
    Arg logprobs {"--num-logprobs", "", "num logprobs to return from model responses", "5"};
    Arg workers {"--workers", "", "number of workers to use per dataset scoring", "5"};
    Arg scorer {"--scorer", "", "metric to score with e.g. accuracy, f1, etc", "accuracy"};

    ArgSet args;

    if (task == "evaluate") {
        Arg model {"--model", "", "model id for inference", nullptr};
        Arg endpoint {"--endpoint", "", "endpoint uri", nullptr};
        Arg dataset {"--dataset", "", "hf dataset id to use, e.g. SetFit/mrpc", nullptr};
        Arg config {"--config", "", "hf dataset id config to use", "default"};
        Arg split {"--split", "", "hf dataset split to use (e.g. train, test..)", "train"};
        args = { std::vector<Arg*>{&model, &endpoint, &dataset, &config, &split}};
    }
    else if (task == "compare") {
        Arg model_a {"--model_a", "", "model id for model A", nullptr};
        Arg endpoint_a {"--endpoint_a", "", "endpoint uri for model A", nullptr};
        Arg model_b {"--model_b", "", "model id for model B", nullptr};
        Arg endpoint_b {"--endpoint_b", "", "endpoint uri for model B", nullptr};
        Arg dataset {"--dataset", "", "hf dataset id to use, e.g. SetFit/mrpc", nullptr};
        Arg config {"--config", "", "hf dataset id config to use", "default"};
        Arg split {"--split", "", "hf dataset split to use (e.g. train, test..)", "train"};
        args = { std::vector<Arg*>{&model_a, &endpoint_a, &model_b, &endpoint_b, &dataset, &config, &split}};
    } else {
        throw std::runtime_error("invalid subcommand: " + std::string(task));
    }

    args.args.emplace_back(&max_tokens);
    args.args.emplace_back(&logprobs);
    args.args.emplace_back(&workers);
    args.args.emplace_back(&scorer);


    args.parse_args(argc, argv);
    args.validate();

    int n_workers = std::stoi(workers.value);
    int n_tokens = std::stoi(max_tokens.value);
    int n_logprobs = std::stoi(logprobs.value);

    DatasetIds id = dataset_id_from_str(args.get("--dataset"));

    Scorer score_strategy = Scorer(scorer_strategy_from_str(args.get("--scorer")));
    if (task == "evaluate") {
        OverallScore score{};
        ScoreConfig cfg = {id, args.get("--endpoint"), args.get("--model"), args.get("--config"), args.get("--split"), n_tokens, n_logprobs};
        RunScoringTask(score, cfg, score_strategy, n_workers);
        score.report();
    } else if (task == "compare") {
        std::thread threads[2];
        OverallScore scores[2];
        threads[0] = std::thread([&scores, id, args, score_strategy, n_logprobs, n_tokens, n_workers] {
            RunScoringTask(scores[0], ScoreConfig{id, args.get("--endpoint_a"), args.get("--model_a"), args.get("--config"), args.get("--split"), n_tokens, n_logprobs}, score_strategy, n_workers);
        });
        threads[1] = std::thread([&scores, id, args, score_strategy, n_logprobs, n_tokens, n_workers] {
            RunScoringTask(scores[1], ScoreConfig{id, args.get("--endpoint_b"), args.get("--model_b"), args.get("--config"), args.get("--split"), n_tokens, n_logprobs}, score_strategy, n_workers);
        });

        for (int i = 0; i < 2; ++i) {
            if (threads[i].joinable()) threads[i].join();
        }

        for (auto& score : scores) {
            score.report();
        }
    } else {
        throw std::runtime_error("invalid subcommand: " + std::string(task));
    }


    return 0;
}

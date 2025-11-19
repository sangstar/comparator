//
// Created by Sanger Steel on 11/18/25.
//

#include <catch2/catch_test_macros.hpp>
#include "../include/parquet_handler.h"


TEST_CASE("Test dataset get uri") {
    const char* desired = "https://huggingface.co/api/datasets/SetFit/mrpc/parquet/default/train";
    auto constructor = HFDatasetURLConstructor{"SetFit/mrpc", "default", "train"};
    auto constructed = constructor.construct();
    REQUIRE(strcmp(constructed.c_str(), desired) == 0);
}

TEST_CASE("Test dataset get") {
    auto res = get_hf_dataset("SetFit/mrpc", "default", "train");
    auto batch = res.get_batch().value();
    auto row = batch.get_row(0);
    auto field_0 = row[0];
    REQUIRE(std::get<std::string_view>(field_0) == "Amrozi accused his brother , whom he called \" the witness \" , of deliberately distorting his evidence .");
}


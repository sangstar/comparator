//
// Created by Sanger Steel on 11/18/25.
//

#ifndef COMPARATOR_PARQUET_HANDLER_H
#define COMPARATOR_PARQUET_HANDLER_H

#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <arrow/api.h>
#include <parquet/arrow/reader.h>


constexpr const char* basic_fmt_string =
        "https://huggingface.co/api/datasets/%s/parquet/%s/%s";


constexpr const char* default_fp = "/tmp/p_%s_.parquet";


using ParquetVal = std::variant<uint64_t, std::string_view>;
using ParquetColumnVals = std::variant<std::vector<uint64_t>, std::vector<std::string_view>>;


using GetterFn = ParquetVal (*)(arrow::Array* arr, size_t idx);


struct HFDatasetURLConstructor {
    const char* dataset;
    const char* config;
    const char* split;


    static std::string get_dataset_parsed(const char* dataset);

    static std::string construct(const char* dataset, const char* config, const char* split);

    [[nodiscard]] std::string construct() const;
};


enum class HFDatasetRespResult {
    Success,
    Error,
    RateLimit
};

struct HFDatasetResp {
    std::string resp;
    unsigned int offset;
    unsigned int length;
    HFDatasetRespResult result;
};


HFDatasetRespResult get_resp_result(std::string& resp);

std::string parse_hf_parquet_file_resp(std::string& resp);

std::string get_dataset_uri(const char* dataset, const char* config, const char* split);


size_t write_file(void* ptr, size_t size, size_t nmemb, void* userdata);

bool download_parquet_dataset(const char* url, const char* out_path);

const char* download_hf_dataset(const char* outpath, const char* dataset, const char* config, const char* split);

std::vector<std::string> strs_from_ArrayData(std::shared_ptr<arrow::ArrayData> data);

std::vector<uint64_t> u64s_from_ArrayData(std::shared_ptr<arrow::ArrayData> data);


struct ParquetColumnEntry {
    std::string_view name;
    arrow::Array *data;
    GetterFn getter;

    static ParquetVal get(arrow::Array* data, GetterFn getter, size_t idx);
};

struct ParquetColumn {
    std::string_view name;
    arrow::Type::type type;
    std::shared_ptr<arrow::Array> arr;

    GetterFn get_int64 = [](arrow::Array* arr, size_t idx){
        auto a = static_cast<arrow::Int64Array*>(arr);
        return ParquetVal((uint64_t)a->Value(idx));
    };

    GetterFn get_int32 = [](arrow::Array* arr, size_t idx){
        auto a = static_cast<arrow::Int32Array*>(arr);
        return ParquetVal((uint32_t)a->Value(idx));
    };

    GetterFn get_string = [](arrow::Array* arr, size_t idx){
        auto a = static_cast<arrow::StringArray*>(arr);
        return ParquetVal(a->GetView(idx));
    };

    ParquetVal at(size_t idx) const;
};


struct ParquetRow {
    unsigned int idx;
    const std::vector<ParquetColumn>* cols;
    ParquetVal operator[](size_t col_idx) const;
};

struct ParquetBatch {
    std::vector<ParquetColumn> cols;
    const int64_t num_rows;
    explicit ParquetBatch(const std::shared_ptr<arrow::RecordBatch>& batch);

    ParquetRow get_row(unsigned int idx) const;
};

struct ParquetTableViewer {
    std::shared_ptr<arrow::Table> table;
    arrow::TableBatchReader reader;
    std::shared_ptr<arrow::RecordBatch> batch;
    explicit ParquetTableViewer(const std::shared_ptr<arrow::Table>& table_);;

    std::optional<ParquetBatch> get_batch();
};

ParquetTableViewer get_hf_dataset(const char* dataset, const char* config, const char* split);

#endif //COMPARATOR_PARQUET_HANDLER_H

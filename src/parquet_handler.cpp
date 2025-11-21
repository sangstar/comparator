//
// Created by Sanger Steel on 11/18/25.
//

#include "parquet_handler.h"
#include <parquet/api/reader.h>
#include <arrow/io/api.h>
#include <arrow/status.h>
#include "curl_handler.h"

#include <filesystem>
namespace fs = std::filesystem;



std::string HFDatasetURLConstructor::get_dataset_parsed(const char* dataset) {
    const char *p = strchr(dataset, '/');

    std::string out;
    out.reserve(strlen(dataset) + sizeof("%2F") - 1);
    out.append(dataset, p);
    out.append("%2F");
    out.append(p + 1);
    return out;
}

std::string HFDatasetURLConstructor::construct(const char* dataset, const char* config, const char* split) {
    char buf[512] = {};
    snprintf(buf, sizeof(buf), fmt_string, dataset, config, split);
    return {buf};
}

std::string HFDatasetURLConstructor::construct() const {
    return HFDatasetURLConstructor::construct(dataset, config, split);
}

HFDatasetRespResult get_resp_result(std::string& resp) {
    bool error_msg = resp[0] == '{'
                     && resp[1] == '\"'
                     && resp[2] == 'e'
                     && resp[3] == 'r'
                     && resp[4] == 'r'
                     && resp[5] == 'o'
                     && resp[6] == 'r'
                     && resp[7] == '\"';
    bool too_many_requests = strstr(resp.c_str(), "<!DOCTYPE") != NULL && strstr(resp.c_str(), "429");
    if (error_msg) return HFDatasetRespResult::Error;
    if (too_many_requests) return HFDatasetRespResult::RateLimit;
    return HFDatasetRespResult::Success;
}

std::string parse_hf_parquet_file_resp(std::string& resp) {
    if (resp[0] == '[' && resp[1] == '\"') {
        std::string out;
        out.append(resp.c_str() + 2);
        if (out.back() == ']' && out[out.size() - 2] == '\"') {
            out.pop_back();
            out.pop_back();
        }
        return out;
    }
    return resp;
}

std::string get_dataset_uri(const char* dataset, const char* config, const char* split) {
    std::string query = HFDatasetURLConstructor::construct(dataset, config, split);
    CurlHandler ch = CurlHandler::make_from_url(query.c_str());
    std::string resp = ch.perform();
    return parse_hf_parquet_file_resp(resp);
}

size_t write_file(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* out = static_cast<std::ofstream*>(userdata);
    out->write(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

bool download_parquet_dataset(const char* url, const char* out_path) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::ofstream out(out_path, std::ios::binary);
    if (!out.is_open()) return false;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);


    auto res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

const char* download_hf_dataset(const char* outpath, const char* dataset, const char* config, const char* split) {
    auto uri = get_dataset_uri(dataset, config, split);
    fprintf(stdout, "downloading dataset %s", uri.c_str());
    bool success = download_parquet_dataset(uri.c_str(), outpath);
    if (!success) {
        throw std::runtime_error("curl failed");
    }
    return outpath;
}

std::vector<std::string> strs_from_ArrayData(std::shared_ptr<arrow::ArrayData> data) {
    std::vector<std::string> strs;
    const int32_t* offsets = data->GetValues<int32_t>(1);
    const uint8_t* chars   = data->GetValues<uint8_t>(2);
    for (int i = 0; i < data->length; ++i) {
        std::string s((const char*)chars + offsets[i], offsets[i+1] - offsets[i]);
        strs.emplace_back(std::move(s));
    }
    return strs;
}

std::vector<uint64_t> u64s_from_ArrayData(std::shared_ptr<arrow::ArrayData> data) {
    std::vector<uint64_t> u64s;
    const int64_t* vals = data->GetValues<int64_t>(1);
    for (int i = 0; i < data->length; ++i) {
        u64s.emplace_back(vals[i + data->offset]);
    }
    return u64s;
}

ParquetVal ParquetColumnEntry::get(arrow::Array* data, GetterFn getter, size_t idx) {
    return getter(data, idx);
}

ParquetVal ParquetColumn::at(size_t idx) const {
    switch (type) {
        case arrow::Type::STRING: {
            return ParquetColumnEntry::get(arr.get(), get_string, idx);
        }
        case arrow::Type::INT64: {
            return ParquetColumnEntry::get(arr.get(), get_int64, idx);
        }
    }
    throw std::runtime_error("unsupported type");
}

ParquetVal ParquetRow::operator[](size_t col_idx) const {
    return (*cols)[col_idx].at(idx);
}

ParquetBatch::ParquetBatch(const std::shared_ptr<arrow::RecordBatch>& batch) : num_rows(batch->num_rows()) {
    for (int i = 0; i < batch->num_columns(); i++) {
        auto name = batch->schema()->field(i)->name();
        auto type = batch->column(i)->data()->type->id();
        auto data = batch->column(i);
        cols.emplace_back(ParquetColumn{name, type, data});
    }
}

ParquetRow ParquetBatch::get_row(unsigned int idx) const {
    return ParquetRow{idx, &cols};
}

ParquetTableViewer::ParquetTableViewer(const std::shared_ptr<arrow::Table>& table_): table(table_),
    reader(*table_) {
}

std::optional<ParquetBatch> ParquetTableViewer::get_batch() {
    if (reader.ReadNext(&batch).ok() && batch) {
        return ParquetBatch(batch);
    }
    return std::nullopt;
}

static thread_local std::string filestr;

ParquetTableViewer get_hf_dataset(const char* dataset, const char* config, const char* split) {
    filestr.append("/tmp/p_");
    size_t i = strcspn(dataset, "/");
    filestr.append(dataset + i + 1);
    filestr.append("_");
    filestr.append(config);
    filestr.append("_");
    filestr.append(split);
    filestr.append(".parquet");

    const char *fp = filestr.c_str();
    if (!fs::exists(fp)) {
        fp = download_hf_dataset(fp, dataset, config, split);
    }

    std::unique_ptr<parquet::arrow::FileReader> pq_reader;

    auto pq_initializer = [fp, &pq_reader]() {
        ARROW_ASSIGN_OR_RAISE(
            auto infile,
            arrow::io::ReadableFile::Open(fp, arrow::default_memory_pool())
        );

        ARROW_ASSIGN_OR_RAISE(
            pq_reader,
            parquet::arrow::OpenFile(
                infile,
                arrow::default_memory_pool()
            )
        );
        return arrow::Status::OK();
    }();

    if (pq_initializer != arrow::Status::OK()) {
        throw std::runtime_error("got bad arrow status on parquet creation");
    }

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(pq_reader->ReadTable(&table));

    return ParquetTableViewer(table);
}

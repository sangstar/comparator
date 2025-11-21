//
// Created by Sanger Steel on 11/21/25.
//

#ifndef COMPARATOR_HELPERS_H
#define COMPARATOR_HELPERS_H
#include <vector>


inline std::vector<std::string> str_split(std::string_view str) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = str.find(',', start);
        if (pos == std::string::npos) {
            out.emplace_back(str.substr(start));
            break;
        }
        out.emplace_back(str.substr(start, pos - start));
        start = pos + 1;
        while (start < str.size() && str[start] == ' ')
            start++; // trim single spaces after commas
    }
    return out;
}


#define ENUM_FROM_STR(e, str, ret) \
    do { \
        std::vector<std::string> vec = str_split(e##Strs); \
        int i = 0; \
        for (const auto& v : vec) { \
            if (std::string_view(vec[i]) == str) ret = e##List[i]; \
            i++; \
        } \
    } while (0)


#define DECLARE_ENUM(name, ...) \
    namespace comparator_enums { \
        enum name { __VA_ARGS__ }; \
        constexpr const char* name##Strs = #__VA_ARGS__; \
        constexpr name name##List[] = { __VA_ARGS__ }; \
        inline std::string name##_to_str(size_t id) { \
        auto vals = str_split(name##Strs); \
        return vals[id]; \
        } \
        inline name str_to_##name(std::string_view str) { \
        name ret; \
        ENUM_FROM_STR(name, str, ret); \
        return ret; \
        } \
    }




#endif //COMPARATOR_HELPERS_H
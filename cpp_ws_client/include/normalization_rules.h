#ifndef NORMALIZATION_H // Check if this symbol is not defined
#define NORMALIZATION_H // Define the symbol to prevent future inclusions

// Include necessary libraries
#include <string>
#include <unordered_map>

// Normalization rule structure
struct NormalizationRule {
    std::string symbol_key;
    std::string price_key;
    std::string volume_key;
    std::string timestamp_key;
    std::string event_key;
    std::string event_value;
};

// Global mapping for multiple exchanges
extern std::unordered_map<std::string, NormalizationRule> exchange_normalization_rules;

#endif // End of the include guard

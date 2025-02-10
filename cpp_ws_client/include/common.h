#ifndef COMMON_H
#define COMMON_H

#include <map>
#include <string>
#include <vector>

using namespace std; // Allows using map, string, and vector without std::

namespace DataSources {
    // Common map for both WebSockets & QUIC, including ticker symbols
    const map<string, pair<string, vector<string>>> dataSources = {
        {"Yahoo Finance", {"wss://streamer.finance.yahoo.com", {"AAPL"}}},
        {"Binance", {"wss://stream.binance.com", {"btcusdt@trade"}}},
        {"Coinbase", {"wss://ws-feed.exchange.coinbase.com", {"BTC-USD"}}},
        {"Polygon", {"wss://delayed.polygon.io", {"options"}}},
        {"Alternative Finance API", {"https://api.alternative-finance.com", {"SP500", "NASDAQ"}}},
        {"Crypto Sentiment", {"https://crypto-sentiment-data.com", {"BTC", "ETH"}}},
        {"Stock Market News", {"https://stock-news.com", {"TSLA", "MSFT"}}}
    };
}

// Allows direct access to dataSources without needing DataSources::dataSources
using namespace DataSources;

#endif // COMMON_H

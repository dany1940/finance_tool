#!/bin/bash
echo "Building C++ WebSocket Client using Clang..."
clang++ -std=c++17 -o client ../src/main.cpp ../src/websocket_client.cpp ../src/kafka_producer.cpp \
-I../include -lwebsocketpp -lboost_system -ljsoncpp -lrdkafka -pthread
echo "Build Complete!"

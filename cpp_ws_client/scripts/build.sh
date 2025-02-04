#!/bin/bash
echo "Building C++ WebSocket Client using Clang..."
clang++ -std=c++17 -o client ../src/main.cpp ../src/websocketClient.cpp ../src/kafkaProducer.cpp \
-I../include -lwebsocketpp -lboost_system -ljsoncpp -lrdkafka -pthread
echo "Build Complete!"

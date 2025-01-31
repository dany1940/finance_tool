# Compiler
CXX = clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -Wall -Wextra -pthread

# ✅ Fixed: Use Correct Boost and WebSocket++ Paths
INCLUDE_PATHS = -Iinclude \
                -I/usr/local/include \
                -I/usr/local/opt/boost/includefin \
                -I/usr/local/include/websocketpp \
				-I/usr/local/Cellar/websocketpp/0.8.2/include/websocketpp/ \
                -I/usr/local/opt/jsoncpp/include \
                -I/usr/local/opt/librdkafka/include \
				-I/usr/local/include/zmq \
				-I/usr/local/Cellar/zeromq/4.3.2/include \
				-I/usr/local/Cellar/librdkafka/2.8.0/include/librdkafka \
				-I//Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/c++/v1/iostream

LIBRARY_PATHS = -L/usr/local/lib \
                -L/usr/local/opt/boost/lib \
                -L/usr/local/opt/jsoncpp/lib \
                -L/usr/local/opt/librdkafka/lib

LIBS = -lboost_system -lboost_thread -ljsoncpp -lrdkafka -lpthread -lzmq

# Source Files
SRCS = src/main.cpp src/websocket_client.cpp src/zmq_handler.cpp src/zmq_kafka_producer.cpp
OBJS = $(SRCS:.cpp=.o)

# Output Binary
TARGET = client

# Default Build Target
all: $(TARGET)

# Compile Executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) $(LIBRARY_PATHS) $(OBJS) $(LIBS) -o $(TARGET)

# Compile Object Files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Clean Build Files
clean:
	rm -f $(OBJS) $(TARGET)

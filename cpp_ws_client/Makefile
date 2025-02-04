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
				-I/usr/local/opt/spdlog/include \
				-I/usr/local/include/zmq \
				-I/usr/local/Cellar/openssl@3/3.4.0/include \
				-I/usr/local/Cellar/zeromq/4.3.2/include \
				-I/usr/local/Cellar/librdkafka/2.8.0/include/librdkafka \
				-I//usr/local/opt/fmt/include \
				-I//Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk/usr/include/c++/v1/iostream

LIBRARY_PATHS = -L/usr/local/lib \
                -L/usr/local/opt/boost/lib \
                -L/usr/local/opt/jsoncpp/lib \
                -L/usr/local/opt/librdkafka/lib \
				-L/usr/local/opt/spdlog/lib \
				-L/usr/local/Cellar/openssl@3/3.4.0/lib

LIBS = -lssl -lcrypto -lboost_system -lboost_thread -ljsoncpp -lrdkafka -lpthread -lzmq -lspdlog -lfmt

# Source Files
# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source and Object Files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Output Binary
TARGET = $(BIN_DIR)/client

# Default Build Target
all: $(TARGET)

# Compile Executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) $(LIBRARY_PATHS) $(OBJS) $(LIBS) -o $(TARGET)

# Compile Object Files (Now in `build/`)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Clean Build Files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Start Zookeeper
start-zookeeper:
	@echo "Starting Zookeeper..."
	@nohup zookeeper-server-start /usr/local/etc/zookeeper/zoo.cfg &

# Stop Zookeeper
stop-zookeeper:
	@echo "Stopping Zookeeper..."
	@pkill -f zookeeper

# Start Kafka
start-kafka:
	@echo "Starting Kafka..."
	@nohup kafka-server-start /usr/local/etc/kafka/server.properties &

# Stop Kafka
stop-kafka:
	@echo "Stopping Kafka..."
	@pkill -f kafka

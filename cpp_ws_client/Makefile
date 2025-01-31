# Compiler
CXX = clang++
CXXFLAGS = -std=c++17 -stdlib=libc++ -Wall -Wextra -pthread

# Include & Library Paths (macOS Homebrew Paths)
INCLUDE_PATHS = -Iinclude \
                -I/opt/homebrew/include \
                -I/opt/homebrew/opt/boost/include \
                -I/opt/homebrew/opt/jsoncpp/include \
                -I/opt/homebrew/opt/websocketpp/include

LIBRARY_PATHS = -L/opt/homebrew/lib \
                -L/opt/homebrew/opt/boost/lib \
                -L/opt/homebrew/opt/jsoncpp/lib \
                -L/opt/homebrew/opt/librdkafka/lib

LIBS = -lboost_system -ljsoncpp -lrdkafka -lpthread

# Source Files
SRCS = src/main.cpp src/websocket_client.cpp src/kafka_producer.cpp
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

# Compiler
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -pthread

# Include Paths
INCLUDE_PATHS = -Iinclude \
                -I/Users/FlorinDumitrascu/ngtcp2/include/    \
                -I/Users/FlorinDumitrascu/nghttp3/include/    \
                -I/Users/FlorinDumitrascu/wolfssl/include/     \
                -I/usr/local/include

# Library Paths
LIBRARY_PATHS = -L/Users/FlorinDumitrascu/ngtcp2/lib \
                -L/Users/FlorinDumitrascu/nghttp3/lib \
                -L/Users/FlorinDumitrascu/wolfssl/lib \
                -L/usr/local/lib

# Libraries to Link
LIBS = -lngtcp2 -lnghttp3 -lwolfssl -lpthread

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
CERTS_DIR = certificates

# Output Binary
TARGET = $(BIN_DIR)/quic_server

# Certificates
SERVER_KEY = $(CERTS_DIR)/server.key
SERVER_CERT = $(CERTS_DIR)/server.crt

# Source and Object Files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Default Build Target
all: $(TARGET)

# Certificates Generation (This will run before compiling the target)
$(SERVER_KEY) $(SERVER_CERT):
	@mkdir -p $(CERTS_DIR)
	openssl genpkey -algorithm RSA -out $(SERVER_KEY)
	openssl req -new -key $(SERVER_KEY) -out $(CERTS_DIR)/server.csr -subj "/CN=localhost"
	openssl x509 -req -in $(CERTS_DIR)/server.csr -signkey $(SERVER_KEY) -out $(SERVER_CERT) -days 365

# Compile Executable
$(TARGET): $(OBJS) $(SERVER_KEY) $(SERVER_CERT)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) $(LIBRARY_PATHS) $(OBJS) $(LIBS) -o $(TARGET)

# Compile Object Files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATHS) -c $< -o $@

# Clean Build Files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(CERTS_DIR)

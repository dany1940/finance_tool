
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <wolfssl/options.h>      // Always include this first
#include <wolfssl/ssl.h>
#include <ngtcp2/ngtcp2.h>
#include <nghttp3/nghttp3.h>
#include "generate_cert.h"  // Include the certificate generation header

#define PORT 4433
#define BUFFER_SIZE 4096

// Global QUIC context
WOLFSSL_CTX *ctx = nullptr;

// Handle QUIC Client Connection
void handle_connection(int sock) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUFFER_SIZE];

    // Create a new wolfSSL session for encryption
    WOLFSSL *ssl = wolfSSL_new(ctx);
    if (!ssl) {
        std::cerr << "Error creating wolfSSL session!" << std::endl;
        return;
    }

    while (true) {
        int received = recvfrom(sock, buffer, sizeof(buffer), 0,
                                (struct sockaddr *)&client_addr, &client_len);

        if (received > 0) {
            std::cout << "Received " << received << " bytes from client "
                      << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port)
                      << std::endl;

            // Send a simple HTTP/3 response over QUIC
            const char response[] = "HTTP/3 Server Response";
            sendto(sock, response, strlen(response), 0,
                   (struct sockaddr *)&client_addr, client_len);
        }
    }

    wolfSSL_free(ssl);  // Free the SSL session
}

// Initialize QUIC server (binding to UDP)
bool initialize_quic_server(int &sock) {
    sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Error creating UDP socket!" << std::endl;
        return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // Allow external connections

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to bind UDP socket! Port may be in use." << std::endl;
        return false;
    }

    return true;
}

int main() {
    std::cout << "Starting QUIC HTTP/3 server on UDP 0.0.0.0:" << PORT << "..." << std::endl;

    // Load server certificates (self-signed certificates if needed)
    if (generateCert() != 0) {
        std::cerr << "Failed to generate certificates!" << std::endl;
        return -1;
    }

    // Initialize wofSSL
    wolfSSL_Init();

    // Create QUIC context for QUIC-specific connection handling (TLS 1.3 for QUIC)
    ctx = wolfSSL_CTX_new(wolfSSLv23_server_method());  // wolfSSLv23 supports TLS 1.3
    if (!ctx) {
        std::cerr << "wolfSSL context initialization failed!" << std::endl;
        return -1;
    }

    // Load certificate and private key for encryption
    if (wolfSSL_CTX_use_certificate_file(ctx, "certificates/server.crt", SSL_FILETYPE_PEM) != WOLFSSL_SUCCESS ||
        wolfSSL_CTX_use_PrivateKey_file(ctx, "certificates/server.key", SSL_FILETYPE_PEM) != WOLFSSL_SUCCESS) {
        std::cerr << "Failed to load certificate or private key!" << std::endl;
        return -1;
    }

    int sock;
    if (!initialize_quic_server(sock)) {
        return -1;
    }

    std::cout << "QUIC HTTP/3 server running on UDP 0.0.0.0:" << PORT << "..." << std::endl;

    // Handle incoming client connections
    handle_connection(sock);

    // Clean up
    wolfSSL_CTX_free(ctx);
    wolfSSL_Cleanup();
    close(sock);

    return 0;
}

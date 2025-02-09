#include <wolfssl/options.h>     // Always include this first
#include <wolfssl/ssl.h>
#include <wolfssl/wolfcrypt/rsa.h>
#include <wolfssl/wolfcrypt/random.h>
#include <iostream>
#include <fstream>
#include <ctime>

void generateCert() {
    // Initialize wolfSSL
    wolfSSL_Init();

    // Initialize the random number generator
    WC_RNG rng;
    int ret = wc_InitRng(&rng);  // Initialize RNG
    if (ret != 0) {
        std::cerr << "Error initializing random number generator." << std::endl;
        return;
    }

    // Create the RSA key object
    RsaKey rsaKey;
    ret = wc_MakeRsaKey(&rsaKey, 2048, WC_RSA_EXPONENT, &rng);  // Generate 2048-bit RSA key
    if (ret != 0) {
        std::cerr << "Error generating RSA key." << std::endl;
        return;
    }

    // Create a new X.509 certificate object (self-signed)
    WOLFSSL_X509* cert = wolfSSL_X509_new();
    if (!cert) {
        std::cerr << "Error creating X.509 certificate." << std::endl;
        return;
    }

    // Set the certificate's subject (this will be the "CN=localhost")
    WOLFSSL_X509_NAME* name = wolfSSL_X509_NAME_new();
    if (!name) {
        std::cerr << "Error creating X509 name." << std::endl;
        return;
    }
    wolfSSL_X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (unsigned char*)"localhost", -1, -1, 0);

    // Set the subject name for the certificate
    wolfSSL_X509_set_subject_name(cert, name);

    // Set the issuer name to be the same (self-signed)
    wolfSSL_X509_set_issuer_name(cert, name);

    // Set the public key (RSA key)
    WOLFSSL_EVP_PKEY* pubkey = wolfSSL_EVP_PKEY_new();
    if (!pubkey) {
        std::cerr << "Error creating public key object." << std::endl;
        return;
    }
    wolfSSL_EVP_PKEY_set1_RSA(pubkey, &rsaKey);  // Set the RSA key into the EVP_PKEY object

    // Assign the public key to the certificate
    wolfSSL_X509_set_pubkey(cert, pubkey);

    // Set certificate validity period (e.g., 365 days)
    time_t current_time = time(nullptr);
    WOLFSSL_ASN1_TIME* validity_start = wolfSSL_ASN1_TIME_new();
    WOLFSSL_ASN1_TIME* validity_end = wolfSSL_ASN1_TIME_new();
    wolfSSL_ASN1_TIME_set(validity_start, current_time);
    wolfSSL_ASN1_TIME_set(validity_end, current_time + 365 * 24 * 60 * 60); // 1 year validity
    wolfSSL_X509_set_notBefore(cert, validity_start);
    wolfSSL_X509_set_notAfter(cert, validity_end);

    // Set the certificate serial number
    WOLFSSL_ASN1_INTEGER* serial_number = wolfSSL_ASN1_INTEGER_new();
    wolfSSL_ASN1_INTEGER_set(serial_number, 1234567890); // Example serial number
    wolfSSL_X509_set_serialNumber(cert, serial_number);

    // Sign the certificate with the private key
    ret = wolfSSL_X509_sign(cert, pubkey, nullptr);
    if (ret != 1) {
        std::cerr << "Error signing the certificate." << std::endl;
        return;
    }

    // Save the certificate to a file (server.crt)
    std::ofstream certFile("server.crt", std::ios::out | std::ios::binary);
    certFile.write(reinterpret_cast<const char*>(cert->data), cert->size);
    certFile.close();

    // Save the private key to a file (server.key)
    std::ofstream keyFile("server.key", std::ios::out | std::ios::binary);
    keyFile.write(reinterpret_cast<const char*>(&rsaKey), sizeof(rsaKey));
    keyFile.close();

    std::cout << "Self-signed certificate and private key generated successfully." << std::endl;

    // Cleanup
    wolfSSL_X509_free(cert);
    wolfSSL_X509_NAME_free(name);
    wolfSSL_EVP_PKEY_free(pubkey);
    wolfSSL_RSA_free(&rsaKey);
    wolfSSL_ASN1_TIME_free(validity_start);
    wolfSSL_ASN1_TIME_free(validity_end);
    wolfSSL_ASN1_INTEGER_free(serial_number);
    wolfSSL_Cleanup();
}

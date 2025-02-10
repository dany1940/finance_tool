#include <wolfssl/options.h>  // Always include this first
#include <wolfssl/ssl.h>
#include <wolfssl/wolfcrypt/rsa.h>
#include <wolfssl/wolfcrypt/asn.h>
#include <wolfssl/openssl/evp.h>
#include <iostream>
#include <fstream>
#include <ctime>

int generateCert() {
    std::cout << "[INFO] Initializing wolfSSL..." << std::endl;
    wolfSSL_Init();

    WC_RNG rng;
    if (wc_InitRng(&rng) != 0) {
        std::cerr << "[ERROR] RNG initialization failed!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] RNG initialized." << std::endl;

    // Generate RSA key
    RsaKey rsaKey;
    if (wc_MakeRsaKey(&rsaKey, 2048, WC_RSA_EXPONENT, &rng) != 0) {
        std::cerr << "[ERROR] RSA key generation failed!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] RSA key generated." << std::endl;

    // Convert RsaKey to WOLFSSL_RSA
    WOLFSSL_RSA* rsa = wolfSSL_RSA_new();
    if (!rsa) {
        std::cerr << "[ERROR] Failed to allocate WOLFSSL_RSA." << std::endl;
        return -1;
    }

    // Set RSA key components
    WOLFSSL_BIGNUM *n = wolfSSL_BN_new();
    WOLFSSL_BIGNUM *e = wolfSSL_BN_new();
    WOLFSSL_BIGNUM *d = wolfSSL_BN_new();

    wolfSSL_BN_bin2bn((const unsigned char*)&rsaKey.n, sizeof(rsaKey.n), n);
    wolfSSL_BN_bin2bn((const unsigned char*)&rsaKey.e, sizeof(rsaKey.e), e);
    wolfSSL_BN_bin2bn((const unsigned char*)&rsaKey.d, sizeof(rsaKey.d), d);

    if (wolfSSL_RSA_set0_key(rsa, n, e, d) != 1) {
        std::cerr << "[ERROR] Failed to set RSA key components!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] RSA key components set." << std::endl;

    // Assign RSA key to EVP_PKEY
    WOLFSSL_EVP_PKEY* pkey = wolfSSL_EVP_PKEY_new();
    if (!pkey) {
        std::cerr << "[ERROR] Failed to create EVP_PKEY." << std::endl;
        return -1;
    }

    if (wolfSSL_EVP_PKEY_assign_RSA(pkey, rsa) != 1) {
        std::cerr << "[ERROR] Failed to assign RSA key to EVP_PKEY!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] RSA key assigned to EVP_PKEY." << std::endl;

    // Create X.509 certificate
    WOLFSSL_X509* cert = wolfSSL_X509_new();
    if (!cert) {
        std::cerr << "[ERROR] Failed to create X.509 certificate!" << std::endl;
        return -1;
    }
    std::cout << "[SUCCESS] X.509 certificate created." << std::endl;

    // Set subject and issuer
    WOLFSSL_X509_NAME* name = wolfSSL_X509_NAME_new();
    wolfSSL_X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (unsigned char*)"localhost", -1, -1, 0);

    wolfSSL_X509_set_subject_name(cert, name);
    wolfSSL_X509_set_issuer_name(cert, name);
    std::cout << "[SUCCESS] X.509 subject and issuer set." << std::endl;

    // Set public key
    wolfSSL_X509_set_pubkey(cert, pkey);
    std::cout << "[SUCCESS] X.509 public key set." << std::endl;

    // Set validity period
    WOLFSSL_ASN1_TIME* start = wolfSSL_ASN1_TIME_new();
    WOLFSSL_ASN1_TIME* end = wolfSSL_ASN1_TIME_new();
    wolfSSL_X509_gmtime_adj(start, 0);
    wolfSSL_X509_gmtime_adj(end, 365 * 24 * 60 * 60);

    wolfSSL_X509_set_notBefore(cert, start);
    wolfSSL_X509_set_notAfter(cert, end);
    std::cout << "[SUCCESS] X.509 certificate validity set." << std::endl;

    // **Sign the certificate with SHA-512**
    const WOLFSSL_EVP_MD* digest = wolfSSL_EVP_sha512();
    if (!digest) {
        std::cerr << "[ERROR] SHA-512 digest not found!" << std::endl;
        return -1;
    }

    if (wolfSSL_X509_sign(cert, pkey, digest) != 1) {
        std::cerr << "[ERROR] Failed to sign the certificate with SHA-512!" << std::endl;

        // Print detailed error
        unsigned long err = wolfSSL_ERR_get_error();
        char err_msg[256];
        wolfSSL_ERR_error_string(err, err_msg);
        std::cerr << "[DEBUG] WolfSSL Error: " << err_msg << std::endl;

        return -1;
    }
    std::cout << "[SUCCESS] Certificate signed with SHA-512." << std::endl;

    // Save certificate
    int len = i2d_X509(cert, nullptr);
    unsigned char* der = (unsigned char*)malloc(len);
    unsigned char* p = der;
    i2d_X509(cert, &p);

    std::ofstream certFile("server.crt", std::ios::out | std::ios::binary);
    certFile.write(reinterpret_cast<const char*>(der), len);
    certFile.close();
    free(der);
    std::cout << "[SUCCESS] Certificate saved to 'server.crt'." << std::endl;

    // Save private key
    std::ofstream keyFile("server.key", std::ios::out | std::ios::binary);
    keyFile.write(reinterpret_cast<const char*>(&rsaKey), sizeof(rsaKey));
    keyFile.close();
    std::cout << "[SUCCESS] Private key saved to 'server.key'." << std::endl;

    // Cleanup
    wolfSSL_X509_free(cert);
    wolfSSL_X509_NAME_free(name);
    wolfSSL_EVP_PKEY_free(pkey);
    wolfSSL_RSA_free(rsa);
    wolfSSL_BN_free(n);
    wolfSSL_BN_free(e);
    wolfSSL_BN_free(d);
    wolfSSL_ASN1_TIME_free(start);
    wolfSSL_ASN1_TIME_free(end);
    wolfSSL_Cleanup();

    return 0;
}

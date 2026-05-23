#ifndef PUTER_XLA_BRIDGE_CORE_H
#define PUTER_XLA_BRIDGE_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XLAStatus {
    int code;
    char message[512];
} XLAStatus;

typedef void* XLAClientRef;
typedef void* XLABufferRef;
typedef void* XLAExecutableRef;

typedef struct XLABuffer {
    void* clientContext;
    void* pjrtBuffer;
} XLABuffer;

void xla_status_set(XLAStatus* status, int code, const char* message);

int xla_open_client(XLAClientRef* outClient, XLAStatus* status);
void xla_close_client(XLAClientRef client);

long long xla_client_device_memory_bytes(XLAClientRef client);

XLABufferRef xla_buffer_from_host(
    XLAClientRef client,
    const void* hostData,
    long long byteCount,
    int elementType,
    const long long* dimensions,
    int rank,
    XLAStatus* status
);

int xla_buffer_to_host(
    XLAClientRef client,
    XLABufferRef buffer,
    void* hostData,
    long long byteCount,
    XLAStatus* status
);

void xla_buffer_release(XLABufferRef buffer);

XLAExecutableRef xla_compile_hlo(
    XLAClientRef client,
    const char* hloText,
    XLAStatus* status
);

void xla_executable_release(XLAExecutableRef executable);

int xla_execute_unary(
    XLAClientRef client,
    XLAExecutableRef executable,
    XLABufferRef input,
    XLABufferRef output,
    XLAStatus* status
);

int xla_execute_binary(
    XLAClientRef client,
    XLAExecutableRef executable,
    XLABufferRef left,
    XLABufferRef right,
    XLABufferRef output,
    XLAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif

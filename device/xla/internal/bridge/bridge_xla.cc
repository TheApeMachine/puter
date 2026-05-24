#include "core.h"
#include "pjrt_c_api.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef const PJRT_Api* (*GetPjrtApiFn)(void);

typedef struct XLAContext {
    void* pluginHandle;
    const PJRT_Api* api;
    PJRT_Client* client;
    PJRT_Device* defaultDevice;
    long long deviceMemoryBytes;
} XLAContext;

static XLAContext* xla_context_from_ref(XLAClientRef clientRef) {
    return (XLAContext*)clientRef;
}

void xla_status_set(XLAStatus* status, int code, const char* message) {
    if (status == NULL) {
        return;
    }

    status->code = code;

    if (message == NULL) {
        status->message[0] = '\0';
        return;
    }

    snprintf(status->message, sizeof(status->message), "%s", message);
}

static void xla_set_pjrt_error(const PJRT_Api* api, PJRT_Error* error, XLAStatus* status) {
    if (error == NULL) {
        return;
    }

    PJRT_Error_Message_Args messageArgs;
    messageArgs.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    messageArgs.extension_start = NULL;
    messageArgs.error = error;
    messageArgs.message = NULL;
    messageArgs.message_size = 0;

    if (api->PJRT_Error_Message(&messageArgs), messageArgs.message != NULL) {
        xla_status_set(status, -1, messageArgs.message);
    } else {
        xla_status_set(status, -1, "PJRT error");
    }

    PJRT_Error_Destroy_Args destroyArgs;
    destroyArgs.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroyArgs.extension_start = NULL;
    destroyArgs.error = error;
    api->PJRT_Error_Destroy(&destroyArgs);
}

static const char* xla_plugin_path(void) {
    const char* fromEnv = getenv("PJRT_PLUGIN_PATH");

    if (fromEnv != NULL && fromEnv[0] != '\0') {
        return fromEnv;
    }

    fromEnv = getenv("XLA_PJRT_PLUGIN");

    if (fromEnv != NULL && fromEnv[0] != '\0') {
        return fromEnv;
    }

    return "libpjrt_c_api_gpu.so";
}

static int xla_map_element_type(int elementType, PJRT_Buffer_Type* outType) {
    switch (elementType) {
    case 1:
        *outType = PJRT_Buffer_Type_F64;
        return 0;
    case 2:
        *outType = PJRT_Buffer_Type_F32;
        return 0;
    case 3:
        *outType = PJRT_Buffer_Type_F16;
        return 0;
    case 4:
        *outType = PJRT_Buffer_Type_BF16;
        return 0;
    case 5:
        *outType = PJRT_Buffer_Type_F8E4M3;
        return 0;
    case 6:
        *outType = PJRT_Buffer_Type_F8E5M2;
        return 0;
    case 7:
        *outType = PJRT_Buffer_Type_S64;
        return 0;
    case 8:
        *outType = PJRT_Buffer_Type_S32;
        return 0;
    case 9:
        *outType = PJRT_Buffer_Type_S16;
        return 0;
    case 10:
        *outType = PJRT_Buffer_Type_S8;
        return 0;
    case 11:
        *outType = PJRT_Buffer_Type_U64;
        return 0;
    case 12:
        *outType = PJRT_Buffer_Type_U32;
        return 0;
    case 13:
        *outType = PJRT_Buffer_Type_U16;
        return 0;
    case 14:
        *outType = PJRT_Buffer_Type_U8;
        return 0;
    case 15:
        *outType = PJRT_Buffer_Type_PRED;
        return 0;
    default:
        return -1;
    }
}

static int xla_select_default_device(const PJRT_Api* api, PJRT_Client* client, PJRT_Device** outDevice, XLAStatus* status) {
    PJRT_Client_AddressableDevices_Args devicesArgs;
    devicesArgs.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    devicesArgs.extension_start = NULL;
    devicesArgs.client = client;
    devicesArgs.addressable_devices = NULL;
    devicesArgs.num_addressable_devices = 0;

    PJRT_Error* error = api->PJRT_Client_AddressableDevices(&devicesArgs);

    if (error != NULL) {
        xla_set_pjrt_error(api, error, status);
        return -1;
    }

    if (devicesArgs.num_addressable_devices == 0 || devicesArgs.addressable_devices == NULL) {
        xla_status_set(status, -1, "PJRT client has no addressable devices");
        return -1;
    }

    *outDevice = devicesArgs.addressable_devices[0];
    return 0;
}

static int xla_query_device_memory(const PJRT_Api* api, PJRT_Device* device, long long* outBytes, XLAStatus* status) {
    PJRT_Device_MemoryStats_Args memoryArgs;
    memoryArgs.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    memoryArgs.extension_start = NULL;
    memoryArgs.device = device;
    memoryArgs.bytes_in_use = 0;
    memoryArgs.peak_bytes_in_use = 0;
    memoryArgs.bytes_limit = 0;
    memoryArgs.bytes_limit_is_set = false;
    memoryArgs.peak_bytes_in_use_is_set = false;
    memoryArgs.largest_free_block_bytes = 0;
    memoryArgs.largest_free_block_bytes_is_set = false;
    memoryArgs.pool_bytes = 0;
    memoryArgs.pool_bytes_is_set = false;
    memoryArgs.peak_pool_bytes = 0;
    memoryArgs.peak_pool_bytes_is_set = false;

    PJRT_Error* error = api->PJRT_Device_MemoryStats(&memoryArgs);

    if (error != NULL) {
        xla_set_pjrt_error(api, error, status);
        return -1;
    }

    if (memoryArgs.bytes_limit_is_set) {
        *outBytes = (long long)memoryArgs.bytes_limit;
        return 0;
    }

    *outBytes = 0;
    return 0;
}

int xla_open_client(XLAClientRef* outClient, XLAStatus* status) {
    if (outClient == NULL) {
        xla_status_set(status, -1, "null outClient");
        return -1;
    }

    const char* pluginPath = xla_plugin_path();
    void* pluginHandle = dlopen(pluginPath, RTLD_NOW | RTLD_LOCAL);

    if (pluginHandle == NULL) {
        xla_status_set(status, -1, dlerror());
        return -1;
    }

    GetPjrtApiFn getPjrtApi = (GetPjrtApiFn)dlsym(pluginHandle, "GetPjrtApi");

    if (getPjrtApi == NULL) {
        xla_status_set(status, -1, "GetPjrtApi symbol missing from PJRT plugin");
        dlclose(pluginHandle);
        return -1;
    }

    const PJRT_Api* api = getPjrtApi();

    if (api == NULL) {
        xla_status_set(status, -1, "GetPjrtApi returned null");
        dlclose(pluginHandle);
        return -1;
    }

    PJRT_Plugin_Initialize_Args initArgs;
    initArgs.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    initArgs.extension_start = NULL;

    PJRT_Error* initError = api->PJRT_Plugin_Initialize(&initArgs);

    if (initError != NULL) {
        xla_set_pjrt_error(api, initError, status);
        dlclose(pluginHandle);
        return -1;
    }

    PJRT_Client_Create_Args createArgs;
    createArgs.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    createArgs.extension_start = NULL;
    createArgs.create_options = NULL;
    createArgs.num_options = 0;
    createArgs.kv_get_callback = NULL;
    createArgs.kv_get_user_arg = NULL;
    createArgs.kv_put_callback = NULL;
    createArgs.kv_put_user_arg = NULL;
    createArgs.client = NULL;
    createArgs.kv_try_get_callback = NULL;
    createArgs.kv_try_get_user_arg = NULL;

    PJRT_Error* createError = api->PJRT_Client_Create(&createArgs);

    if (createError != NULL) {
        xla_set_pjrt_error(api, createError, status);
        dlclose(pluginHandle);
        return -1;
    }

    XLAContext* context = (XLAContext*)calloc(1, sizeof(XLAContext));
    context->pluginHandle = pluginHandle;
    context->api = api;
    context->client = createArgs.client;

    if (xla_select_default_device(api, context->client, &context->defaultDevice, status) != 0) {
        PJRT_Client_Destroy_Args destroyArgs;
        destroyArgs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroyArgs.extension_start = NULL;
        destroyArgs.client = context->client;
        api->PJRT_Client_Destroy(&destroyArgs);
        dlclose(pluginHandle);
        free(context);
        return -1;
    }

    if (xla_query_device_memory(api, context->defaultDevice, &context->deviceMemoryBytes, status) != 0) {
        context->deviceMemoryBytes = 0;
    }

    *outClient = (XLAClientRef)context;
    return 0;
}

void xla_close_client(XLAClientRef clientRef) {
    XLAContext* context = xla_context_from_ref(clientRef);

    if (context == NULL) {
        return;
    }

    if (context->client != NULL && context->api != NULL) {
        PJRT_Client_Destroy_Args destroyArgs;
        destroyArgs.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroyArgs.extension_start = NULL;
        destroyArgs.client = context->client;
        context->api->PJRT_Client_Destroy(&destroyArgs);
        context->client = NULL;
    }

    if (context->pluginHandle != NULL) {
        dlclose(context->pluginHandle);
        context->pluginHandle = NULL;
    }

    free(context);
}

long long xla_client_device_memory_bytes(XLAClientRef clientRef) {
    XLAContext* context = xla_context_from_ref(clientRef);

    if (context == NULL) {
        return 0;
    }

    return context->deviceMemoryBytes;
}

XLABufferRef xla_buffer_from_host(
    XLAClientRef clientRef,
    const void* hostData,
    long long byteCount,
    int elementType,
    const long long* dimensions,
    int rank,
    XLAStatus* status
) {
    XLAContext* context = xla_context_from_ref(clientRef);

    if (context == NULL || context->api == NULL || context->client == NULL) {
        xla_status_set(status, -1, "invalid XLA client");
        return NULL;
    }

    PJRT_Buffer_Type bufferType = PJRT_Buffer_Type_INVALID;

    if (xla_map_element_type(elementType, &bufferType) != 0) {
        xla_status_set(status, -1, "unsupported XLA element type");
        return NULL;
    }

    PJRT_Client_BufferFromHostBuffer_Args bufferArgs;
    bufferArgs.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    bufferArgs.extension_start = NULL;
    bufferArgs.client = context->client;
    bufferArgs.data = hostData;
    bufferArgs.type = bufferType;
    bufferArgs.dims = dimensions;
    bufferArgs.num_dims = (size_t)rank;
    bufferArgs.byte_strides = NULL;
    bufferArgs.num_byte_strides = 0;
    bufferArgs.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    bufferArgs.device = context->defaultDevice;
    bufferArgs.memory = NULL;
    bufferArgs.device_layout = NULL;
    bufferArgs.done_with_host_buffer = NULL;
    bufferArgs.buffer = NULL;

    PJRT_Error* error = context->api->PJRT_Client_BufferFromHostBuffer(&bufferArgs);

    if (error != NULL) {
        xla_set_pjrt_error(context->api, error, status);
        return NULL;
    }

    if (bufferArgs.done_with_host_buffer != NULL) {
        PJRT_Event_Await_Args awaitArgs;
        awaitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        awaitArgs.extension_start = NULL;
        awaitArgs.event = bufferArgs.done_with_host_buffer;

        context->api->PJRT_Event_Await(&awaitArgs);

        PJRT_Event_Destroy_Args eventDestroyArgs;
        eventDestroyArgs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eventDestroyArgs.extension_start = NULL;
        eventDestroyArgs.event = bufferArgs.done_with_host_buffer;
        context->api->PJRT_Event_Destroy(&eventDestroyArgs);
    }

    XLABuffer* wrapper = (XLABuffer*)calloc(1, sizeof(XLABuffer));
    wrapper->clientContext = context;
    wrapper->pjrtBuffer = bufferArgs.buffer;
    return (XLABufferRef)wrapper;
}

int xla_buffer_to_host(
    XLAClientRef clientRef,
    XLABufferRef bufferRef,
    void* hostData,
    long long byteCount,
    XLAStatus* status
) {
    XLAContext* context = xla_context_from_ref(clientRef);
    XLABuffer* wrapper = (XLABuffer*)bufferRef;
    PJRT_Buffer* buffer = wrapper == NULL ? NULL : (PJRT_Buffer*)wrapper->pjrtBuffer;

    if (context == NULL || context->api == NULL || buffer == NULL || hostData == NULL) {
        xla_status_set(status, -1, "invalid XLA buffer download");
        return -1;
    }

    PJRT_Buffer_ToHostBuffer_Args toHostArgs;
    toHostArgs.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    toHostArgs.extension_start = NULL;
    toHostArgs.src = buffer;
    toHostArgs.host_layout = NULL;
    toHostArgs.dst = hostData;
    toHostArgs.dst_size = (size_t)byteCount;
    toHostArgs.event = NULL;

    PJRT_Error* error = context->api->PJRT_Buffer_ToHostBuffer(&toHostArgs);

    if (error != NULL) {
        xla_set_pjrt_error(context->api, error, status);
        return -1;
    }

    if (toHostArgs.event != NULL) {
        PJRT_Event_Await_Args awaitArgs;
        awaitArgs.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        awaitArgs.extension_start = NULL;
        awaitArgs.event = toHostArgs.event;
        context->api->PJRT_Event_Await(&awaitArgs);

        PJRT_Event_Destroy_Args eventDestroyArgs;
        eventDestroyArgs.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        eventDestroyArgs.extension_start = NULL;
        eventDestroyArgs.event = toHostArgs.event;
        context->api->PJRT_Event_Destroy(&eventDestroyArgs);
    }

    return 0;
}

void xla_buffer_release(XLABufferRef bufferRef) {
    XLABuffer* wrapper = (XLABuffer*)bufferRef;

    if (wrapper == NULL) {
        return;
    }

    XLAContext* context = (XLAContext*)wrapper->clientContext;
    PJRT_Buffer* buffer = (PJRT_Buffer*)wrapper->pjrtBuffer;

    if (context != NULL && context->api != NULL && buffer != NULL) {
        PJRT_Buffer_Delete_Args deleteArgs;
        deleteArgs.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
        deleteArgs.extension_start = NULL;
        deleteArgs.buffer = buffer;
        context->api->PJRT_Buffer_Delete(&deleteArgs);
    }

    free(wrapper);
}

XLAExecutableRef xla_compile_hlo(
    XLAClientRef clientRef,
    const char* hloText,
    XLAStatus* status
) {
    XLAContext* context = xla_context_from_ref(clientRef);

    if (context == NULL || context->api == NULL || context->client == NULL || hloText == NULL) {
        xla_status_set(status, -1, "invalid XLA compile request");
        return NULL;
    }

    PJRT_Program program;
    program.struct_size = PJRT_Program_STRUCT_SIZE;
    program.extension_start = NULL;
    program.code = hloText;
    program.code_size = strlen(hloText);
    program.format = "hlo";
    program.format_size = 3;

    PJRT_Client_Compile_Args compileArgs;
    compileArgs.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    compileArgs.extension_start = NULL;
    compileArgs.client = context->client;
    compileArgs.program = &program;
    compileArgs.compile_options = NULL;
    compileArgs.compile_options_size = 0;
    compileArgs.executable = NULL;

    PJRT_Error* error = context->api->PJRT_Client_Compile(&compileArgs);

    if (error != NULL) {
        xla_set_pjrt_error(context->api, error, status);
        return NULL;
    }

    typedef struct XLAExecutable {
        XLAContext* context;
        PJRT_LoadedExecutable* executable;
    } XLAExecutable;

    XLAExecutable* loaded = (XLAExecutable*)calloc(1, sizeof(XLAExecutable));
    loaded->context = context;
    loaded->executable = compileArgs.executable;
    return (XLAExecutableRef)loaded;
}

void xla_executable_release(XLAExecutableRef executableRef) {
    typedef struct XLAExecutable {
        XLAContext* context;
        PJRT_LoadedExecutable* executable;
    } XLAExecutable;

    XLAExecutable* loaded = (XLAExecutable*)executableRef;

    if (loaded == NULL) {
        return;
    }

    if (loaded->context != NULL && loaded->context->api != NULL && loaded->executable != NULL) {
        PJRT_LoadedExecutable_Destroy_Args destroyArgs;
        destroyArgs.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroyArgs.extension_start = NULL;
        destroyArgs.executable = loaded->executable;
        loaded->context->api->PJRT_LoadedExecutable_Destroy(&destroyArgs);
    }

    free(loaded);
}

static int xla_execute_impl(
    XLAClientRef clientRef,
    XLAExecutableRef executableRef,
    PJRT_Buffer* const* inputs,
    size_t numInputs,
    PJRT_Buffer** outputs,
    XLAStatus* status
) {
    typedef struct XLAExecutable {
        XLAContext* context;
        PJRT_LoadedExecutable* executable;
    } XLAExecutable;

    XLAExecutable* loaded = (XLAExecutable*)executableRef;
    XLAContext* context = xla_context_from_ref(clientRef);

    if (loaded == NULL || context == NULL || context->api == NULL || outputs == NULL) {
        xla_status_set(status, -1, "invalid XLA execute request");
        return -1;
    }

    if (numInputs > 0 && inputs == NULL) {
        xla_status_set(status, -1, "invalid XLA execute request");
        return -1;
    }

    PJRT_ExecuteOptions executeOptions;
    memset(&executeOptions, 0, sizeof(executeOptions));
    executeOptions.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    executeOptions.extension_start = NULL;

    PJRT_Buffer* const* argumentList = inputs;
    PJRT_Buffer** outputList = outputs;
    PJRT_Buffer* const* const argumentLists[] = {argumentList};
    PJRT_Buffer** outputLists[] = {outputList};

    PJRT_LoadedExecutable_Execute_Args executeArgs;
    executeArgs.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    executeArgs.extension_start = NULL;
    executeArgs.executable = loaded->executable;
    executeArgs.options = &executeOptions;
    executeArgs.argument_lists = argumentLists;
    executeArgs.num_devices = 1;
    executeArgs.num_args = numInputs;
    executeArgs.output_lists = outputLists;
    executeArgs.device_complete_events = NULL;
    executeArgs.execute_device = context->defaultDevice;

    PJRT_Error* error = context->api->PJRT_LoadedExecutable_Execute(&executeArgs);

    if (error != NULL) {
        xla_set_pjrt_error(context->api, error, status);
        return -1;
    }

    return 0;
}

int xla_execute_unary(
    XLAClientRef clientRef,
    XLAExecutableRef executableRef,
    XLABufferRef input,
    XLABufferRef output,
    XLAStatus* status
) {
    XLABuffer* inputWrapper = (XLABuffer*)input;
    XLABuffer* outputWrapper = (XLABuffer*)output;
    PJRT_Buffer* inputs[] = {(PJRT_Buffer*)inputWrapper->pjrtBuffer};
    PJRT_Buffer* outputs[] = {(PJRT_Buffer*)outputWrapper->pjrtBuffer};
    return xla_execute_impl(clientRef, executableRef, inputs, 1, outputs, status);
}

int xla_execute_binary(
    XLAClientRef clientRef,
    XLAExecutableRef executableRef,
    XLABufferRef left,
    XLABufferRef right,
    XLABufferRef output,
    XLAStatus* status
) {
    XLABuffer* leftWrapper = (XLABuffer*)left;
    XLABuffer* rightWrapper = (XLABuffer*)right;
    XLABuffer* outputWrapper = (XLABuffer*)output;
    PJRT_Buffer* inputs[] = {
        (PJRT_Buffer*)leftWrapper->pjrtBuffer,
        (PJRT_Buffer*)rightWrapper->pjrtBuffer,
    };
    PJRT_Buffer* outputs[] = {(PJRT_Buffer*)outputWrapper->pjrtBuffer};
    return xla_execute_impl(clientRef, executableRef, inputs, 2, outputs, status);
}

int xla_execute_variadic(
    XLAClientRef clientRef,
    XLAExecutableRef executableRef,
    XLABufferRef* inputs,
    int inputCount,
    XLABufferRef output,
    XLAStatus* status
) {
    if (inputs == NULL || inputCount <= 0) {
        XLAStatus localStatus;
        xla_status_set(&localStatus, -1, "invalid XLA variadic execute request");
        if (status != NULL) {
            *status = localStatus;
        }
        return -1;
    }

    XLABuffer** inputWrappers = (XLABuffer**)inputs;
    PJRT_Buffer** pjrtInputs = (PJRT_Buffer**)calloc((size_t)inputCount, sizeof(PJRT_Buffer*));

    if (pjrtInputs == NULL) {
        XLAStatus localStatus;
        xla_status_set(&localStatus, -1, "XLA variadic execute allocation failed");
        if (status != NULL) {
            *status = localStatus;
        }
        return -1;
    }

    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
        pjrtInputs[inputIndex] = (PJRT_Buffer*)inputWrappers[inputIndex]->pjrtBuffer;
    }

    XLABuffer* outputWrapper = (XLABuffer*)output;
    PJRT_Buffer* outputs[] = {(PJRT_Buffer*)outputWrapper->pjrtBuffer};
    int result = xla_execute_impl(clientRef, executableRef, pjrtInputs, (size_t)inputCount, outputs, status);
    free(pjrtInputs);
    return result;
}

int xla_execute_nullary(
    XLAClientRef clientRef,
    XLAExecutableRef executableRef,
    XLABufferRef output,
    XLAStatus* status
) {
    XLABuffer* outputWrapper = (XLABuffer*)output;
    PJRT_Buffer* outputs[] = {(PJRT_Buffer*)outputWrapper->pjrtBuffer};
    return xla_execute_impl(clientRef, executableRef, NULL, 0, outputs, status);
}

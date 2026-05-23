#ifndef PUTER_XLA_PJRT_LOADER_H
#define PUTER_XLA_PJRT_LOADER_H

#include <stddef.h>
#include <stdint.h>

typedef struct PJRT_Api PJRT_Api;

typedef struct PJRT_Error PJRT_Error;
typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;

typedef enum PJRT_Buffer_Type {
    PJRT_Buffer_Type_INVALID = 0,
    PJRT_Buffer_Type_PRED = 1,
    PJRT_Buffer_Type_S8 = 2,
    PJRT_Buffer_Type_S16 = 3,
    PJRT_Buffer_Type_S32 = 4,
    PJRT_Buffer_Type_S64 = 5,
    PJRT_Buffer_Type_U8 = 6,
    PJRT_Buffer_Type_U16 = 7,
    PJRT_Buffer_Type_U32 = 8,
    PJRT_Buffer_Type_U64 = 9,
    PJRT_Buffer_Type_F16 = 10,
    PJRT_Buffer_Type_F32 = 11,
    PJRT_Buffer_Type_F64 = 12,
    PJRT_Buffer_Type_BF16 = 16,
    PJRT_Buffer_Type_F8E4M3 = 17,
    PJRT_Buffer_Type_F8E5M2 = 18,
} PJRT_Buffer_Type;

typedef struct PJRT_LoadedExecutable_Execute_Args {
    size_t struct_size;
    PJRT_LoadedExecutable* executable;
    PJRT_Buffer* const* argument_lists;
    size_t num_devices;
    size_t num_args;
    PJRT_Buffer** const* output_lists;
    PJRT_Event** device_complete_events;
    PJRT_Error* (*done_callback)(PJRT_LoadedExecutable_Execute_Args* args);
} PJRT_LoadedExecutable_Execute_Args;

typedef struct PJRT_Event PJRT_Event;

const PJRT_Api* puter_pjrt_load_api(void* pluginHandle);
void puter_pjrt_unload_api(void* pluginHandle);

#endif

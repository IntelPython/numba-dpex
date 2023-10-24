// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NRT_HELPER_H_
#define _NRT_HELPER_H_

#define NO_IMPORT_ARRAY
#include "_meminfo_helper.h"

void *NRT_MemInfo_external_allocator(NRT_MemInfo *mi);
void *NRT_MemInfo_data(NRT_MemInfo *mi);
void NRT_MemInfo_release(NRT_MemInfo *mi);
void NRT_MemInfo_call_dtor(NRT_MemInfo *mi);
void NRT_MemInfo_acquire(NRT_MemInfo *mi);
size_t NRT_MemInfo_size(NRT_MemInfo *mi);
void *NRT_MemInfo_parent(NRT_MemInfo *mi);
size_t NRT_MemInfo_refcount(NRT_MemInfo *mi);
void NRT_Free(void *ptr);
void NRT_dealloc(NRT_MemInfo *mi);
void NRT_MemInfo_destroy(NRT_MemInfo *mi);
void NRT_MemInfo_pyobject_dtor(void *data);

#endif /* _NRT_HELPER_H_ */

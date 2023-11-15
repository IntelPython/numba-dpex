// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the numba-dpex native representation for a dpctl.SyclEvent
///
//===----------------------------------------------------------------------===//

#ifndef _EVENTSTRUCT_H_
#define _EVENTSTRUCT_H_

#include "_nrt_helper.h"
#include "dpctl_sycl_interface.h"
#include "numba/core/runtime/nrt_external.h"
#include <Python.h>

typedef struct
{
    NRT_MemInfo *meminfo;
    PyObject *parent;
    void *event_ref;
} eventstruct_t;

void NRT_MemInfo_EventRef_Delete(void *data);

#endif /* _EVENTSTRUCT_H_ */

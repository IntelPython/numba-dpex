// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_eventstruct.h"
#include "_dbg_printer.h"

/*!
 * @brief A destructor that is called from NRT on object destruction. Deletes
 * dpctl event reference.
 *
 * @param    data           A dpctl event reference.
 * @return   {return}       Nothing.
 */
void NRT_MemInfo_EventRef_Delete(void *data)
{
    DPCTLSyclEventRef eref = data;

    DPCTLEvent_Delete(eref);

    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: deleting dpctl event reference.\n"););
}

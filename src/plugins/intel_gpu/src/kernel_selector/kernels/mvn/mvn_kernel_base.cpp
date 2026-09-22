// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_base.h"

#include <algorithm>
#include <vector>

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
// The kernels scale the input by the reciprocal of the deviation. In half precision
// that reciprocal becomes infinity for a small epsilon, so a zero variance gives a NaN.
float get_epsilon(const mvn_params& params) {
    if (params.epsilon <= 0.f) {
        return params.epsilon;
    }

    if (params.inputs[0].GetDType() != Datatype::F16 && params.outputs[0].GetDType() != Datatype::F16) {
        return params.epsilon;
    }

    constexpr float half_max = 65504.f;
    const float min_epsilon = params.mvnEpsMode == MVNEpsMode::INSIDE_SQRT ? 1.f / (half_max * half_max)
                                                                          : 1.f / half_max;

    return std::max(params.epsilon, min_epsilon);
}
}  // namespace

bool MVNKernelBase::Validate(const Params& params) const {
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    for (const auto& fused_op : orgParams.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op)) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }

    return true;
}

JitConstants MVNKernelBase::GetJitConstants(const mvn_params& params, MVNKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("EPSILON", get_epsilon(params)),
        MakeJitConstant(toString(params.mvnMode), ""),
        MakeJitConstant("NORMALIZE_VARIANCE", params.mvnNormalizeVariance),
        MakeJitConstant("EPS_" + toString(params.mvnEpsMode), ""),
    });

    return jit;
}

MVNKernelBase::DispatchData MVNKernelBase::SetDefault(const mvn_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    if (params.mvnMode == MVNMode::WITHIN_CHANNELS) {
        dispatchData.gws = {output.Batch().v, output.Feature().v, 1};
    } else {
        dispatchData.gws = {output.Batch().v, 1, 1};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

void MVNKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const mvn_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData MVNKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::MVN);

    if (!Validate(params)) {
        return {};
    }

    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<mvn_params>(params);

    auto finalKernelName = GetKernelName(orgParams);
    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params);
    auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     finalKernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    return {kd};
}

Datatype MVNKernelBase::GetActivationType(const mvn_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F16) {
        return Datatype::F16;
    }
    return Datatype::F32;
}

}  // namespace kernel_selector

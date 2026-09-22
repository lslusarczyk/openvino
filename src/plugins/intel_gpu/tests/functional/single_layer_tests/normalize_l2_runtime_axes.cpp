// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_constants.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"

#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

// The axes of NormalizeL2 can come at run time. The plugin needs them at build time,
// so the model must be rejected with an error and not read from a null constant.
TEST(smoke_NormalizeL2RuntimeAxes, compile_reports_unsupported) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3, 4});
    auto axes = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
    auto norm = std::make_shared<ov::op::v0::NormalizeL2>(data, axes, 1e-6f, ov::op::EpsMode::ADD);
    auto model = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(norm),
                                             ov::ParameterVector{data, axes},
                                             "normalize_l2_runtime_axes");

    ov::Core core;
    EXPECT_THROW(core.compile_model(model, ov::test::utils::DEVICE_GPU), ov::Exception);
}
}  // namespace

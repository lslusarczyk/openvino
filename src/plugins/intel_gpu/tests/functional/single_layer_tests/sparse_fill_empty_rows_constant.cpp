// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sparse_fill_empty_rows.hpp"

namespace {

// The plugin reads the inputs of SparseFillEmptyRows when they are constants.
// The order of the inputs is values, dense_shape, indices and default_value.
class SparseFillEmptyRowsConstantTest : public testing::WithParamInterface<size_t>,
                                       virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<size_t>& obj) {
        return "output=" + std::to_string(obj.param);
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        auto values = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{3}, {1.0f, 2.0f, 3.0f});
        auto dense_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 2});
        auto indices = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3, 2}, {0, 0, 0, 1, 3, 1});
        auto default_value = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});

        auto op = std::make_shared<ov::op::v16::SparseFillEmptyRows>(values, dense_shape, indices, default_value);

        // a parameter keeps the model from folding away
        auto dummy = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        function = std::make_shared<ov::Model>(
            ov::OutputVector{op->output(GetParam()), dummy->output(0)},
            ov::ParameterVector{dummy},
            "sparse_fill_empty_rows_constant");
    }
};

TEST_P(SparseFillEmptyRowsConstantTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRowsConstant,
                         SparseFillEmptyRowsConstantTest,
                         ::testing::Values(0, 1, 2),
                         SparseFillEmptyRowsConstantTest::getTestCaseName);
}  // namespace

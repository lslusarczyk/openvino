// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {

using MatMulConstantBatchParams = std::tuple<ov::Shape,  // first operand
                                             ov::Shape,  // second operand, a constant
                                             bool>;      // transpose the second operand

// The plugin turns the second operand of a MatMul into weights of a fully connected
// primitive. Weights that keep a batch must not be transposed by a change of layout only.
class MatMulConstantBatchTest : public testing::WithParamInterface<MatMulConstantBatchParams>,
                                virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulConstantBatchParams>& obj) {
        const auto& [shape_a, shape_b, transpose_b] = obj.param;

        std::ostringstream result;
        result << "a=" << shape_a << "_b=" << shape_b << "_transpose_b=" << transpose_b;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape_a, shape_b, transpose_b] = GetParam();

        // half precision picks the onednn gemm, which reads the weights of the permute
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
        abs_threshold = 0.05f;
        rel_threshold = 0.05f;

        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape_a);

        auto weights_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape_b);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(data, weights, false, transpose_b);

        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(matmul),
                                               ov::ParameterVector{data},
                                               "matmul_constant_batch");
    }
};

TEST_P(MatMulConstantBatchTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_MatMulConstantBatch,
                         MatMulConstantBatchTest,
                         ::testing::Values(MatMulConstantBatchParams{ov::Shape{3, 4}, ov::Shape{4, 3}, false},
                                           MatMulConstantBatchParams{ov::Shape{1, 3, 4}, ov::Shape{1, 4, 3}, false},
                                           MatMulConstantBatchParams{ov::Shape{1, 1, 3, 4}, ov::Shape{1, 1, 4, 3}, false},
                                           MatMulConstantBatchParams{ov::Shape{1, 3, 4}, ov::Shape{1, 3, 4}, true}),
                         MatMulConstantBatchTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/scatter_elements_update.hpp"

namespace {

enum class Rank1ConstantOp { GatherElements, ScatterElementsUpdate };

using Rank1ConstantInputParams = std::tuple<Rank1ConstantOp, ov::element::Type>;

// Indices of a rank 1 constant are laid out as feature by default, while the kernels
// of gather and scatter index the data by batch.
class Rank1ConstantInputTest : public testing::WithParamInterface<Rank1ConstantInputParams>,
                               virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Rank1ConstantInputParams>& obj) {
        const auto& [op_kind, input_precision] = obj.param;

        std::ostringstream result;
        result << "op=" << (op_kind == Rank1ConstantOp::GatherElements ? "GatherElements" : "ScatterElementsUpdate");
        result << "_input_precision=" << input_precision;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [op_kind, input_precision] = GetParam();

        auto data = std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape{6});
        auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {5, 3, 1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});

        ov::ParameterVector params{data};
        std::shared_ptr<ov::Node> op;
        if (op_kind == Rank1ConstantOp::GatherElements) {
            op = std::make_shared<ov::op::v6::GatherElements>(data, indices, 0);
        } else {
            auto updates = std::make_shared<ov::op::v0::Parameter>(input_precision, ov::Shape{3});
            params.push_back(updates);
            op = std::make_shared<ov::op::v12::ScatterElementsUpdate>(data, indices, updates, axis);
        }

        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(op), params, "rank1_constant");
    }
};

TEST_P(Rank1ConstantInputTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Rank1ConstantInput,
                         Rank1ConstantInputTest,
                         ::testing::Combine(::testing::Values(Rank1ConstantOp::GatherElements,
                                                              Rank1ConstantOp::ScatterElementsUpdate),
                                            ::testing::Values(ov::element::f32, ov::element::f16)),
                         Rank1ConstantInputTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/topk.hpp"

namespace {

enum class SecondOutputOp { MaxPool, TopK };

// The plugin writes the second output of these operations through a mutable_data node.
// A model that keeps only that output leaves the primitive without a first output.
class SecondOutputOnlyTest : public testing::WithParamInterface<SecondOutputOp>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SecondOutputOp>& obj) {
        return obj.param == SecondOutputOp::MaxPool ? "op=MaxPool" : "op=TopK";
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;

        std::shared_ptr<ov::Node> op;
        ov::ParameterVector params;
        if (GetParam() == SecondOutputOp::MaxPool) {
            auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 4, 5});
            params.push_back(data);
            op = std::make_shared<ov::op::v8::MaxPool>(data, ov::Strides{1, 1}, ov::Strides{1, 1},
                                                       ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{2, 3});
        } else {
            auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
            params.push_back(data);
            auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3});
            op = std::make_shared<ov::op::v11::TopK>(data, k, 1, ov::op::TopKMode::MAX,
                                                     ov::op::TopKSortType::SORT_VALUES, ov::element::i32);
        }

        function = std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(op->output(1)),
                                              params, "second_output_only");
    }
};

TEST_P(SecondOutputOnlyTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_SecondOutputOnly,
                         SecondOutputOnlyTest,
                         ::testing::Values(SecondOutputOp::MaxPool, SecondOutputOp::TopK),
                         SecondOutputOnlyTest::getTestCaseName);
}  // namespace

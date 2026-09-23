// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/topk.hpp"

namespace {

enum class SecondOutputOp { MaxPool, TopK, LSTMSequence };

const char* op_name(SecondOutputOp op_kind) {
    switch (op_kind) {
    case SecondOutputOp::MaxPool:
        return "MaxPool";
    case SecondOutputOp::TopK:
        return "TopK";
    default:
        return "LSTMSequence";
    }
}

// A model that keeps only the second output of these operations leaves the primitive
// without a first output, or reaches the first output through the port of the result.
class SecondOutputOnlyTest : public testing::WithParamInterface<SecondOutputOp>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SecondOutputOp>& obj) {
        return std::string("op=") + op_name(obj.param);
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration.insert(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));

        std::shared_ptr<ov::Node> op;
        ov::ParameterVector params;
        if (GetParam() == SecondOutputOp::MaxPool) {
            auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 4, 5});
            params.push_back(data);
            op = std::make_shared<ov::op::v8::MaxPool>(data, ov::Strides{1, 1}, ov::Strides{1, 1},
                                                       ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{2, 3});
        } else if (GetParam() == SecondOutputOp::TopK) {
            auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 8});
            params.push_back(data);
            auto k = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3});
            op = std::make_shared<ov::op::v11::TopK>(data, k, 1, ov::op::TopKMode::MAX,
                                                     ov::op::TopKSortType::SORT_VALUES, ov::element::i32);
        } else {
            const size_t batch = 1, seq_len = 3, input_size = 2, hidden_size = 2, num_dir = 1;
            auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch, seq_len, input_size});
            auto h0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch, num_dir, hidden_size});
            auto c0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch, num_dir, hidden_size});
            params = {x, h0, c0};

            auto lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{batch},
                                                        std::vector<int32_t>(batch, seq_len));
            auto w = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{num_dir, 4 * hidden_size, input_size},
                                                  std::vector<float>(num_dir * 4 * hidden_size * input_size, 0.1f));
            auto r = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{num_dir, 4 * hidden_size, hidden_size},
                                                  std::vector<float>(num_dir * 4 * hidden_size * hidden_size, 0.2f));
            auto b = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{num_dir, 4 * hidden_size},
                                                  std::vector<float>(num_dir * 4 * hidden_size, 0.05f));
            op = std::make_shared<ov::op::v5::LSTMSequence>(x, h0, c0, lengths, w, r, b, hidden_size,
                                                           ov::op::RecurrentSequenceDirection::FORWARD);
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
                         ::testing::Values(SecondOutputOp::MaxPool,
                                           SecondOutputOp::TopK,
                                           SecondOutputOp::LSTMSequence),
                         SecondOutputOnlyTest::getTestCaseName);
}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "activation_inst.h"
#include "reorder_inst.h"
#include "convolution_inst.h"
#include "pass_manager.h"
#include "to_string_utils.h"

#include "program_wrapper.h"
#include "program_helpers.h"

#include <memory>
#include <fully_connected_inst.h>

using namespace cldnn;
using namespace ::tests;

TEST(add_onednn_optimization_attributes, init_attribute_for_fused_onednn_primitive) {
    auto& engine = get_test_engine();

    auto in_layout = layout{ov::PartialShape({-1, 3, 112, 112}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 3, 112, 112}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({128, 3, 3, 3}), data_types::f16, format::bfyx});
    auto const1 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});
    auto const2 = engine.allocate_memory(layout{ov::PartialShape({1, 128, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(data("const1", const1));
    topology.add(data("const2", const2));
    topology.add(convolution("convolution", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("convolution"), input_info("const1"), eltwise_mode::sum));
    topology.add(activation("prelu", input_info("eltwise"), "const2", activation_func::relu_negative_slope));
    topology.add(reorder("reorder", input_info("prelu"), format::bfyx, data_types::f32));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_FALSE(has_node(*prog, "eltwise"));
    ASSERT_FALSE(has_node(*prog, "prelu"));
}

TEST(add_onednn_optimization_attributes, sum_post_op_for_residual_connection) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto input = engine.allocate_memory(layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx});
    auto weight = engine.allocate_memory(layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    topology.add(convolution("conv1", input_info("input"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv2", input_info("conv1"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(convolution("conv3", input_info("conv2"), "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(eltwise("eltwise", input_info("conv1"), input_info("conv3"), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));


    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    auto &conv3 = prog->get_node("conv3");
    auto &cldnn_post_ops = conv3.get_fused_primitives();
    ASSERT_EQ(cldnn_post_ops.size(), 1);
    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(conv3, cldnn_post_ops[0]);

    // Check whether fusing_type is properly selected as sum for residual connection pattern
    ASSERT_EQ(fusing_type, add_fusing_type::sum);

}


// A network input used as the addend is still a network input, however few users it has: the
// buffer belongs to the caller, who may reuse it for the next inference, so a sum post-op must
// not accumulate into it. This case used to be admitted on the strength of the single user.
TEST(add_onednn_optimization_attributes, fc_sum_u8_single_user_input_uses_binary) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ ov::PartialShape({32, 16}), data_types::u8, format::bfyx };
    auto weights_mem = engine.allocate_memory({ ov::PartialShape({8, 16}), data_types::u8, format::bfyx });
    auto extra_layout = layout{ ov::PartialShape({32, 8}), data_types::u8, format::bfyx };

    topology topology;
    topology.add(data("weights", weights_mem));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", extra_layout));

    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::u8));
    topology.add(eltwise("sum", { input_info("fc"), input_info("extra_input") }, eltwise_mode::sum));
    topology.add(reorder("out", input_info("sum"), format::bfyx, data_types::u8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    // eltwise should be fused into fc as a post-op
    auto& fc_node = prog->get_node("fc");
    auto& fused = fc_node.get_fused_primitives();
    ASSERT_EQ(fused.size(), 1);

    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(fc_node, fused[0]);
    ASSERT_EQ(fusing_type, add_fusing_type::binary_per_tensor);
}

TEST(add_onednn_optimization_attributes, fc_sum_u8_residual_input_uses_binary) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ ov::PartialShape({32, 16}), data_types::u8, format::bfyx };
    auto weights1_mem = engine.allocate_memory({ ov::PartialShape({16, 16}), data_types::u8, format::bfyx });
    auto weights2_mem = engine.allocate_memory({ ov::PartialShape({16, 16}), data_types::u8, format::bfyx });

    topology topology;
    topology.add(data("weights1", weights1_mem));
    topology.add(data("weights2", weights2_mem));
    topology.add(input_layout("input", in_layout));

    topology.add(fully_connected("fc1", input_info("input"), { "weights1" }, "", data_types::u8));
    topology.add(fully_connected("fc2", input_info("fc1"), { "weights2" }, "", data_types::u8));
    topology.add(eltwise("sum", { input_info("fc2"), input_info("input") }, eltwise_mode::sum));
    topology.add(reorder("out", input_info("sum"), format::bfyx, data_types::u8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);
    ASSERT_NE(prog, nullptr);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();

    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    auto& fc_node = prog->get_node("fc2");
    auto& fused = fc_node.get_fused_primitives();
    ASSERT_EQ(fused.size(), 1);

    auto fusing_type = onednn_add_fusing_helpers::get_add_fusing_type(fc_node, fused[0]);
    ASSERT_EQ(fusing_type, add_fusing_type::binary_per_tensor);
}

// Builds input -> conv1 -> ... -> conv{depth} plus eltwise(conv1, conv{depth}), which puts conv1
// exactly depth-1 dependency(0) hops above the node the eltwise fuses into.
static add_fusing_type residual_chain_fusing_type(cldnn::engine& engine, size_t depth) {
    auto in_layout = layout{ov::PartialShape({1, 16, 32, 32}), data_types::f16, format::bfyx};
    auto weight = engine.allocate_memory(layout{ov::PartialShape({16, 16, 1, 1}), data_types::f16, format::bfyx});

    topology topology;
    topology.add(input_layout("input", in_layout));
    topology.add(data("weight", weight));
    for (size_t i = 1; i <= depth; i++) {
        auto prev = (i == 1) ? input_info("input") : input_info("conv" + std::to_string(i - 1));
        topology.add(convolution("conv" + std::to_string(i), prev, "weight", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    }
    auto last = "conv" + std::to_string(depth);
    topology.add(eltwise("eltwise", input_info("conv1"), input_info(last), eltwise_mode::sum));
    topology.add(reorder("reorder", input_info("eltwise"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, false);

    prog->get_layout_optimizer().add_all_onednn_impls_optimization_attribute();
    program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    program_wrapper::apply_opt_pass<add_onednn_optimization_attributes>(*prog);

    auto& last_node = prog->get_node(last);
    auto& cldnn_post_ops = last_node.get_fused_primitives();
    EXPECT_EQ(cldnn_post_ops.size(), 1u) << "the residual eltwise did not fuse into " << last;
    if (cldnn_post_ops.size() != 1)
        return add_fusing_type::not_supported;
    // conv1 feeds both conv2 and the fused eltwise, which is what puts get_add_fusing_type on the
    // is_direct_ancestor path rather than the single-user shortcut.
    EXPECT_EQ(prog->get_node("conv1").get_users().size(), 2u);
    return onednn_add_fusing_helpers::get_add_fusing_type(last_node, cldnn_post_ops[0]);
}

// An attention residual reaches six hops back -- projection, reshape, attention, reshape, QKV
// matmul, norm, block input -- and the walk must still find it. See the hop-by-hop listing on
// max_ancestor_walk_depth in program_helpers.cpp.
TEST(add_onednn_optimization_attributes, sum_post_op_for_six_hop_residual) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    ASSERT_EQ(residual_chain_fusing_type(engine, 7), add_fusing_type::sum);
}

// Pins the bound from both sides, so that neither raising nor lowering it passes unnoticed.
// The two depths straddle max_ancestor_walk_depth in program_helpers.cpp: a chain of n
// convolutions puts the target n-1 hops up, and a bound of N reaches N-1 hops. The lower side is
// the same depth the test above uses, since the bound is held at the minimum that case needs.
TEST(add_onednn_optimization_attributes, ancestor_walk_depth_boundary) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    ASSERT_EQ(residual_chain_fusing_type(engine, 7), add_fusing_type::sum);
    ASSERT_EQ(residual_chain_fusing_type(engine, 8), add_fusing_type::binary_per_tensor);
}

// Consuming the caller's input buffer shows up as the network disagreeing with itself: the
// second inference reads an addend the first one already accumulated into, so the same inputs
// stop producing the same output.
TEST(add_onednn_optimization_attributes, sum_post_op_on_input_addend_is_repeatable) {
    auto& engine = get_test_engine();

    if (!engine.get_device_info().supports_immad)
        return;

    auto in_layout = layout{ov::PartialShape({32, 16}), data_types::f16, format::bfyx};
    auto extra_layout = layout{ov::PartialShape({32, 8}), data_types::f16, format::bfyx};

    auto weights_mem = engine.allocate_memory(layout{ov::PartialShape({8, 16}), data_types::f16, format::bfyx});
    auto input_mem = engine.allocate_memory(in_layout);
    auto extra_mem = engine.allocate_memory(extra_layout);

    {
        cldnn::mem_lock<ov::float16, mem_lock_type::write> lock(weights_mem, engine.get_service_stream());
        for (size_t i = 0; i < lock.size(); i++)
            lock[i] = ov::float16(static_cast<float>(i % 5) * 0.125f);
    }
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::write> lock(input_mem, engine.get_service_stream());
        for (size_t i = 0; i < lock.size(); i++)
            lock[i] = ov::float16(static_cast<float>(i % 3) * 0.5f);
    }
    {
        cldnn::mem_lock<ov::float16, mem_lock_type::write> lock(extra_mem, engine.get_service_stream());
        for (size_t i = 0; i < lock.size(); i++)
            lock[i] = ov::float16(static_cast<float>(i % 11) - 5.0f);
    }

    topology topology;
    topology.add(data("weights", weights_mem));
    topology.add(input_layout("input", in_layout));
    topology.add(input_layout("extra_input", extra_layout));
    topology.add(fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f16));
    topology.add(eltwise("sum", { input_info("fc"), input_info("extra_input") }, eltwise_mode::sum));
    topology.add(reorder("out", input_info("sum"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topology, config);
    net.set_input_data("input", input_mem);
    net.set_input_data("extra_input", extra_mem);

    auto read_output = [&]() {
        auto outputs = net.execute();
        auto mem = outputs.at("out").get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> lock(mem, net.get_stream());
        return std::vector<ov::float16>(lock.begin(), lock.end());
    };

    auto first = read_output();
    auto second = read_output();

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); i++)
        ASSERT_EQ(static_cast<float>(first[i]), static_cast<float>(second[i]))
            << "the two inferences disagree at " << i;
}

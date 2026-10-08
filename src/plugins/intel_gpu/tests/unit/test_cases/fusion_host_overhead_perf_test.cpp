// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>

using namespace cldnn;
using namespace ::tests;

// Host-side cost of running a dynamic network whose primitives carry a chain of fused operations.
// Each inference revalidates the fusions, so anything done per fused descriptor shows up here as a
// slope: time per iteration against the number of fused operations per primitive.
//
// These are perf probes, so they are DISABLED like the rest of the perf tests in this directory.
// Run them explicitly:
//   ov_gpu_unit_tests --gtest_also_run_disabled_tests --gtest_filter=fusion_host_overhead*
// and tune with the FUSION_PERF_DEPTH / FUSION_PERF_LAYERS / FUSION_PERF_ITERS environment
// variables.

namespace {

int env_int(const char* name, int fallback) {
    const char* value = getenv(name);
    return value ? atoi(value) : fallback;
}

double measure_us_per_iter(network& net, int iters) {
    for (int i = 0; i < 100; i++)
        net.execute();
    net.get_stream().finish();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++)
        net.execute();
    net.get_stream().finish();
    auto stop = std::chrono::steady_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    return static_cast<double>(ns) / 1000.0 / iters;
}

// Best of five, so that a single scheduling hiccup does not decide the result.
void report(const char* tag, network& net, int iters) {
    std::vector<double> runs;
    for (int r = 0; r < 5; r++)
        runs.push_back(measure_us_per_iter(net, iters));
    std::sort(runs.begin(), runs.end());
    std::cout << tag << ": executed primitives=" << net.get_executed_primitives().size()
              << ", best=" << runs.front() << "us, median=" << runs[2] << "us per iteration" << std::endl;
}

ExecutionConfig dynamic_config(cldnn::engine& engine) {
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    return config;
}

}  // namespace

// A single primitive carrying a long chain of fused activations. The executed graph stays at two
// primitives for any depth, so every change in the reported time is host overhead.
TEST(fusion_host_overhead_perf_test, DISABLED_deep_fusion_chain) {
    auto& engine = get_test_engine();
    auto static_layout = layout{ov::PartialShape({1, 64}), data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(static_layout);
    const int depth = env_int("FUSION_PERF_DEPTH", 40);

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(2), static_layout.data_type, static_layout.format}));
    std::string prev = "input";
    for (int i = 0; i < depth; i++) {
        auto id = "act" + std::to_string(i);
        topology.add(activation(id, input_info(prev), activation_func::relu));
        prev = id;
    }
    topology.add(reorder("out", input_info(prev), static_layout.format, static_layout.data_type));

    network net(engine, topology, dynamic_config(engine));
    net.set_input_data("input", input_mem);
    report("deep_fusion_chain", net, env_int("FUSION_PERF_ITERS", 2000));
}

// The same question on a model-shaped graph: many primitives, each with a short fused chain.
TEST(fusion_host_overhead_perf_test, DISABLED_mlp_with_fused_chain) {
    auto& engine = get_test_engine();
    const int layers = env_int("FUSION_PERF_LAYERS", 20);
    const int fused_per_layer = env_int("FUSION_PERF_FUSE", 4);
    auto static_layout = layout{ov::PartialShape({1, 64}), data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(static_layout);
    const activation_func funcs[] = {activation_func::relu,
                                     activation_func::abs,
                                     activation_func::sqrt,
                                     activation_func::negative};

    topology topology;
    topology.add(input_layout("input", layout{ov::PartialShape::dynamic(2), static_layout.data_type, static_layout.format}));
    std::string prev = "input";
    std::vector<memory::ptr> weights;
    for (int i = 0; i < layers; i++) {
        auto weights_mem = engine.allocate_memory(layout{ov::PartialShape({64, 64}), data_types::f16, format::bfyx});
        weights.push_back(weights_mem);
        auto weights_id = "w" + std::to_string(i);
        auto fc_id = "fc" + std::to_string(i);
        topology.add(data(weights_id, weights_mem));
        topology.add(fully_connected(fc_id, input_info(prev), {weights_id}, "", data_types::f16));
        prev = fc_id;
        for (int f = 0; f < fused_per_layer; f++) {
            auto id = fc_id + "_act" + std::to_string(f);
            topology.add(activation(id, input_info(prev), funcs[f % 4]));
            prev = id;
        }
    }
    topology.add(reorder("out", input_info(prev), static_layout.format, static_layout.data_type));

    network net(engine, topology, dynamic_config(engine));
    net.set_input_data("input", input_mem);
    report("mlp_with_fused_chain", net, env_int("FUSION_PERF_ITERS", 1500));
}

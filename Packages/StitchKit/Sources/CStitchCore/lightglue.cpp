#include "stitchcore.h"

#include <cstdlib>

#include "common.hpp"

extern "C" void sc_extraction_free(sc_extraction *result) {
    if (result == nullptr) return;
    std::free(result->keypoints);
    std::free(result->descriptors);
    *result = sc_extraction{};
}

#ifdef TESSERA_ONNXRUNTIME

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

using stitchcore::write_error;

namespace {

Ort::Env &shared_environment() {
    static Ort::Env environment(ORT_LOGGING_LEVEL_ERROR, "stitchcore");
    return environment;
}

const char *compute_units(sc_execution execution) {
    switch (execution) {
        case SC_EXECUTION_COREML_CPU_ONLY: return "CPUOnly";
        case SC_EXECUTION_COREML_CPU_AND_GPU: return "CPUAndGPU";
        case SC_EXECUTION_COREML_ALL: return "ALL";
        case SC_EXECUTION_CPU: return nullptr;
    }
    return nullptr;
}

}  // namespace

struct sc_onnx_model {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

extern "C" sc_onnx_model *sc_onnx_model_create(const char *model_path, sc_execution execution,
                                               const char *cache_directory, int32_t intra_op_threads,
                                               int32_t low_memory, char *error, size_t error_length) {
    try {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (intra_op_threads > 0) {
            options.SetIntraOpNumThreads(std::min(intra_op_threads, 64));
        }
        if (low_memory) {
            options.DisableCpuMemArena();
            options.DisableMemPattern();
        }
        if (const char *units = compute_units(execution)) {
            std::unordered_map<std::string, std::string> provider{
                {"ModelFormat", "MLProgram"},
                {"MLComputeUnits", units},
                {"RequireStaticInputShapes", "1"},
            };
            if (cache_directory != nullptr && cache_directory[0] != '\0') {
                provider["ModelCacheDirectory"] = cache_directory;
            }
            options.AppendExecutionProvider("CoreML", provider);
        }

        auto result = std::make_unique<sc_onnx_model>();
        result->session = std::make_unique<Ort::Session>(shared_environment(), model_path, options);
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < result->session->GetInputCount(); ++i) {
            result->input_names.emplace_back(result->session->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < result->session->GetOutputCount(); ++i) {
            result->output_names.emplace_back(result->session->GetOutputNameAllocated(i, allocator).get());
        }
        return result.release();
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return nullptr;
    }
}

extern "C" void sc_onnx_model_free(sc_onnx_model *model) { delete model; }

namespace {

// Runs the model after checking that the caller supplies exactly the inputs it declares and that it
// produces at least `minimum_outputs` outputs; a foreign model file then fails with a message.
std::vector<Ort::Value> run(sc_onnx_model *model, std::vector<Ort::Value> &inputs, size_t minimum_outputs) {
    if (inputs.size() != model->input_names.size()) {
        throw std::runtime_error("model expects " + std::to_string(model->input_names.size()) + " inputs, got " +
                                 std::to_string(inputs.size()) + ": wrong model file?");
    }
    if (model->output_names.size() < minimum_outputs) {
        throw std::runtime_error("model has " + std::to_string(model->output_names.size()) + " outputs, needs " +
                                 std::to_string(minimum_outputs) + ": wrong model file?");
    }
    std::vector<const char *> input_names, output_names;
    for (const auto &name : model->input_names) input_names.push_back(name.c_str());
    for (const auto &name : model->output_names) output_names.push_back(name.c_str());
    return model->session->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                               output_names.data(), output_names.size());
}

// Checks an output's element type and shape before its data is read.
void expect(const Ort::Value &value, ONNXTensorElementDataType type, const std::vector<int64_t> &shape,
            const char *name) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != type) {
        throw std::runtime_error(std::string("unexpected element type for ") + name);
    }
    const auto actual = info.GetShape();
    bool ok = actual.size() == shape.size();
    for (size_t i = 0; ok && i < shape.size(); ++i) {
        ok = actual[i] >= 0 && (shape[i] < 0 || actual[i] == shape[i]);
    }
    if (!ok) {
        throw std::runtime_error(std::string("unexpected shape for ") + name);
    }
}

Ort::MemoryInfo &cpu_memory() {
    static Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return memory;
}

}  // namespace

namespace {

Ort::Value tensor(const float *data, int64_t channels, int64_t height, int64_t width) {
    const std::array<int64_t, 4> shape{1, channels, height, width};
    return Ort::Value::CreateTensor<float>(cpu_memory(), const_cast<float *>(data),
                                           static_cast<size_t>(channels * height * width), shape.data(),
                                           shape.size());
}

int32_t collect(std::vector<Ort::Value> &outputs, sc_extraction *result);

}  // namespace

extern "C" int32_t sc_extract_sparse(sc_onnx_model *model, const float *logits, const float *ranker,
                                     const float *features, int32_t width, int32_t height, int32_t channels,
                                     sc_extraction *result, char *error, size_t error_length) {
    if (model == nullptr || result == nullptr || logits == nullptr || ranker == nullptr || features == nullptr) {
        write_error(error, error_length, "missing argument");
        return 1;
    }
    *result = sc_extraction{};
    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(tensor(logits, 1, height, width));
        inputs.push_back(tensor(ranker, 1, height, width));
        inputs.push_back(tensor(features, channels, height, width));
        std::vector<Ort::Value> outputs = run(model, inputs, 2);
        return collect(outputs, result);
    } catch (const std::exception &e) {
        sc_extraction_free(result);
        write_error(error, error_length, e.what());
        return 1;
    }
}

extern "C" int32_t sc_select_keypoints(sc_onnx_model *model, const float *logits, const float *ranker,
                                       int32_t width, int32_t height, sc_extraction *result, char *error,
                                       size_t error_length) {
    if (model == nullptr || result == nullptr || logits == nullptr || ranker == nullptr) {
        write_error(error, error_length, "missing argument");
        return 1;
    }
    *result = sc_extraction{};
    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(tensor(logits, 1, height, width));
        inputs.push_back(tensor(ranker, 1, height, width));
        std::vector<Ort::Value> outputs = run(model, inputs, 1);
        expect(outputs.at(0), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 2}, "keypoints");
        const auto shape = outputs.at(0).GetTensorTypeAndShapeInfo().GetShape();
        const int32_t count = static_cast<int32_t>(shape.at(1));
        result->keypoint_count = count;
        result->descriptor_size = 0;
        result->keypoints = stitchcore::allocate_array<float>(static_cast<size_t>(count) * 2);
        result->descriptors = stitchcore::allocate_array<float>(0);
        std::memcpy(result->keypoints, outputs[0].GetTensorData<float>(), sizeof(float) * count * 2);
        return 0;
    } catch (const std::exception &e) {
        sc_extraction_free(result);
        write_error(error, error_length, e.what());
        return 1;
    }
}

extern "C" int32_t sc_extract(sc_onnx_model *model, const float *chw, int32_t width, int32_t height,
                              sc_extraction *result, char *error, size_t error_length) {
    if (model == nullptr || result == nullptr || chw == nullptr) {
        write_error(error, error_length, "missing argument");
        return 1;
    }
    *result = sc_extraction{};
    try {
        std::vector<Ort::Value> inputs;
        inputs.push_back(tensor(chw, 3, height, width));
        std::vector<Ort::Value> outputs = run(model, inputs, 2);
        return collect(outputs, result);
    } catch (const std::exception &e) {
        sc_extraction_free(result);
        write_error(error, error_length, e.what());
        return 1;
    }
}

namespace {

int32_t collect(std::vector<Ort::Value> &outputs, sc_extraction *result) {
    // keypoints: [1, K, 2]; descriptors: [1, K, D]
    expect(outputs.at(0), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, 2}, "keypoints");
    expect(outputs.at(1), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, -1, -1}, "descriptors");
    const auto keypoint_shape = outputs.at(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto descriptor_shape = outputs.at(1).GetTensorTypeAndShapeInfo().GetShape();
    if (descriptor_shape[1] != keypoint_shape[1]) {
        throw std::runtime_error("keypoint and descriptor counts differ");
    }
    const int32_t count = static_cast<int32_t>(keypoint_shape.at(1));
    const int32_t size = static_cast<int32_t>(descriptor_shape.at(2));
    result->keypoint_count = count;
    result->descriptor_size = size;
    result->keypoints = stitchcore::allocate_array<float>(static_cast<size_t>(count) * 2);
    result->descriptors = stitchcore::allocate_array<float>(static_cast<size_t>(count) * size);
    std::memcpy(result->keypoints, outputs[0].GetTensorData<float>(), sizeof(float) * count * 2);
    std::memcpy(result->descriptors, outputs[1].GetTensorData<float>(), sizeof(float) * count * size);
    return 0;
}

}  // namespace

extern "C" int32_t sc_match_learned(sc_onnx_model *model, const float *keypoints, const float *descriptors,
                                    int32_t keypoint_count, int32_t descriptor_size, int32_t *partner,
                                    float *confidence, char *error, size_t error_length) {
    if (model == nullptr || keypoints == nullptr || descriptors == nullptr || partner == nullptr ||
        confidence == nullptr || keypoint_count <= 0 || descriptor_size <= 0) {
        write_error(error, error_length, "missing argument");
        return 1;
    }
    try {
        const std::array<int64_t, 3> keypoint_shape{2, keypoint_count, 2};
        const std::array<int64_t, 3> descriptor_shape{2, keypoint_count, descriptor_size};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu_memory(), const_cast<float *>(keypoints),
                                                         static_cast<size_t>(4) * keypoint_count,
                                                         keypoint_shape.data(), keypoint_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(cpu_memory(), const_cast<float *>(descriptors),
                                                         static_cast<size_t>(2) * keypoint_count * descriptor_size,
                                                         descriptor_shape.data(), descriptor_shape.size()));
        std::vector<Ort::Value> outputs = run(model, inputs, 2);
        expect(outputs.at(0), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {1, keypoint_count}, "match0");
        expect(outputs.at(1), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {1, keypoint_count}, "confidence0");
        const int64_t *match0 = outputs.at(0).GetTensorData<int64_t>();
        const float *confidence0 = outputs.at(1).GetTensorData<float>();
        for (int32_t i = 0; i < keypoint_count; ++i) {
            const bool valid = match0[i] >= 0 && match0[i] < keypoint_count && std::isfinite(confidence0[i]);
            partner[i] = valid ? static_cast<int32_t>(match0[i]) : -1;
            confidence[i] = valid ? confidence0[i] : 0.0f;
        }
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return 1;
    }
}

extern "C" int32_t sc_onnx_available(void) { return 1; }

#else

// Built without ONNX Runtime: every ONNX entry point fails with a message, and the callers fall back
// to the Core ML and C++ paths.

namespace {
constexpr const char *kNoRuntime = "Tessera was built without ONNX Runtime";
}

extern "C" sc_onnx_model *sc_onnx_model_create(const char *, sc_execution, const char *, int32_t, int32_t,
                                              char *error, size_t error_length) {
    stitchcore::write_error(error, error_length, kNoRuntime);
    return nullptr;
}

extern "C" void sc_onnx_model_free(sc_onnx_model *) {}

extern "C" int32_t sc_extract(sc_onnx_model *, const float *, int32_t, int32_t, sc_extraction *, char *error,
                              size_t error_length) {
    stitchcore::write_error(error, error_length, kNoRuntime);
    return 1;
}

extern "C" int32_t sc_extract_sparse(sc_onnx_model *, const float *, const float *, const float *, int32_t, int32_t,
                                     int32_t, sc_extraction *, char *error, size_t error_length) {
    stitchcore::write_error(error, error_length, kNoRuntime);
    return 1;
}

extern "C" int32_t sc_select_keypoints(sc_onnx_model *, const float *, const float *, int32_t, int32_t,
                                       sc_extraction *, char *error, size_t error_length) {
    stitchcore::write_error(error, error_length, kNoRuntime);
    return 1;
}

extern "C" int32_t sc_match_learned(sc_onnx_model *, const float *, const float *, int32_t, int32_t, int32_t *,
                                    float *, char *error, size_t error_length) {
    stitchcore::write_error(error, error_length, kNoRuntime);
    return 1;
}

extern "C" int32_t sc_onnx_available(void) { return 0; }

#endif

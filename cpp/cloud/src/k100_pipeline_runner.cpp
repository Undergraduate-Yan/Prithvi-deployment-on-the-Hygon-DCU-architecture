#include "k100_pipeline_runner.hpp"

#include <hip/hip_runtime.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace k100 {
namespace {

constexpr const char* kPlanSchemaV1 = "K100_PIPELINE_PLAN_V1";
constexpr const char* kPlanSchemaV2 = "K100_PIPELINE_PLAN_V2";
constexpr const char* kExpectedHardware = "海光 K100 AI 加速卡";
constexpr const char* kExpectedRuntime = "1.19.2";
constexpr const char* kExpectedImage =
    "sha256:bdc7e817dfd8587ed477834b5304f35c6b0ffcbe9f51645cf64259bd81c17b01";

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error(message);
}

void HipCheck(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    Fail(std::string(operation) + ": " + hipGetErrorString(status));
  }
}

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> fields;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, delimiter)) {
    fields.push_back(field);
  }
  if (!value.empty() && value.back() == delimiter) {
    fields.emplace_back();
  }
  return fields;
}

std::size_t ParseSize(const std::string& value, const std::string& label) {
  if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return c >= '0' && c <= '9';
      })) {
    Fail("invalid non-negative integer for " + label + ": " + value);
  }
  try {
    const auto parsed = std::stoull(value);
    if (parsed > std::numeric_limits<std::size_t>::max()) {
      Fail("integer out of range for " + label);
    }
    return static_cast<std::size_t>(parsed);
  } catch (const std::exception&) {
    Fail("invalid integer for " + label + ": " + value);
  }
}

std::vector<std::int64_t> ParseShape(const std::string& text) {
  if (text.empty()) {
    Fail("empty tensor shape");
  }
  std::vector<std::int64_t> result;
  for (const auto& field : Split(text, ',')) {
    const auto value = ParseSize(field, "tensor dimension");
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      Fail("tensor dimensions must be positive int64 values");
    }
    result.push_back(static_cast<std::int64_t>(value));
  }
  return result;
}

TensorDType ParseDType(const std::string& text) {
  if (text == "float32") {
    return TensorDType::Float32;
  }
  if (text == "float16") {
    return TensorDType::Float16;
  }
  Fail("unsupported tensor dtype: " + text);
}

ONNXTensorElementDataType ToOnnxDType(TensorDType dtype) {
  return dtype == TensorDType::Float16 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
                                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

std::size_t DTypeBytes(TensorDType dtype) {
  return dtype == TensorDType::Float16 ? sizeof(std::uint16_t) : sizeof(float);
}

std::size_t ElementCount(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    Fail("scalar/empty shapes are not supported");
  }
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0 || count > std::numeric_limits<std::size_t>::max() /
                                      static_cast<std::size_t>(dimension)) {
      Fail("invalid or overflowing static tensor shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

void RequireRegularFileSize(const std::string& path, std::uintmax_t expected,
                            const std::string& label) {
  std::error_code error;
  const std::filesystem::path file(path);
  if (!std::filesystem::is_regular_file(file, error) || error) {
    Fail(label + " is not a readable regular file: " + path);
  }
  const auto observed = std::filesystem::file_size(file, error);
  if (error || observed != expected) {
    Fail(label + " size drift: " + path);
  }
}

void SetCacheEnvironment(const std::string& path) {
  const std::vector<std::pair<const char*, std::string>> values = {
      {"ORT_MIGRAPHX_SAVE_COMPILED_MODEL", "0"},
      {"ORT_MIGRAPHX_LOAD_COMPILED_MODEL", "1"},
      {"ORT_MIGRAPHX_SAVE_COMPILE_PATH", path},
      {"ORT_MIGRAPHX_LOAD_COMPILE_PATH", path},
      {"MIGRAPHX_GPU_COMPILE_PARALLEL", "1"},
      {"ORT_MIGRAPHX_EXHAUSTIVE_TUNE", "0"},
      {"OMP_NUM_THREADS", "1"},
      {"OPENBLAS_NUM_THREADS", "1"},
      {"MKL_NUM_THREADS", "1"},
      {"NUMEXPR_NUM_THREADS", "1"},
      {"MALLOC_ARENA_MAX", "2"},
  };
  for (const auto& item : values) {
    if (::setenv(item.first, item.second.c_str(), 1) != 0) {
      Fail(std::string("setenv failed for ") + item.first + ": errno=" +
           std::to_string(errno));
    }
  }
}

struct HipBuffer {
  explicit HipBuffer(std::size_t byte_count) : bytes(byte_count) {
    if (bytes == 0) {
      Fail("refusing zero-byte device allocation");
    }
    HipCheck(hipMalloc(&pointer, bytes), "hipMalloc");
  }
  ~HipBuffer() {
    if (pointer != nullptr) {
      static_cast<void>(hipFree(pointer));
    }
  }
  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  void* pointer{};
  std::size_t bytes{};
};

struct DeviceTensor {
  DeviceTensor(const TensorSpec& spec, const Ort::MemoryInfo& memory_info)
      : name(spec.name), shape(spec.shape), elements(ElementCount(shape)),
        dtype(spec.dtype), buffer(elements * DTypeBytes(dtype)),
        value(Ort::Value::CreateTensor(memory_info, buffer.pointer, buffer.bytes,
                                       shape.data(), shape.size(), ToOnnxDType(dtype))) {}

  std::string name;
  std::vector<std::int64_t> shape;
  std::size_t elements{};
  TensorDType dtype{TensorDType::Float32};
  HipBuffer buffer;
  Ort::Value value;
};

void ValidateTensorMetadata(const Ort::TypeInfo& type_info, const TensorSpec& spec,
                            const std::string& label) {
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  if (tensor_info.GetElementType() != ToOnnxDType(spec.dtype)) {
    Fail(label + " dtype does not match the frozen plan");
  }
  const auto observed_shape = tensor_info.GetShape();
  bool compatible = observed_shape.size() == spec.shape.size();
  for (std::size_t index = 0; compatible && index < observed_shape.size(); ++index) {
    const auto observed = observed_shape[index];
    const auto expected = spec.shape[index];
    compatible = observed == expected ||
        (index == 0 && observed < 0 && expected == 1);
  }
  if (!compatible) {
    Fail(label + " static shape drift");
  }
}

}  // namespace

PipelinePlan LoadPipelinePlan(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    Fail("cannot open pipeline plan: " + path);
  }
  std::string line;
  if (!std::getline(input, line) || (line != kPlanSchemaV1 && line != kPlanSchemaV2)) {
    Fail("unsupported pipeline plan schema");
  }
  const bool typed_plan = line == kPlanSchemaV2;
  PipelinePlan plan;
  std::size_t declared_sessions = std::numeric_limits<std::size_t>::max();
  bool ended = false;
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      Fail("blank lines are forbidden in pipeline plan");
    }
    const auto fields = Split(line, '\t');
    const auto require_fields = [&](std::size_t count) {
      if (fields.size() != count) {
        Fail("field-count drift at plan line " + std::to_string(line_number));
      }
    };
    if (fields[0] == "variant") {
      require_fields(2);
      plan.variant = fields[1];
    } else if (fields[0] == "hardware") {
      require_fields(2);
      plan.hardware = fields[1];
    } else if (fields[0] == "runtime") {
      require_fields(2);
      plan.runtime_version = fields[1];
    } else if (fields[0] == "image") {
      require_fields(2);
      plan.image_id = fields[1];
    } else if (fields[0] == "session_count") {
      require_fields(2);
      declared_sessions = ParseSize(fields[1], "session_count");
    } else if (fields[0] == "input" || fields[0] == "final_logits") {
      require_fields(3);
      TensorSpec spec{fields[1], ParseShape(fields[2]), TensorDType::Float32};
      if (fields[0] == "input") {
        plan.external_input = std::move(spec);
      } else {
        plan.final_logits = std::move(spec);
      }
    } else if (fields[0] == "S") {
      if (fields.size() < 10) {
        Fail("truncated session row at plan line " + std::to_string(line_number));
      }
      SessionSpec spec;
      spec.index = ParseSize(fields[1], "session index");
      spec.session_id = fields[2];
      spec.model_path = fields[3];
      spec.model_size = ParseSize(fields[4], "model size");
      spec.cache_path = fields[5];
      spec.cache_size = ParseSize(fields[6], "cache size");
      const auto input_count = ParseSize(fields[7], "input count");
      std::size_t cursor = 8;
      if (input_count == 0 || cursor + input_count >= fields.size()) {
        Fail("invalid input list at plan line " + std::to_string(line_number));
      }
      spec.inputs.insert(spec.inputs.end(), fields.begin() + static_cast<std::ptrdiff_t>(cursor),
                         fields.begin() + static_cast<std::ptrdiff_t>(cursor + input_count));
      cursor += input_count;
      const auto output_count = ParseSize(fields[cursor++], "output count");
      const std::size_t fields_per_output = typed_plan ? 3 : 2;
      if (output_count == 0 || cursor + output_count * fields_per_output != fields.size()) {
        Fail("invalid output list at plan line " + std::to_string(line_number));
      }
      for (std::size_t index = 0; index < output_count; ++index) {
        spec.outputs.push_back({fields[cursor], ParseShape(fields[cursor + 1]),
                                typed_plan ? ParseDType(fields[cursor + 2])
                                           : TensorDType::Float32});
        cursor += fields_per_output;
      }
      plan.sessions.push_back(std::move(spec));
    } else if (fields[0] == "END") {
      require_fields(1);
      ended = true;
      break;
    } else {
      Fail("unknown pipeline plan row at line " + std::to_string(line_number));
    }
  }
  if (!ended || input.peek() != std::ifstream::traits_type::eof()) {
    Fail("pipeline plan must contain one terminal END row");
  }
  if (plan.variant.empty() || plan.hardware != kExpectedHardware ||
      plan.runtime_version != kExpectedRuntime || plan.image_id != kExpectedImage ||
      plan.external_input.name != "input" ||
      plan.external_input.shape != std::vector<std::int64_t>({1, 6, 224, 224}) ||
      plan.final_logits.name != "logits" ||
      plan.final_logits.shape != std::vector<std::int64_t>({1, 4, 224, 224}) ||
      declared_sessions != plan.sessions.size() || plan.sessions.empty()) {
    Fail("pipeline plan frozen contract drift");
  }
  std::unordered_map<std::string, TensorSpec> available;
  available.emplace(plan.external_input.name, plan.external_input);
  for (std::size_t ordinal = 0; ordinal < plan.sessions.size(); ++ordinal) {
    const auto& session = plan.sessions[ordinal];
    if (session.index != ordinal || session.session_id.empty()) {
      Fail("session order/id drift");
    }
    for (const auto& name : session.inputs) {
      if (available.find(name) == available.end()) {
        Fail("session input has no earlier producer: " + name);
      }
    }
    for (const auto& output : session.outputs) {
      if (output.name.empty() || available.find(output.name) != available.end()) {
        Fail("duplicate/empty tensor producer: " + output.name);
      }
      available.emplace(output.name, output);
    }
  }
  const auto final = available.find(plan.final_logits.name);
  if (final == available.end() || final->second.shape != plan.final_logits.shape ||
      final->second.dtype != TensorDType::Float32) {
    Fail("final logits producer contract drift");
  }
  return plan;
}

struct PipelineRunner::Impl {
  struct SessionState {
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::IoBinding> binding;
  };

  Impl(PipelinePlan incoming, int requested_device, std::string requested_profile_dir)
      : plan(std::move(incoming)), device_id(requested_device),
        profile_dir(std::move(requested_profile_dir)),
        env(ORT_LOGGING_LEVEL_WARNING, "k100_phase2e_cpp_runner"),
        device_memory("Cuda", OrtDeviceAllocator, requested_device, OrtMemTypeDefault) {
    if (device_id < 0) {
      Fail("device id must be non-negative");
    }
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "MIGraphXExecutionProvider") ==
        providers.end()) {
      Fail("MIGraphXExecutionProvider is unavailable");
    }
    if (!profile_dir.empty()) {
      std::error_code error;
      if (!std::filesystem::is_directory(profile_dir, error) || error) {
        Fail("profile directory must already exist");
      }
    }
    tensors.emplace(plan.external_input.name,
                    std::make_unique<DeviceTensor>(plan.external_input, device_memory));
    Ort::AllocatorWithDefaultOptions allocator;
    states.reserve(plan.sessions.size());
    for (const auto& spec : plan.sessions) {
      RequireRegularFileSize(spec.model_path, spec.model_size, "model");
      RequireRegularFileSize(spec.cache_path, spec.cache_size, "cache");
      SetCacheEnvironment(spec.cache_path);
      Ort::SessionOptions options;
      options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);
      options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
      if (!profile_dir.empty()) {
        const auto prefix = profile_dir + "/session_" +
                            (spec.index < 10 ? std::string("0") : std::string()) +
                            std::to_string(spec.index);
        options.EnableProfiling(prefix.c_str());
      }
      Ort::ThrowOnError(
          OrtSessionOptionsAppendExecutionProvider_MIGraphX(options, device_id));
      SessionState state;
      state.session = std::make_unique<Ort::Session>(env, spec.model_path.c_str(), options);
      if (state.session->GetInputCount() != spec.inputs.size() ||
          state.session->GetOutputCount() < spec.outputs.size()) {
        Fail("session I/O arity drift: " + spec.session_id);
      }
      for (std::size_t index = 0; index < spec.inputs.size(); ++index) {
        const auto observed = state.session->GetInputNameAllocated(index, allocator);
        if (spec.inputs[index] != observed.get()) {
          Fail("session input-name/order drift: " + spec.session_id);
        }
        const auto tensor = tensors.find(spec.inputs[index]);
        if (tensor == tensors.end()) {
          Fail("missing device input tensor: " + spec.inputs[index]);
        }
        ValidateTensorMetadata(state.session->GetInputTypeInfo(index),
                               TensorSpec{spec.inputs[index], tensor->second->shape,
                                          tensor->second->dtype},
                               "session input " + spec.inputs[index]);
      }
      for (std::size_t index = 0; index < spec.outputs.size(); ++index) {
        const auto observed = state.session->GetOutputNameAllocated(index, allocator);
        if (spec.outputs[index].name != observed.get()) {
          Fail("session output-name/order drift: " + spec.session_id);
        }
        ValidateTensorMetadata(state.session->GetOutputTypeInfo(index), spec.outputs[index],
                               "session output " + spec.outputs[index].name);
        const auto inserted = tensors.emplace(
            spec.outputs[index].name,
            std::make_unique<DeviceTensor>(spec.outputs[index], device_memory));
        if (!inserted.second) {
          Fail("duplicate output tensor allocation: " + spec.outputs[index].name);
        }
      }
      state.binding = std::make_unique<Ort::IoBinding>(*state.session);
      for (const auto& name : spec.inputs) {
        state.binding->BindInput(name.c_str(), tensors.at(name)->value);
      }
      for (const auto& output : spec.outputs) {
        state.binding->BindOutput(output.name.c_str(), tensors.at(output.name)->value);
      }
      states.push_back(std::move(state));
    }
    input = tensors.at(plan.external_input.name).get();
    logits = tensors.at(plan.final_logits.name).get();
  }

  PipelinePlan plan;
  int device_id{};
  std::string profile_dir;
  bool profiles_finished{false};
  Ort::Env env;
  Ort::MemoryInfo device_memory;
  std::unordered_map<std::string, std::unique_ptr<DeviceTensor>> tensors;
  std::vector<SessionState> states;
  DeviceTensor* input{};
  DeviceTensor* logits{};
};

PipelineRunner::PipelineRunner(PipelinePlan plan, int device_id,
                               const std::string& profile_dir)
    : impl_(std::make_unique<Impl>(std::move(plan), device_id, profile_dir)) {}

PipelineRunner::~PipelineRunner() = default;

void PipelineRunner::CopyInputFromHost(const float* source, std::size_t element_count) {
  if (source == nullptr || element_count != impl_->input->elements) {
    Fail("host input element-count drift");
  }
  HipCheck(hipMemcpy(impl_->input->buffer.pointer, source,
                     element_count * sizeof(float), hipMemcpyHostToDevice),
           "hipMemcpy host-to-device input");
}

void PipelineRunner::RunPipeline() {
  Ort::RunOptions run_options;
  for (auto& state : impl_->states) {
    state.session->Run(run_options, *state.binding);
  }
}

void PipelineRunner::CopyLogitsToHost(float* destination, std::size_t element_count) {
  if (destination == nullptr || element_count != impl_->logits->elements) {
    Fail("host logits element-count drift");
  }
  HipCheck(hipMemcpy(destination, impl_->logits->buffer.pointer,
                     element_count * sizeof(float), hipMemcpyDeviceToHost),
           "hipMemcpy device-to-host logits");
}

std::vector<std::string> PipelineRunner::FinishProfiles() {
  if (impl_->profile_dir.empty()) {
    return {};
  }
  if (impl_->profiles_finished) {
    Fail("profiles were already finalized");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> paths;
  paths.reserve(impl_->states.size());
  for (auto& state : impl_->states) {
    auto path = state.session->EndProfilingAllocated(allocator);
    paths.emplace_back(path.get());
  }
  impl_->profiles_finished = true;
  return paths;
}

std::size_t PipelineRunner::input_elements() const { return impl_->input->elements; }
std::size_t PipelineRunner::logits_elements() const { return impl_->logits->elements; }
std::size_t PipelineRunner::session_count() const { return impl_->states.size(); }

}  // namespace k100

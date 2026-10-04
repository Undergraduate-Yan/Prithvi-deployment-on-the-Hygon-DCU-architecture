#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace k100 {

struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;
};

struct SessionSpec {
  std::size_t index{};
  std::string session_id;
  std::string model_path;
  std::uintmax_t model_size{};
  std::string cache_path;
  std::uintmax_t cache_size{};
  std::vector<std::string> inputs;
  std::vector<TensorSpec> outputs;
};

struct PipelinePlan {
  std::string variant;
  std::string hardware;
  std::string runtime_version;
  std::string image_id;
  TensorSpec external_input;
  TensorSpec final_logits;
  std::vector<SessionSpec> sessions;
};

PipelinePlan LoadPipelinePlan(const std::string& path);

class PipelineRunner {
 public:
  PipelineRunner(PipelinePlan plan, int device_id, const std::string& profile_dir);
  ~PipelineRunner();
  PipelineRunner(const PipelineRunner&) = delete;
  PipelineRunner& operator=(const PipelineRunner&) = delete;

  void CopyInputFromHost(const float* source, std::size_t element_count);
  void RunPipeline();
  void CopyLogitsToHost(float* destination, std::size_t element_count);
  std::vector<std::string> FinishProfiles();

  [[nodiscard]] std::size_t input_elements() const;
  [[nodiscard]] std::size_t logits_elements() const;
  [[nodiscard]] std::size_t session_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace k100

#include "k100_pipeline_runner.hpp"

#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
  std::string plan;
  std::string input_raw;
  std::string output_raw;
  std::string latency_csv;
  std::string profile_dir;
  std::string stability_signature_csv;
  std::string scope{"logits"};
  int device{};
  std::size_t warmups{};
  std::size_t iterations{};
};

struct CallRecord {
  std::int64_t started_unix_ns{};
  std::int64_t ended_unix_ns{};
  double latency_ms{};
  int cpu_before{};
  int cpu_after{};
};

std::size_t ParseSize(const std::string& value, const std::string& label) {
  if (value.empty() || value[0] == '-') {
    throw std::runtime_error("invalid " + label);
  }
  std::size_t consumed{};
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("invalid " + label);
  }
  return static_cast<std::size_t>(parsed);
}

Arguments ParseArguments(int argc, char** argv) {
  std::map<std::string, std::string> values;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
      throw std::runtime_error("arguments must be --key value pairs");
    }
    if (!values.emplace(argv[index], argv[index + 1]).second) {
      throw std::runtime_error("duplicate argument: " + std::string(argv[index]));
    }
  }
  const auto required = [&](const std::string& key) -> std::string {
    const auto found = values.find(key);
    if (found == values.end() || found->second.empty()) {
      throw std::runtime_error("missing required argument: " + key);
    }
    return found->second;
  };
  Arguments args;
  args.plan = required("--plan");
  args.input_raw = required("--input-raw");
  args.output_raw = required("--output-raw");
  args.latency_csv = required("--latency-csv");
  args.scope = required("--scope");
  args.device = static_cast<int>(ParseSize(required("--device"), "device"));
  args.warmups = ParseSize(required("--warmups"), "warmups");
  args.iterations = ParseSize(required("--iterations"), "iterations");
  const auto profile = values.find("--profile-dir");
  if (profile != values.end()) {
    args.profile_dir = profile->second;
  }
  const auto signatures = values.find("--stability-signature-csv");
  if (signatures != values.end()) {
    args.stability_signature_csv = signatures->second;
  }
  const std::vector<std::string> allowed = {
      "--plan",       "--input-raw", "--output-raw", "--latency-csv",
      "--scope",      "--device",    "--warmups",    "--iterations",
      "--profile-dir", "--stability-signature-csv"};
  for (const auto& item : values) {
    if (std::find(allowed.begin(), allowed.end(), item.first) == allowed.end()) {
      throw std::runtime_error("unknown argument: " + item.first);
    }
  }
  if ((args.scope != "model" && args.scope != "logits" && args.scope != "mask") ||
      args.iterations == 0 || args.iterations > 250000 || args.warmups > 100000 ||
      args.device < 0) {
    throw std::runtime_error("argument contract violation");
  }
  return args;
}

template <typename T>
std::vector<T> ReadExact(const std::string& path, std::size_t element_count) {
  const auto expected_bytes = element_count * sizeof(T);
  if (std::filesystem::file_size(path) != expected_bytes) {
    throw std::runtime_error("raw input byte-count drift");
  }
  std::vector<T> result(element_count);
  std::ifstream stream(path, std::ios::binary);
  stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(expected_bytes));
  if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("raw input read failed");
  }
  return result;
}

template <typename T>
void WriteExact(const std::string& path, const std::vector<T>& values) {
  if (std::filesystem::exists(path)) {
    throw std::runtime_error("refusing to overwrite output: " + path);
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) {
    throw std::runtime_error("raw output write failed");
  }
}

void MakeMask(const std::vector<float>& logits, std::vector<std::uint8_t>& mask) {
  constexpr std::size_t kPixels = 224 * 224;
  constexpr std::size_t kClasses = 4;
  if (logits.size() != kClasses * kPixels || mask.size() != kPixels) {
    throw std::runtime_error("mask conversion shape drift");
  }
  for (std::size_t index = 0; index < kPixels; ++index) {
    std::size_t best = 0;
    for (std::size_t class_index = 1; class_index < kClasses; ++class_index) {
      if (logits[class_index * kPixels + index] > logits[best * kPixels + index]) {
        best = class_index;
      }
    }
    mask[index] = static_cast<std::uint8_t>(best);
  }
}

void ValidateFinite(const std::vector<float>& values) {
  if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("non-finite logits detected");
  }
}

std::uint64_t MaskSignature(const std::vector<std::uint8_t>& mask) {
  std::uint64_t value = 1469598103934665603ULL;
  for (const auto item : mask) {
    value ^= item;
    value *= 1099511628211ULL;
  }
  return value;
}

std::int64_t UnixNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string CurrentAffinity() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_getaffinity failed");
  }
  std::ostringstream value;
  bool first = true;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &set)) {
      if (!first) {
        value << ';';
      }
      value << cpu;
      first = false;
    }
  }
  if (first) {
    throw std::runtime_error("empty CPU affinity");
  }
  return value.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto process_started = std::chrono::steady_clock::now();
    const auto args = ParseArguments(argc, argv);
    const auto affinity = CurrentAffinity();
    const auto plan = k100::LoadPipelinePlan(args.plan);
    const auto manifest_loaded = std::chrono::steady_clock::now();
    k100::PipelineRunner runner(plan, args.device, args.profile_dir);
    const auto sessions_loaded = std::chrono::steady_clock::now();
    auto input = ReadExact<float>(args.input_raw, runner.input_elements());
    ValidateFinite(input);
    std::vector<float> logits(runner.logits_elements());
    std::vector<std::uint8_t> mask(224 * 224);
    runner.CopyInputFromHost(input.data(), input.size());
    const auto runner_initialized = std::chrono::steady_clock::now();
    const auto execute = [&]() {
      runner.RunPipeline();
      if (args.scope != "model") {
        runner.CopyLogitsToHost(logits.data(), logits.size());
        if (args.scope == "mask") {
          MakeMask(logits, mask);
        }
      }
    };
    for (std::size_t index = 0; index < args.warmups; ++index) {
      execute();
    }
    std::vector<CallRecord> calls;
    calls.reserve(args.iterations);
    std::vector<std::uint64_t> signatures;
    if (!args.stability_signature_csv.empty()) {
      if (args.scope == "model") {
        throw std::runtime_error("stability signatures require logits or mask scope");
      }
      signatures.reserve(args.iterations);
    }
    for (std::size_t index = 0; index < args.iterations; ++index) {
      CallRecord record;
      record.cpu_before = sched_getcpu();
      record.started_unix_ns = UnixNanoseconds();
      const auto started = std::chrono::steady_clock::now();
      execute();
      const auto ended = std::chrono::steady_clock::now();
      record.ended_unix_ns = UnixNanoseconds();
      record.cpu_after = sched_getcpu();
      record.latency_ms =
          std::chrono::duration<double, std::milli>(ended - started).count();
      calls.push_back(record);
      if (!args.stability_signature_csv.empty()) {
        ValidateFinite(logits);
        MakeMask(logits, mask);
        signatures.push_back(MaskSignature(mask));
      }
    }
    if (args.scope == "model") {
      runner.CopyLogitsToHost(logits.data(), logits.size());
    }
    ValidateFinite(logits);
    const auto profiles = runner.FinishProfiles();
    if (args.scope != "mask") {
      WriteExact(args.output_raw, logits);
    } else {
      WriteExact(args.output_raw, mask);
    }
    if (std::filesystem::exists(args.latency_csv)) {
      throw std::runtime_error("refusing to overwrite latency CSV");
    }
    std::ofstream csv(args.latency_csv, std::ios::trunc);
    csv << "iteration,scope,started_unix_ns,ended_unix_ns,latency_ms,cpu_before,cpu_after,affinity,error,retry\n"
        << std::setprecision(17);
    for (std::size_t index = 0; index < calls.size(); ++index) {
      const auto& call = calls[index];
      csv << index << ',' << args.scope << ',' << call.started_unix_ns << ','
          << call.ended_unix_ns << ',' << call.latency_ms << ',' << call.cpu_before << ','
          << call.cpu_after << ',' << affinity << ",,0\n";
    }
    if (!csv) {
      throw std::runtime_error("latency CSV write failed");
    }
    if (!args.stability_signature_csv.empty()) {
      if (std::filesystem::exists(args.stability_signature_csv)) {
        throw std::runtime_error("refusing to overwrite stability signature CSV");
      }
      if (signatures.empty() ||
          !std::all_of(signatures.begin(), signatures.end(),
                       [&](std::uint64_t value) { return value == signatures.front(); })) {
        throw std::runtime_error("prediction drift detected");
      }
      std::ofstream signature_csv(args.stability_signature_csv, std::ios::trunc);
      signature_csv << "iteration,signature,non_finite_count\n";
      for (std::size_t index = 0; index < signatures.size(); ++index) {
        signature_csv << index << ',' << signatures[index] << ",0\n";
      }
      if (!signature_csv) {
        throw std::runtime_error("stability signature CSV write failed");
      }
    }
    const auto process_finished = std::chrono::steady_clock::now();
    const auto milliseconds = [](auto started, auto ended) {
      return std::chrono::duration<double, std::milli>(ended - started).count();
    };
    std::cout << "status=passed\n"
              << "variant=" << plan.variant << '\n'
              << "sessions=" << runner.session_count() << '\n'
              << "scope=" << args.scope << '\n'
              << "measurements=" << calls.size() << '\n'
              << "affinity=" << affinity << '\n'
              << "manifest_ms=" << milliseconds(process_started, manifest_loaded) << '\n'
              << "session_load_ms=" << milliseconds(manifest_loaded, sessions_loaded) << '\n'
              << "runner_init_ms=" << milliseconds(sessions_loaded, runner_initialized) << '\n'
              << "first_inference_ms=" << calls.front().latency_ms << '\n'
              << "total_ms=" << milliseconds(process_started, process_finished) << '\n';
    for (std::size_t index = 0; index < profiles.size(); ++index) {
      std::cout << "profile_" << index << '=' << profiles[index] << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "status=failed\nerror=" << error.what() << '\n';
    return 2;
  }
}

#pragma once

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace flashinfer {
namespace kernel_trace {

inline bool is_truthy(char const* value) {
  if (value == nullptr) return false;
  std::string v(value);
  return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "YES" || v == "on" ||
         v == "ON";
}

inline std::string env_or(char const* name, char const* fallback) {
  char const* value = std::getenv(name);
  return value == nullptr ? std::string(fallback) : std::string(value);
}

inline char const* first_env(char const* primary, char const* legacy) {
  char const* value = std::getenv(primary);
  if (value != nullptr) return value;
  return std::getenv(legacy);
}

inline bool enabled() {
  static bool const value = is_truthy(std::getenv("FLASHINFER_KERNEL_TRACE")) ||
                            is_truthy(std::getenv("FLASHINFER_TRTLLM_MOE_TRACE"));
  return value;
}

inline std::string rank() {
  char const* value = std::getenv("RANK");
  if (value != nullptr) return std::string(value);
  return env_or("SLURM_PROCID", "0");
}

inline std::string local_rank() {
  char const* value = std::getenv("LOCAL_RANK");
  if (value != nullptr) return std::string(value);
  return env_or("SLURM_LOCALID", "0");
}

inline std::string hostname() {
  char buffer[256];
  if (gethostname(buffer, sizeof(buffer)) != 0) {
    return "unknown";
  }
  buffer[sizeof(buffer) - 1] = '\0';
  return std::string(buffer);
}

inline void replace_all(std::string& text, std::string const& key, std::string const& value) {
  std::size_t pos = 0;
  while ((pos = text.find(key, pos)) != std::string::npos) {
    text.replace(pos, key.size(), value);
    pos += value.size();
  }
}

inline std::string trace_path() {
  char const* configured_path =
      first_env("FLASHINFER_KERNEL_TRACE_FILE", "FLASHINFER_TRTLLM_MOE_TRACE_FILE");
  std::string path;
  if (configured_path != nullptr) {
    path = configured_path;
  } else if (is_truthy(std::getenv("FLASHINFER_KERNEL_TRACE")) &&
             !is_truthy(std::getenv("FLASHINFER_TRTLLM_MOE_TRACE"))) {
    path = "/tmp/flashinfer_kernel_trace.rank%r.local%l.pid%p.jsonl";
  } else {
    path = "/tmp/flashinfer_trtllm_moe_trace.rank%r.local%l.pid%p.jsonl";
  }
  replace_all(path, "%p", std::to_string(getpid()));
  replace_all(path, "%r", rank());
  replace_all(path, "%l", local_rank());
  replace_all(path, "%h", hostname());
  return path;
}

inline std::string trace_mode() {
  char const* value = first_env("FLASHINFER_KERNEL_TRACE_MODE", "FLASHINFER_TRTLLM_MOE_TRACE_MODE");
  return value == nullptr ? std::string("shape_once") : std::string(value);
}

inline std::string trace_stage() {
  char const* value =
      first_env("FLASHINFER_KERNEL_TRACE_STAGE", "FLASHINFER_TRTLLM_MOE_TRACE_STAGE");
  return value == nullptr ? std::string("unknown") : std::string(value);
}

inline std::vector<std::string> parse_stages(char const* value) {
  std::vector<std::string> stages;
  if (value == nullptr) return stages;

  std::string const configured(value);
  std::size_t start = 0;
  while (start <= configured.size()) {
    std::size_t const end = configured.find(',', start);
    std::string stage = configured.substr(start, end - start);
    std::size_t const first = stage.find_first_not_of(" \t\n\r\f\v");
    if (first != std::string::npos) {
      std::size_t const last = stage.find_last_not_of(" \t\n\r\f\v");
      stages.push_back(stage.substr(first, last - first + 1));
    }
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return stages;
}

inline bool stage_enabled(std::string const& stage) {
  static std::vector<std::string> const allowed_stages = parse_stages(
      first_env("FLASHINFER_KERNEL_TRACE_STAGES", "FLASHINFER_TRTLLM_MOE_TRACE_STAGES"));
  if (allowed_stages.empty()) return true;
  for (std::string const& allowed_stage : allowed_stages) {
    if (stage == allowed_stage) return true;
  }
  return false;
}

inline bool should_trace_current_stage() { return enabled() && stage_enabled(trace_stage()); }

inline bool dedupe_enabled() { return trace_mode() == "shape_once"; }

inline int64_t first_n_limit() {
  if (trace_mode() != "first_n") {
    return -1;
  }

  char const* value =
      first_env("FLASHINFER_KERNEL_TRACE_FIRST_N", "FLASHINFER_TRTLLM_MOE_TRACE_FIRST_N");
  if (value == nullptr) {
    return 10;
  }

  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return 10;
  }
  return parsed < 0 ? 0 : parsed;
}

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline std::string json_escape(std::string const& value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
    }
  }
  return out.str();
}

inline std::string quote(std::string const& value) { return "\"" + json_escape(value) + "\""; }

template <typename T>
inline std::string json_array(std::vector<T> const& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) out << ",";
    out << values[i];
  }
  out << "]";
  return out.str();
}

inline void append_body(std::string const& body, std::string const& dedupe_key = "") {
  if (!enabled()) return;

  std::string const stage = trace_stage();
  if (!stage_enabled(stage)) return;

  static std::mutex mutex;
  static std::unordered_set<std::string> seen;
  static int64_t event_count = 0;
  std::lock_guard<std::mutex> guard(mutex);

  if (dedupe_enabled() && !dedupe_key.empty()) {
    std::string const staged_dedupe_key = stage + ":" + dedupe_key;
    if (seen.find(staged_dedupe_key) != seen.end()) {
      return;
    }
    seen.insert(staged_dedupe_key);
  }

  int64_t const limit = first_n_limit();
  if (limit >= 0) {
    if (event_count >= limit) {
      return;
    }
    ++event_count;
  }

  std::ofstream out(trace_path(), std::ios::app);
  if (!out) return;

  out << "{\"schema_version\":2"
      << ",\"source\":\"flashinfer_cpp\""
      << ",\"ts_ns\":" << now_ns() << ",\"pid\":" << getpid()
      << ",\"hostname\":" << quote(hostname()) << ",\"rank\":" << quote(rank())
      << ",\"local_rank\":" << quote(local_rank()) << ",\"trace_stage\":" << quote(stage) << ","
      << body << "}\n";
}

}  // namespace kernel_trace
}  // namespace flashinfer

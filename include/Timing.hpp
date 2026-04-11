#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace laplace {

class Timer {
 public:
  using clock = std::chrono::steady_clock;

  Timer() : start_(clock::now()) {}

  void reset() {
    start_ = clock::now();
  }

  double elapsedSeconds() const {
    return std::chrono::duration<double>(clock::now() - start_).count();
  }

 private:
  clock::time_point start_;
};

struct TimingEntry {
  std::string name;
  double seconds = 0.0;
};

class TimingReport {
 public:
  void add(const std::string& name, const double seconds) {
    entries_.push_back(TimingEntry{name, seconds});
  }

  const std::vector<TimingEntry>& entries() const {
    return entries_;
  }

  double totalSeconds() const {
    double total = 0.0;
    for (const auto& entry : entries_) {
      total += entry.seconds;
    }
    return total;
  }

 private:
  std::vector<TimingEntry> entries_;
};

}  // namespace laplace

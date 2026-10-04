// Progress reports and stop requests of a native call (sc_progress).

#pragma once

#include <algorithm>

#include "stitchcore.h"

namespace stitchcore {

/// Thrown by Monitor::check after a stop. Deliberately not a std::exception, so that the solvers' catch clauses
/// for cv::Exception and std::exception let it through to the C entry point.
struct Cancelled {};

/// Reports and stop requests of one native call; lives on the calling thread for the whole call.
class Monitor {
public:
    explicit Monitor(const sc_progress *progress)
        : report_(progress ? progress->report : nullptr), context_(progress ? progress->context : nullptr) {}
    Monitor(const Monitor &) = delete;
    Monitor &operator=(const Monitor &) = delete;

    bool active() const { return report_ != nullptr; }
    bool stopped() const { return stopped_; }

    /// Share of the call that the fractions passed from now on cover.
    void range(double from, double to) {
        from_ = from;
        to_ = to;
    }

    /// Reports from + (to - from) * fraction, never below the last report; false once a stop was asked, after
    /// which the callback is not called again.
    bool poll(double fraction) {
        if (!report_) return true;
        if (stopped_) return false;
        last_ = std::max(last_, from_ + (to_ - from_) * std::clamp(fraction, 0.0, 1.0));
        stopped_ = report_(context_, last_) != 0;
        return !stopped_;
    }

    /// poll, throwing Cancelled once a stop was asked.
    void check(double fraction) {
        if (!poll(fraction)) throw Cancelled{};
    }

private:
    sc_progress_fn report_;
    void *context_;
    double from_ = 0, to_ = 1, last_ = 0;
    bool stopped_ = false;
};

}  // namespace stitchcore

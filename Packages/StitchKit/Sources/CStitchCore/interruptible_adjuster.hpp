/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// OpenCV 5.0.0's bundle adjusters (BundleAdjusterRay, BundleAdjusterReproj) with a Levenberg-Marquardt loop that
// reports its progress and stops when asked. estimate() and the two calcJacobian() are copied from
// modules/stitching/src/motion_estimators.cpp; calcError, setUpInitialCameraParams and obtainRefinedCameraParams
// stay OpenCV's compiled code. The only floating-point expressions compiled here are eps * eps, val - step,
// val + step, 2 * step and (e2 - e1) / h, none of which can be contracted, so the iterates are the same bit for bit.
//
// The Jacobian differs in one way: OpenCV evaluated the whole error twice for every parameter of every camera,
// although moving camera i changes only the rows of the pairs it is in. Here calcError runs on those pairs alone
// (edges_ and total_num_matches_ narrowed for the call: it computes each pair from its own two cameras), and
// every other row gets (e - e) / h from one evaluation at the current parameters, which is what the two whole
// evaluations gave it, NaN included. On 51 photos joined by 65 pairs that is about twenty times less work.

#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include "progress.hpp"

static_assert(CV_VERSION_MAJOR == 5 && CV_VERSION_MINOR == 0 && CV_VERSION_REVISION == 0,
              "Interruptible copies estimate() and calcJacobian() of OpenCV 5.0.0: compare them with the new version");

namespace stitchcore {

// The adjusters' steps are private overrides of protected virtuals of the base. Pointers to those members, formed
// in a class derived from the base ([class.protected]), call them by virtual dispatch.
struct AdjusterSteps : cv::detail::BundleAdjusterBase {
    using Base = cv::detail::BundleAdjusterBase;
    static void setUp(Base &b, const std::vector<cv::detail::CameraParams> &c) {
        (b.*&AdjusterSteps::setUpInitialCameraParams)(c);
    }
    static void obtain(const Base &b, std::vector<cv::detail::CameraParams> &c) {
        (b.*&AdjusterSteps::obtainRefinedCameraParams)(c);
    }
    static void error(Base &b, cv::Mat &e) { (b.*&AdjusterSteps::calcError)(e); }
};

// OpenCV's adjuster `Stock` whose Levenberg-Marquardt loop reports to `monitor` and stops when asked: a stop
// throws Cancelled from estimate() before the cameras are written.
template <class Stock> class Interruptible final : public Stock {
public:
    explicit Interruptible(Monitor &monitor) : monitor_(monitor) {}

protected:
    bool estimate(const std::vector<cv::detail::ImageFeatures> &features,
                  const std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                  std::vector<cv::detail::CameraParams> &cameras) override {
        this->num_images_ = static_cast<int>(features.size());
        this->features_ = &features[0];
        this->pairwise_matches_ = &pairwise_matches[0];

        AdjusterSteps::setUp(*this, cameras);

        // Leave only consistent image pairs
        this->edges_.clear();
        for (int i = 0; i < this->num_images_ - 1; ++i) {
            for (int j = i + 1; j < this->num_images_; ++j) {
                const cv::detail::MatchesInfo &matches_info = this->pairwise_matches_[i * this->num_images_ + j];
                if (matches_info.confidence > this->conf_thresh_) this->edges_.push_back(std::make_pair(i, j));
            }
        }

        // Compute number of correspondences
        this->total_num_matches_ = 0;
        for (size_t i = 0; i < this->edges_.size(); ++i)
            this->total_num_matches_ += static_cast<int>(
                pairwise_matches[this->edges_[i].first * this->num_images_ + this->edges_[i].second].num_inliers);

        int nerrs = this->total_num_matches_ * this->num_errs_per_measurement_;

        auto callb = [&](cv::InputOutputArray param, cv::OutputArray err, cv::OutputArray jac) -> bool {
            if (!monitor_.poll(fraction(evaluations_ / 2.0))) return false;
            // workaround against losing value
            cv::Mat backup = this->cam_params_.clone();
            param.copyTo(this->cam_params_);
            if (jac.needed()) {
                cv::Mat m = jac.getMat();
                calcJacobian(m);
            }
            if (err.needed() && !monitor_.stopped()) {
                cv::Mat m = err.getMat();
                AdjusterSteps::error(*this, m);
            }
            backup.copyTo(this->cam_params_);
            if (!jac.needed()) ++evaluations_;
            return monitor_.poll(fraction(evaluations_ / 2.0));
        };

        cv::LevMarq solver(this->cam_params_, callb,
                           cv::LevMarq::Settings()
                               .setMaxIterations((unsigned int)this->term_criteria_.maxCount)
                               .setStepNormTolerance(this->term_criteria_.epsilon)
                               .setSmallEnergyTolerance(this->term_criteria_.epsilon * this->term_criteria_.epsilon)
                               .setGeodesic(true),
                           cv::noArray(), cv::MatrixType::AUTO, cv::VariableType::LINEAR, nerrs);
        solver.optimize();
        if (monitor_.stopped()) throw Cancelled{};

        // Check if all camera parameters are valid
        bool ok = true;
        for (int i = 0; i < this->cam_params_.rows; ++i) {
            if (cvIsNaN(this->cam_params_.template at<double>(i, 0))) {
                ok = false;
                break;
            }
        }
        if (!ok) return false;

        AdjusterSteps::obtain(*this, cameras);

        // Normalize motion to center image
        cv::detail::Graph span_tree;
        std::vector<int> span_tree_centers;
        cv::detail::findMaxSpanningTree(this->num_images_, pairwise_matches, span_tree, span_tree_centers);
        cv::Mat R_inv = cameras[span_tree_centers[0]].R.inv();
        for (int i = 0; i < this->num_images_; ++i) cameras[i].R = R_inv * cameras[i].R;
        return true;
    }

private:
    void calcJacobian(cv::Mat &jac) override;

    // Probes of the loop so far, out of its iteration cap.
    double fraction(double probes) const { return std::min(1.0, probes / this->term_criteria_.maxCount); }

    // Before the perturbation of camera `i` of a Jacobian: false after a stop, with nothing perturbed yet.
    bool camera(int i) { return monitor_.poll(fraction(evaluations_ / 2.0 + double(i) / this->num_images_)); }

    // The pairs of each camera, where each pair's rows start in the whole error and how many it has.
    struct Pairs {
        std::vector<std::vector<std::pair<int, int>>> edges;
        std::vector<std::vector<int>> first, rows;
        std::vector<int> matches;
    };

    Pairs pairs_of_cameras() const {
        Pairs pairs;
        pairs.edges.assign(this->num_images_, {});
        pairs.first.assign(this->num_images_, {});
        pairs.rows.assign(this->num_images_, {});
        pairs.matches.assign(this->num_images_, 0);
        int row = 0;
        for (const auto &edge : this->edges_) {
            // calcError writes num_errs_per_measurement_ rows for each inlier of the pair.
            const cv::detail::MatchesInfo &info = this->pairwise_matches_[edge.first * this->num_images_ + edge.second];
            int inliers = 0;
            for (size_t k = 0; k < info.matches.size(); ++k) inliers += info.inliers_mask[k] ? 1 : 0;
            for (int camera : {edge.first, edge.second}) {
                pairs.edges[camera].push_back(edge);
                pairs.first[camera].push_back(row);
                pairs.rows[camera].push_back(inliers * this->num_errs_per_measurement_);
                pairs.matches[camera] += inliers;
            }
            row += inliers * this->num_errs_per_measurement_;
        }
        return pairs;
    }

    // calcError on the pairs of `camera` alone, into `err`.
    void camera_error(const Pairs &pairs, int camera, cv::Mat &err) {
        std::swap(this->edges_, narrowed_);
        this->edges_ = pairs.edges[camera];
        const int total = this->total_num_matches_;
        this->total_num_matches_ = pairs.matches[camera];
        struct Restore {
            Interruptible &self;
            int total;
            ~Restore() {
                std::swap(self.edges_, self.narrowed_);
                self.total_num_matches_ = total;
            }
        } restore{*this, total};
        AdjusterSteps::error(*this, err);
    }

    // Column `column` of the Jacobian for cam_params_ entry `param` of `camera`: central differences with `step`.
    void derivative(cv::Mat &jac, const Pairs &pairs, int camera, int param, int column, double step) {
        const double val = this->cam_params_.template at<double>(param, 0);
        this->cam_params_.template at<double>(param, 0) = val - step;
        camera_error(pairs, camera, before_);
        this->cam_params_.template at<double>(param, 0) = val + step;
        camera_error(pairs, camera, after_);
        this->cam_params_.template at<double>(param, 0) = val;
        const double h = 2 * step;
        cv::Mat res = jac.col(column);
        for (int r = 0; r < current_.rows; ++r)
            res.at<double>(r, 0) = (current_.at<double>(r, 0) - current_.at<double>(r, 0)) / h;
        int local = 0;
        for (size_t e = 0; e < pairs.edges[camera].size(); ++e) {
            const int start = pairs.first[camera][e];
            for (int q = 0; q < pairs.rows[camera][e]; ++q, ++local)
                res.at<double>(start + q, 0) = (after_.at<double>(local, 0) - before_.at<double>(local, 0)) / h;
        }
    }

    Monitor &monitor_;
    int evaluations_ = 0;     // error evaluations without a Jacobian: two per probe with geodesic acceleration
    cv::Mat before_, after_;  // the stock adjusters' err1_ and err2_ are private
    cv::Mat current_;         // the whole error at the parameters of the Jacobian
    std::vector<std::pair<int, int>> narrowed_;  // the whole edges_ while calcError runs on one camera's pairs
};

template <> inline void Interruptible<cv::detail::BundleAdjusterRay>::calcJacobian(cv::Mat &jac) {
    jac.create(this->total_num_matches_ * 3, this->num_images_ * 4, CV_64F);

    const double step = 1e-3;
    const Pairs pairs = pairs_of_cameras();
    AdjusterSteps::error(*this, current_);

    for (int i = 0; i < this->num_images_; ++i) {
        if (!camera(i)) return;
        for (int j = 0; j < 4; ++j) derivative(jac, pairs, i, i * 4 + j, i * 4 + j, step);
    }
}

template <> inline void Interruptible<cv::detail::BundleAdjusterReproj>::calcJacobian(cv::Mat &jac) {
    jac.create(this->total_num_matches_ * 2, this->num_images_ * 7, CV_64F);
    jac.setTo(0);

    const double step = 1e-4;
    const cv::Mat mask = this->refinement_mask_;
    const Pairs pairs = pairs_of_cameras();
    AdjusterSteps::error(*this, current_);

    for (int i = 0; i < this->num_images_; ++i) {
        if (!camera(i)) return;
        if (mask.at<uchar>(0, 0)) derivative(jac, pairs, i, i * 7, i * 7, step);
        if (mask.at<uchar>(0, 2)) derivative(jac, pairs, i, i * 7 + 1, i * 7 + 1, step);
        if (mask.at<uchar>(1, 2)) derivative(jac, pairs, i, i * 7 + 2, i * 7 + 2, step);
        if (mask.at<uchar>(1, 1)) derivative(jac, pairs, i, i * 7 + 3, i * 7 + 3, step);
        for (int j = 4; j < 7; ++j) derivative(jac, pairs, i, i * 7 + j, i * 7 + j, step);
    }
}

}  // namespace stitchcore

#include "robot.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <sstream>
#include <stdexcept>

namespace pinokin {

Robot::Robot(const std::string& urdf_path, const std::string& ee_frame) {
    pinocchio::urdf::buildModel(urdf_path, model_);
    data_ = pinocchio::Data(model_);
    init_ee_frame(ee_frame);
}

Robot Robot::from_urdf_string(const std::string& urdf_str,
                              const std::string& ee_frame) {
    Robot r;
    std::istringstream stream(urdf_str);
    pinocchio::urdf::buildModelFromXML(urdf_str, r.model_);
    r.data_ = pinocchio::Data(r.model_);
    r.init_ee_frame(ee_frame);
    return r;
}

void Robot::init_ee_frame(const std::string& ee_frame) {
    if (ee_frame.empty()) {
        // Default: last frame in model
        if (model_.nframes == 0) {
            throw std::runtime_error("URDF model has no frames");
        }
        ee_frame_id_ = static_cast<pinocchio::FrameIndex>(model_.nframes - 1);
    } else {
        set_ee_frame(ee_frame);
    }
}

void Robot::set_ee_frame(const std::string& name) {
    if (!model_.existFrame(name)) {
        throw std::runtime_error("Frame '" + name + "' not found in model");
    }
    ee_frame_id_ = model_.getFrameId(name);
}

void Robot::set_tool_transform(const Eigen::Matrix4d& T_tool) {
    T_tool_ = T_tool;
    tool_offset_ = T_tool.block<3, 1>(0, 3);
    has_tool_ = true;
}

void Robot::clear_tool_transform() {
    T_tool_ = Eigen::Matrix4d::Identity();
    tool_offset_ = Eigen::Vector3d::Zero();
    has_tool_ = false;
}

Eigen::Matrix4d Robot::fkine(const Eigen::VectorXd& q) const {
    pinocchio::framesForwardKinematics(model_, data_, q);
    Eigen::Matrix4d T = data_.oMf[ee_frame_id_].toHomogeneousMatrix();
    if (has_tool_) {
        T = T * T_tool_;
    }
    return T;
}

void Robot::jacob0(const Eigen::VectorXd& q, Eigen::Ref<Eigen::MatrixXd> J) const {
    J.setZero();
    // LOCAL_WORLD_ALIGNED: world-frame orientation, referenced at the frame origin.
    pinocchio::computeFrameJacobian(model_, data_, q, ee_frame_id_,
                                    pinocchio::LOCAL_WORLD_ALIGNED, J);
    if (has_tool_) {
        // v_tool = v_ee + omega x (R_ee * p_tool)
        // J_v_tool = J_v_ee - skew(R_ee * p_tool) * J_w
        pinocchio::framesForwardKinematics(model_, data_, q);
        Eigen::Matrix3d R_ee = data_.oMf[ee_frame_id_].rotation();
        Eigen::Vector3d r = R_ee * tool_offset_;
        Eigen::Matrix3d skew_r;
        skew_r <<     0, -r(2),  r(1),
                   r(2),     0, -r(0),
                  -r(1),  r(0),     0;
        J.topRows(3) -= skew_r * J.bottomRows(3);
    }
}

void Robot::jacobe(const Eigen::VectorXd& q, Eigen::Ref<Eigen::MatrixXd> J) const {
    J.setZero();
    pinocchio::computeFrameJacobian(model_, data_, q, ee_frame_id_,
                                    pinocchio::LOCAL, J);
    if (has_tool_) {
        // In the ee frame, the tool offset is just p_tool (no rotation needed).
        // J_v_tool = J_v_ee - skew(p_tool) * J_w
        Eigen::Matrix3d skew_p;
        skew_p <<          0, -tool_offset_(2),  tool_offset_(1),
                  tool_offset_(2),            0, -tool_offset_(0),
                 -tool_offset_(1),  tool_offset_(0),            0;
        J.topRows(3) -= skew_p * J.bottomRows(3);
    }
}

std::vector<Eigen::Matrix4d> Robot::batch_fk(const Eigen::MatrixXd& joint_positions) const {
    const int n_configs = static_cast<int>(joint_positions.rows());
    std::vector<Eigen::Matrix4d> results(n_configs);

    for (int i = 0; i < n_configs; ++i) {
        Eigen::VectorXd q = joint_positions.row(i).transpose();
        pinocchio::framesForwardKinematics(model_, data_, q);
        results[i] = data_.oMf[ee_frame_id_].toHomogeneousMatrix();
        if (has_tool_) {
            results[i] = results[i] * T_tool_;
        }
    }

    return results;
}

// Private default constructor for from_urdf_string
Robot::Robot() : ee_frame_id_(0) {}

} // namespace pinokin

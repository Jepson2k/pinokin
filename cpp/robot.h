#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace pinokin {

class Robot {
public:
    explicit Robot(const std::string& urdf_path, const std::string& ee_frame = "");
    static Robot from_urdf_string(const std::string& urdf_str,
                                  const std::string& ee_frame = "");

    Eigen::Matrix4d fkine(const Eigen::VectorXd& q) const;
    void fkine_into(const Eigen::VectorXd& q, Eigen::Ref<Eigen::Matrix4d> out) const;

    // World-frame Jacobian with [linear; angular] row ordering
    void jacob0(const Eigen::VectorXd& q, Eigen::Ref<Eigen::MatrixXd> J) const;

    // End-effector-frame Jacobian with [linear; angular] row ordering
    void jacobe(const Eigen::VectorXd& q, Eigen::Ref<Eigen::MatrixXd> J) const;

    // Batch FK: N joint configs -> N 4x4 matrices (loop in C++)
    std::vector<Eigen::Matrix4d> batch_fk(const Eigen::MatrixXd& joint_positions) const;

    const std::string& name() const { return model_.name; }
    int nq() const { return static_cast<int>(model_.nq); }
    const Eigen::VectorXd& lower_limits() const { return model_.lowerPositionLimit; }
    const Eigen::VectorXd& upper_limits() const { return model_.upperPositionLimit; }
    const Eigen::VectorXd& velocity_limits() const { return model_.velocityLimit; }
    void set_ee_frame(const std::string& name);

    void set_tool_transform(const Eigen::Matrix4d& T_tool);
    void clear_tool_transform();
    bool has_tool_transform() const { return has_tool_; }
    const Eigen::Matrix4d& tool_transform() const { return T_tool_; }
    const Eigen::Vector3d& tool_offset() const { return tool_offset_; }

    const pinocchio::Model& model() const { return model_; }
    pinocchio::Data& data() const { return data_; }
    pinocchio::FrameIndex ee_frame_id() const { return ee_frame_id_; }

private:
    Robot();  // used by from_urdf_string
    void init_ee_frame(const std::string& ee_frame);

    pinocchio::Model model_;
    mutable pinocchio::Data data_;
    pinocchio::FrameIndex ee_frame_id_;
    bool has_tool_ = false;
    Eigen::Matrix4d T_tool_ = Eigen::Matrix4d::Identity();
    Eigen::Vector3d tool_offset_ = Eigen::Vector3d::Zero();
};

} // namespace pinokin

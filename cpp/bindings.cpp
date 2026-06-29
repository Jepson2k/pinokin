#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>

#include "robot.h"
#include "ik_solver.h"
#include "collision_checker.h"

namespace nb = nanobind;
using namespace pinokin;

NB_MODULE(_core, m) {
    m.doc() = "pinokin: FK, Jacobians, IK, and collision for URDF robots via Pinocchio";

    nb::class_<Robot>(m, "Robot")
        .def(nb::init<const std::string&, const std::string&>(),
             nb::arg("urdf_path"), nb::arg("ee_frame") = "")
        .def_static("from_urdf_string", &Robot::from_urdf_string,
                     nb::arg("urdf_string"), nb::arg("ee_frame") = "")
        .def("fkine", &Robot::fkine, nb::arg("q"))
        .def("fkine_into", &Robot::fkine_into, nb::arg("q"), nb::arg("out"))
        .def("jacob0", [](const Robot& r, const Eigen::VectorXd& q) {
            Eigen::MatrixXd J(6, r.nq());
            r.jacob0(q, J);
            return J;
        }, nb::arg("q"))
        .def("jacob0_into", [](const Robot& r, const Eigen::VectorXd& q,
                               Eigen::Ref<Eigen::MatrixXd> out) {
            r.jacob0(q, out);
        }, nb::arg("q"), nb::arg("out"))
        .def("jacobe", [](const Robot& r, const Eigen::VectorXd& q) {
            Eigen::MatrixXd J(6, r.nq());
            r.jacobe(q, J);
            return J;
        }, nb::arg("q"))
        .def("batch_fk", &Robot::batch_fk, nb::arg("joint_positions"))
        .def_prop_ro("name", &Robot::name)
        .def_prop_ro("nq", &Robot::nq)
        .def_prop_ro("lower_limits", &Robot::lower_limits,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("upper_limits", &Robot::upper_limits,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("velocity_limits", &Robot::velocity_limits,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("qlim", [](const Robot& r) {
            Eigen::MatrixXd qlim(2, r.nq());
            qlim.row(0) = r.lower_limits();
            qlim.row(1) = r.upper_limits();
            return qlim;
        })
        .def("set_ee_frame", &Robot::set_ee_frame, nb::arg("name"))
        .def("set_tool_transform", &Robot::set_tool_transform, nb::arg("T_tool"))
        .def("clear_tool_transform", &Robot::clear_tool_transform)
        .def_prop_ro("has_tool_transform", &Robot::has_tool_transform);

    nb::enum_<IKSolver::Method>(m, "Method")
        .value("GN", IKSolver::Method::GN)
        .value("NR", IKSolver::Method::NR)
        .value("LM", IKSolver::Method::LM);

    nb::enum_<IKSolver::Damping>(m, "Damping")
        .value("Chan", IKSolver::Damping::Chan)
        .value("Wampler", IKSolver::Damping::Wampler)
        .value("Sugihara", IKSolver::Damping::Sugihara);

    nb::class_<IKSolver::BatchResult>(m, "BatchResult")
        .def_ro("joint_positions", &IKSolver::BatchResult::joint_positions)
        .def_ro("valid", &IKSolver::BatchResult::valid)
        .def_ro("all_valid", &IKSolver::BatchResult::all_valid);

    nb::class_<IKSolver>(m, "IKSolver")
        .def(nb::init<const Robot&, IKSolver::Method, IKSolver::Damping,
                       double, double, int, int, bool>(),
             nb::arg("robot"),
             nb::arg("method") = IKSolver::Method::LM,
             nb::arg("damping") = IKSolver::Damping::Sugihara,
             nb::arg("tol") = 1e-6,
             nb::arg("lm_lambda") = 1.0,
             nb::arg("max_iter") = 30,
             nb::arg("max_restarts") = 100,
             nb::arg("enforce_limits") = true,
             nb::keep_alive<1, 2>())
        .def("solve",
             [](IKSolver& s, const Eigen::Matrix4d& Tep,
                nb::object q0_obj) -> bool {
                 if (q0_obj.is_none()) {
                     return s.solve(Tep);
                 }
                 Eigen::VectorXd q0 = nb::cast<Eigen::VectorXd>(q0_obj);
                 return s.solve(Tep, &q0);
             },
             nb::arg("Tep"), nb::arg("q0") = nb::none())
        .def("batch_ik", &IKSolver::batch_ik,
             nb::arg("poses"), nb::arg("q_start"),
             nb::arg("stop_on_failure") = false)
        .def("set_we", &IKSolver::set_we, nb::arg("we"))
        .def_prop_ro("q", &IKSolver::q,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("success", &IKSolver::success)
        .def_prop_ro("residual", &IKSolver::residual)
        .def_prop_ro("iterations", &IKSolver::iterations)
        .def_prop_ro("restarts", &IKSolver::restarts);

    nb::class_<CollisionChecker>(m, "CollisionChecker")
        .def(nb::init<const Robot&, const std::string&,
                       const std::vector<std::string>&, bool, bool, double>(),
             nb::arg("robot"),
             nb::arg("urdf_path"),
             nb::arg("package_dirs") = std::vector<std::string>{},
             nb::arg("add_all_pairs") = true,
             nb::arg("remove_adjacent_pairs") = true,
             nb::arg("clearance_margin") = 0.0,
             nb::keep_alive<1, 2>())
        .def("load_srdf", &CollisionChecker::load_srdf, nb::arg("srdf_path"))
        .def("add_collision_pair", &CollisionChecker::add_collision_pair,
             nb::arg("first"), nb::arg("second"))
        .def("remove_collision_pair", &CollisionChecker::remove_collision_pair,
             nb::arg("first"), nb::arg("second"))
        .def("in_collision", &CollisionChecker::in_collision, nb::arg("q"))
        .def("colliding_pairs", &CollisionChecker::colliding_pairs, nb::arg("q"))
        .def("min_distance", &CollisionChecker::min_distance, nb::arg("q"))
        .def("check_segment", &CollisionChecker::check_segment,
             nb::arg("q0"), nb::arg("q1"), nb::arg("n_steps"),
             nb::arg("include_endpoints") = true)
        .def("check_path", &CollisionChecker::check_path, nb::arg("q_path"))
        .def("set_clearance_margin", &CollisionChecker::set_clearance_margin,
             nb::arg("margin"))
        .def_prop_ro("clearance_margin", &CollisionChecker::clearance_margin)
        .def("add_obstacle", &CollisionChecker::add_obstacle,
             nb::arg("name"), nb::arg("kind"), nb::arg("params"),
             nb::arg("world_pose"))
        .def("add_obstacle_box", &CollisionChecker::add_obstacle_box,
             nb::arg("name"), nb::arg("half_extents"), nb::arg("world_pose"))
        .def("add_obstacle_sphere", &CollisionChecker::add_obstacle_sphere,
             nb::arg("name"), nb::arg("radius"), nb::arg("world_pose"))
        .def("add_obstacle_cylinder", &CollisionChecker::add_obstacle_cylinder,
             nb::arg("name"), nb::arg("radius"), nb::arg("length"),
             nb::arg("world_pose"))
        .def("add_obstacle_mesh", &CollisionChecker::add_obstacle_mesh,
             nb::arg("name"), nb::arg("mesh_path"), nb::arg("world_pose"),
             nb::arg("mesh_scale") = Eigen::Vector3d::Ones())
        .def("attach_box_to_frame", &CollisionChecker::attach_box_to_frame,
             nb::arg("name"), nb::arg("half_extents"),
             nb::arg("parent_frame"), nb::arg("placement"))
        .def("attach_sphere_to_frame", &CollisionChecker::attach_sphere_to_frame,
             nb::arg("name"), nb::arg("radius"),
             nb::arg("parent_frame"), nb::arg("placement"))
        .def("attach_cylinder_to_frame", &CollisionChecker::attach_cylinder_to_frame,
             nb::arg("name"), nb::arg("radius"), nb::arg("length"),
             nb::arg("parent_frame"), nb::arg("placement"))
        .def("attach_mesh_to_frame", &CollisionChecker::attach_mesh_to_frame,
             nb::arg("name"), nb::arg("mesh_path"),
             nb::arg("parent_frame"), nb::arg("placement"),
             nb::arg("mesh_scale") = Eigen::Vector3d::Ones())
        .def("set_geometry_pose_by_name",
             &CollisionChecker::set_geometry_pose_by_name,
             nb::arg("name"), nb::arg("pose"))
        .def("geometry_world_pose",
             &CollisionChecker::geometry_world_pose, nb::arg("name"))
        .def("remove_geometry_by_name",
             &CollisionChecker::remove_geometry_by_name, nb::arg("name"))
        .def("reparent_geometry_by_name",
             &CollisionChecker::reparent_geometry_by_name,
             nb::arg("name"), nb::arg("new_parent_frame"),
             nb::arg("new_placement"))
        .def("update_placements", &CollisionChecker::update_placements,
             nb::arg("q"))
        .def_prop_ro("num_collision_pairs",
                     &CollisionChecker::num_collision_pairs)
        .def_prop_ro("num_geometry_objects",
                     &CollisionChecker::num_geometry_objects)
        .def_prop_ro("geometry_names",
                     &CollisionChecker::geometry_names)
        .def("has_geometry", &CollisionChecker::has_geometry, nb::arg("name"));
}

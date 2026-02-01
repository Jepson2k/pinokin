#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "robot.h"
#include "ik_solver.h"

namespace nb = nanobind;
using namespace pinokin;

NB_MODULE(_core, m) {
    m.doc() = "pinokin: FK, Jacobians, and IK for URDF robots via Pinocchio";

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
             nb::keep_alive<1, 2>())  // IKSolver refs Robot
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
             nb::arg("poses"), nb::arg("q_start"))
        .def_static("unwrap_angles", &IKSolver::unwrap_angles,
                     nb::arg("q_solution"), nb::arg("q_current"))
        .def("set_we", &IKSolver::set_we, nb::arg("we"))
        .def_prop_ro("q", &IKSolver::q,
                     nb::rv_policy::reference_internal)
        .def_prop_ro("success", &IKSolver::success)
        .def_prop_ro("residual", &IKSolver::residual)
        .def_prop_ro("iterations", &IKSolver::iterations)
        .def_prop_ro("restarts", &IKSolver::restarts);
}

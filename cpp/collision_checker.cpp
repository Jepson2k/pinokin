#include "collision_checker.h"
#include "robot.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/spatial/se3.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

// Pinocchio 3.x ships against coal (renamed hpp-fcl). Older builds still
// expose hpp::fcl. Detect at preprocess time and alias the namespace.
#if __has_include(<coal/shape/geometric_shapes.h>)
#  include <coal/shape/geometric_shapes.h>
#  include <coal/mesh_loader/loader.h>
namespace pinokin_fcl = coal;
#elif __has_include(<hpp/fcl/shape/geometric_shapes.h>)
#  include <hpp/fcl/shape/geometric_shapes.h>
#  include <hpp/fcl/mesh_loader/loader.h>
namespace pinokin_fcl = hpp::fcl;
#else
#  error "Neither <coal/...> nor <hpp/fcl/...> headers found. " \
         "pinokin requires Pinocchio built with collision support."
#endif

namespace pinokin {

namespace {

pinocchio::SE3 se3_from_matrix4(const Eigen::Matrix4d& T) {
    return pinocchio::SE3(T.template block<3, 3>(0, 0),
                          T.template block<3, 1>(0, 3));
}

// Build a CollisionGeometryPtr in the form Pinocchio expects.
// pinocchio::GeometryObject takes its CollisionGeometryPtr typedef, which
// in 3.x is std::shared_ptr<coal::CollisionGeometry>. We construct via
// std::make_shared and let implicit conversion handle the rest.
template <typename Shape, typename... Args>
auto make_shape_ptr(Args&&... args) {
    return std::make_shared<Shape>(std::forward<Args>(args)...);
}

}  // namespace

CollisionChecker::CollisionChecker(const Robot& robot,
                                   const std::string& urdf_path,
                                   const std::vector<std::string>& package_dirs,
                                   bool add_all_pairs,
                                   bool remove_adjacent_pairs)
    : robot_(robot) {
    std::vector<std::string> dirs = package_dirs;
    if (dirs.empty()) {
        try {
            std::filesystem::path p(urdf_path);
            if (p.has_parent_path()) {
                dirs.push_back(p.parent_path().string());
            }
        } catch (...) {
            // ignore; pinocchio will surface a mesh-resolution error
        }
    }

    pinocchio::urdf::buildGeom(robot_.model(), urdf_path,
                               pinocchio::COLLISION, geom_model_, dirs);

    kinds_.assign(geom_model_.geometryObjects.size(), GeomKind::Link);

    if (add_all_pairs) {
        geom_model_.addAllCollisionPairs();
        if (remove_adjacent_pairs) {
            populate_default_pairs(true);
        }
    }

    rebuild_geom_data_();
    rebuild_name_index_();
}

void CollisionChecker::load_srdf(const std::string& srdf_path) {
    // Default to verbose=false; throw on parse error rather than silent skip.
    pinocchio::srdf::removeCollisionPairs(robot_.model(), geom_model_,
                                          srdf_path, /*verbose=*/false);
    rebuild_geom_data_();
}

void CollisionChecker::add_collision_pair(std::size_t first,
                                          std::size_t second) {
    geom_model_.addCollisionPair(pinocchio::CollisionPair(first, second));
    rebuild_geom_data_();
}

void CollisionChecker::remove_collision_pair(std::size_t first,
                                             std::size_t second) {
    geom_model_.removeCollisionPair(pinocchio::CollisionPair(first, second));
    rebuild_geom_data_();
}

bool CollisionChecker::in_collision(const Eigen::VectorXd& q) const {
    return pinocchio::computeCollisions(robot_.model(), robot_.data(),
                                        geom_model_, geom_data_, q,
                                        /*stopAtFirstCollision=*/true);
}

std::vector<std::pair<std::size_t, std::size_t>>
CollisionChecker::colliding_pairs(const Eigen::VectorXd& q) const {
    pinocchio::computeCollisions(robot_.model(), robot_.data(),
                                 geom_model_, geom_data_, q,
                                 /*stopAtFirstCollision=*/false);
    std::vector<std::pair<std::size_t, std::size_t>> out;
    const auto& pairs = geom_model_.collisionPairs;
    out.reserve(pairs.size());
    for (std::size_t k = 0; k < pairs.size(); ++k) {
        if (geom_data_.collisionResults[k].isCollision()) {
            out.emplace_back(pairs[k].first, pairs[k].second);
        }
    }
    return out;
}

double CollisionChecker::min_distance(const Eigen::VectorXd& q) const {
    pinocchio::computeDistances(robot_.model(), robot_.data(),
                                geom_model_, geom_data_, q);
    double best = std::numeric_limits<double>::infinity();
    for (const auto& r : geom_data_.distanceResults) {
        if (r.min_distance < best) {
            best = r.min_distance;
        }
    }
    return best;
}

bool CollisionChecker::check_segment(const Eigen::VectorXd& q0,
                                     const Eigen::VectorXd& q1,
                                     int n_steps,
                                     bool include_endpoints) const {
    if (n_steps < 1) n_steps = 1;
    if (seg_q_.size() != q0.size()) seg_q_.resize(q0.size());

    if (include_endpoints) {
        if (in_collision(q0)) return true;
    }
    for (int i = 1; i < n_steps; ++i) {
        const double s = static_cast<double>(i) / n_steps;
        seg_q_ = (1.0 - s) * q0 + s * q1;
        if (in_collision(seg_q_)) return true;
    }
    if (include_endpoints) {
        if (in_collision(q1)) return true;
    }
    return false;
}

int CollisionChecker::check_path(const Eigen::MatrixXd& q_path) const {
    const Eigen::Index n = q_path.rows();
    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::VectorXd qi = q_path.row(i).transpose();
        if (in_collision(qi)) return static_cast<int>(i);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Runtime geometry add/remove
// ---------------------------------------------------------------------------

std::size_t CollisionChecker::add_geometry_object_(
    pinocchio::GeometryObject obj, GeomKind kind) {
    if (name_to_handle_.find(obj.name) != name_to_handle_.end()) {
        throw std::runtime_error(
            "CollisionChecker: geometry with name '" + obj.name +
            "' already exists; remove it first");
    }
    const std::size_t new_id = geom_model_.addGeometryObject(obj);
    kinds_.push_back(kind);
    name_to_handle_[obj.name] = new_id;

    // Apply pair policy.
    const std::size_t n = geom_model_.geometryObjects.size();
    for (std::size_t other = 0; other < n - 1; ++other) {
        const GeomKind ok = kinds_[other];

        bool add_pair = false;
        if (kind == GeomKind::World) {
            // World vs links and World vs attached, but not World vs World.
            add_pair = (ok == GeomKind::Link || ok == GeomKind::Attached);
        } else if (kind == GeomKind::Attached) {
            const auto& new_obj = geom_model_.geometryObjects[new_id];
            const auto& other_obj = geom_model_.geometryObjects[other];
            if (ok == GeomKind::Link) {
                // Skip same-joint and parent/child to avoid trivial contacts.
                add_pair = !is_adjacent_joint_(new_obj.parentJoint,
                                               other_obj.parentJoint);
            } else if (ok == GeomKind::World || ok == GeomKind::Attached) {
                add_pair = true;
            }
        }

        if (add_pair) {
            geom_model_.addCollisionPair(pinocchio::CollisionPair(other, new_id));
        }
    }

    rebuild_geom_data_();
    return new_id;
}

std::size_t CollisionChecker::add_obstacle_box(
    const std::string& name,
    const Eigen::Vector3d& half_extents,
    const Eigen::Matrix4d& world_pose) {
    auto shape = make_shape_ptr<pinokin_fcl::Box>(
        2.0 * half_extents.x(), 2.0 * half_extents.y(), 2.0 * half_extents.z());
    pinocchio::GeometryObject obj(name,
                                  /*parent_frame=*/0,
                                  /*parent_joint=*/0,
                                  shape,
                                  se3_from_matrix4(world_pose));
    return add_geometry_object_(std::move(obj), GeomKind::World);
}

std::size_t CollisionChecker::add_obstacle_sphere(
    const std::string& name, double radius,
    const Eigen::Matrix4d& world_pose) {
    auto shape = make_shape_ptr<pinokin_fcl::Sphere>(radius);
    pinocchio::GeometryObject obj(name, 0, 0, shape,
                                  se3_from_matrix4(world_pose));
    return add_geometry_object_(std::move(obj), GeomKind::World);
}

std::size_t CollisionChecker::add_obstacle_cylinder(
    const std::string& name, double radius, double length,
    const Eigen::Matrix4d& world_pose) {
    auto shape = make_shape_ptr<pinokin_fcl::Cylinder>(radius, length);
    pinocchio::GeometryObject obj(name, 0, 0, shape,
                                  se3_from_matrix4(world_pose));
    return add_geometry_object_(std::move(obj), GeomKind::World);
}

std::size_t CollisionChecker::add_obstacle_mesh(
    const std::string& name, const std::string& mesh_path,
    const Eigen::Matrix4d& world_pose,
    const Eigen::Vector3d& mesh_scale) {
    pinokin_fcl::MeshLoader loader;
    auto shape = loader.load(mesh_path, mesh_scale);
    pinocchio::GeometryObject obj(name, 0, 0, shape,
                                  se3_from_matrix4(world_pose));
    obj.meshPath = mesh_path;
    obj.meshScale = mesh_scale;
    return add_geometry_object_(std::move(obj), GeomKind::World);
}

pinocchio::JointIndex CollisionChecker::resolve_parent_joint_(
    const std::string& parent_frame,
    Eigen::Matrix4d& placement_in_joint) const {
    const auto& model = robot_.model();
    if (!model.existFrame(parent_frame)) {
        throw std::runtime_error(
            "CollisionChecker: frame '" + parent_frame + "' not found");
    }
    const auto fid = model.getFrameId(parent_frame);
    const auto& frame = model.frames[fid];
    // Compose: jMnew = jMframe * user_placement
    const pinocchio::SE3 jMnew =
        frame.placement * se3_from_matrix4(placement_in_joint);
    placement_in_joint = jMnew.toHomogeneousMatrix();
    return frame.parentJoint;
}

std::size_t CollisionChecker::attach_box_to_frame(
    const std::string& name, const Eigen::Vector3d& half_extents,
    const std::string& parent_frame, const Eigen::Matrix4d& placement) {
    Eigen::Matrix4d p = placement;
    const auto pj = resolve_parent_joint_(parent_frame, p);
    auto shape = make_shape_ptr<pinokin_fcl::Box>(
        2.0 * half_extents.x(), 2.0 * half_extents.y(), 2.0 * half_extents.z());
    pinocchio::GeometryObject obj(name, /*parent_frame=*/0, pj, shape,
                                  se3_from_matrix4(p));
    return add_geometry_object_(std::move(obj), GeomKind::Attached);
}

std::size_t CollisionChecker::attach_sphere_to_frame(
    const std::string& name, double radius,
    const std::string& parent_frame, const Eigen::Matrix4d& placement) {
    Eigen::Matrix4d p = placement;
    const auto pj = resolve_parent_joint_(parent_frame, p);
    auto shape = make_shape_ptr<pinokin_fcl::Sphere>(radius);
    pinocchio::GeometryObject obj(name, 0, pj, shape, se3_from_matrix4(p));
    return add_geometry_object_(std::move(obj), GeomKind::Attached);
}

std::size_t CollisionChecker::attach_cylinder_to_frame(
    const std::string& name, double radius, double length,
    const std::string& parent_frame, const Eigen::Matrix4d& placement) {
    Eigen::Matrix4d p = placement;
    const auto pj = resolve_parent_joint_(parent_frame, p);
    auto shape = make_shape_ptr<pinokin_fcl::Cylinder>(radius, length);
    pinocchio::GeometryObject obj(name, 0, pj, shape, se3_from_matrix4(p));
    return add_geometry_object_(std::move(obj), GeomKind::Attached);
}

std::size_t CollisionChecker::attach_mesh_to_frame(
    const std::string& name, const std::string& mesh_path,
    const std::string& parent_frame, const Eigen::Matrix4d& placement,
    const Eigen::Vector3d& mesh_scale) {
    Eigen::Matrix4d p = placement;
    const auto pj = resolve_parent_joint_(parent_frame, p);
    pinokin_fcl::MeshLoader loader;
    auto shape = loader.load(mesh_path, mesh_scale);
    pinocchio::GeometryObject obj(name, 0, pj, shape, se3_from_matrix4(p));
    obj.meshPath = mesh_path;
    obj.meshScale = mesh_scale;
    return add_geometry_object_(std::move(obj), GeomKind::Attached);
}

void CollisionChecker::set_geometry_pose(std::size_t handle,
                                         const Eigen::Matrix4d& pose) {
    if (handle >= geom_model_.geometryObjects.size()) {
        throw std::runtime_error(
            "CollisionChecker: invalid geometry handle");
    }
    geom_model_.geometryObjects[handle].placement = se3_from_matrix4(pose);
    // Refresh oMg for static (world) objects so subsequent diagnostic
    // queries reflect the new pose without a fresh q evaluation.
    if (kinds_[handle] == GeomKind::World) {
        geom_data_.oMg[handle] = geom_model_.geometryObjects[handle].placement;
    }
}

void CollisionChecker::set_geometry_pose_by_name(const std::string& name,
                                                 const Eigen::Matrix4d& pose) {
    auto it = name_to_handle_.find(name);
    if (it == name_to_handle_.end()) {
        throw std::runtime_error(
            "CollisionChecker: no geometry named '" + name + "'");
    }
    set_geometry_pose(it->second, pose);
}

Eigen::Matrix4d CollisionChecker::geometry_world_pose(
    const std::string& name) const {
    auto it = name_to_handle_.find(name);
    if (it == name_to_handle_.end()) {
        throw std::runtime_error(
            "CollisionChecker: no geometry named '" + name + "'");
    }
    return geom_data_.oMg[it->second].toHomogeneousMatrix();
}

void CollisionChecker::remove_geometry(std::size_t handle) {
    if (handle >= geom_model_.geometryObjects.size()) {
        throw std::runtime_error(
            "CollisionChecker: invalid geometry handle");
    }
    const std::string name = geom_model_.geometryObjects[handle].name;
    geom_model_.removeGeometryObject(name);
    kinds_.erase(kinds_.begin() + handle);
    rebuild_geom_data_();
    rebuild_name_index_();
}

void CollisionChecker::remove_geometry_by_name(const std::string& name) {
    auto it = name_to_handle_.find(name);
    if (it == name_to_handle_.end()) {
        // Idempotent — quietly succeed.
        return;
    }
    remove_geometry(it->second);
}

void CollisionChecker::reparent_geometry(std::size_t handle,
                                         const std::string& new_parent_frame,
                                         const Eigen::Matrix4d& new_placement) {
    if (handle >= geom_model_.geometryObjects.size()) {
        throw std::runtime_error(
            "CollisionChecker: invalid geometry handle");
    }
    Eigen::Matrix4d p = new_placement;
    pinocchio::JointIndex pj = 0;
    if (new_parent_frame == "universe" || new_parent_frame.empty()) {
        pj = 0;
        kinds_[handle] = GeomKind::World;
    } else {
        pj = resolve_parent_joint_(new_parent_frame, p);
        kinds_[handle] = GeomKind::Attached;
    }
    geom_model_.geometryObjects[handle].parentJoint = pj;
    geom_model_.geometryObjects[handle].placement = se3_from_matrix4(p);
    // Pair set may need re-evaluation if the kind changed. Conservative:
    // rebuild data so existing pairs keep working; users wanting full
    // re-pair should remove + re-add.
    rebuild_geom_data_();
}

void CollisionChecker::reparent_geometry_by_name(
    const std::string& name,
    const std::string& new_parent_frame,
    const Eigen::Matrix4d& new_placement) {
    auto it = name_to_handle_.find(name);
    if (it == name_to_handle_.end()) {
        throw std::runtime_error(
            "CollisionChecker: no geometry named '" + name + "'");
    }
    reparent_geometry(it->second, new_parent_frame, new_placement);
}

void CollisionChecker::update_placements(const Eigen::VectorXd& q) const {
    pinocchio::updateGeometryPlacements(robot_.model(), robot_.data(),
                                        geom_model_, geom_data_, q);
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

std::size_t CollisionChecker::num_collision_pairs() const {
    return geom_model_.collisionPairs.size();
}

std::size_t CollisionChecker::num_geometry_objects() const {
    return geom_model_.geometryObjects.size();
}

std::vector<std::string> CollisionChecker::geometry_names() const {
    std::vector<std::string> names;
    names.reserve(geom_model_.geometryObjects.size());
    for (const auto& obj : geom_model_.geometryObjects) {
        names.push_back(obj.name);
    }
    return names;
}

bool CollisionChecker::has_geometry(const std::string& name) const {
    return name_to_handle_.find(name) != name_to_handle_.end();
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

bool CollisionChecker::is_adjacent_joint_(pinocchio::JointIndex j1,
                                          pinocchio::JointIndex j2) const {
    if (j1 == j2) return true;
    const auto& parents = robot_.model().parents;
    if (j1 < parents.size() && parents[j1] == j2) return true;
    if (j2 < parents.size() && parents[j2] == j1) return true;
    return false;
}

void CollisionChecker::populate_default_pairs(bool remove_adjacent_pairs) {
    if (!remove_adjacent_pairs) return;
    const auto& objects = geom_model_.geometryObjects;
    std::vector<pinocchio::CollisionPair> to_remove;
    to_remove.reserve(geom_model_.collisionPairs.size());
    for (const auto& cp : geom_model_.collisionPairs) {
        if (cp.first >= objects.size() || cp.second >= objects.size()) continue;
        const auto j1 = objects[cp.first].parentJoint;
        const auto j2 = objects[cp.second].parentJoint;
        if (is_adjacent_joint_(j1, j2)) {
            to_remove.push_back(cp);
        }
    }
    for (const auto& cp : to_remove) {
        geom_model_.removeCollisionPair(cp);
    }
}

void CollisionChecker::rebuild_geom_data_() {
    geom_data_ = pinocchio::GeometryData(geom_model_);
}

void CollisionChecker::rebuild_name_index_() {
    name_to_handle_.clear();
    name_to_handle_.reserve(geom_model_.geometryObjects.size());
    for (std::size_t i = 0; i < geom_model_.geometryObjects.size(); ++i) {
        name_to_handle_[geom_model_.geometryObjects[i].name] = i;
    }
    // kinds_ may have been resized by URDF load; ensure size matches.
    if (kinds_.size() != geom_model_.geometryObjects.size()) {
        kinds_.assign(geom_model_.geometryObjects.size(), GeomKind::Link);
    }
}

}  // namespace pinokin

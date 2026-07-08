#pragma once

#include <optional>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

#include <Eigen/Dense>
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>

namespace pinokin {

class Robot;

// Wraps pinocchio::GeometryModel + GeometryData and provides fast
// boolean / pair-list / distance queries plus runtime obstacle, gripper,
// and payload management. Owns a private pinocchio::Data: every query
// recomputes FK from its q argument, so sharing the Robot's Data bought
// nothing and only risked cross-component interference.
//
// Pair policy:
//   - URDF-loaded link geometry: all-pairs minus parent/child adjacency
//     (controlled by remove_adjacent_pairs).
//   - Runtime-added world obstacles: paired against every robot link and
//     every attached object; NOT against other world obstacles.
//   - Frame-attached objects (gripper, payload): paired against every
//     robot link not on the parent-joint chain, against world obstacles,
//     and against other frame-attached objects.
//
// load_srdf() can later disable specific pairs that the default policy
// over-enables.
class CollisionChecker {
public:
    CollisionChecker(const Robot& robot,
                     const std::string& urdf_path,
                     const std::vector<std::string>& package_dirs = {},
                     bool add_all_pairs = true,
                     bool remove_adjacent_pairs = true,
                     double clearance_margin = 0.0);

    void load_srdf(const std::string& srdf_path);
    void add_collision_pair(std::size_t first, std::size_t second);
    void remove_collision_pair(std::size_t first, std::size_t second);

    // Fixed clearance: geometry within this distance counts as colliding
    // (coal CollisionRequest.security_margin), applied to every pair that
    // has no per-obstacle margin override.
    void set_clearance_margin(double margin);
    double clearance_margin() const { return clearance_margin_; }

    // Fast boolean check, early-exits on first colliding pair.
    bool in_collision(const Eigen::VectorXd& q) const;

    // All pairs in collision (no early exit). Pairs are geometry object
    // names (URDF link names for arm geometry, the user-supplied name for
    // runtime-attached objects).
    std::vector<std::pair<std::string, std::string>>
    colliding_pairs(const Eigen::VectorXd& q) const;

    // Min distance across all enabled pairs. Negative => penetration.
    double min_distance(const Eigen::VectorXd& q) const;

    // Linearly-interpolated segment check. Returns true if any sample
    // is in collision.
    bool check_segment(const Eigen::VectorXd& q0,
                       const Eigen::VectorXd& q1,
                       int n_steps,
                       bool include_endpoints = true) const;

    // Pre-sampled trajectory check. Returns first colliding row index,
    // or -1 if clear. Row-major to match numpy's default layout.
    using PathMatrix =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    int check_path(const PathMatrix& q_path) const;

    // ----- Runtime geometry: world obstacles (parented to universe) ----
    // Generic obstacle add — mirrors coal's shape ctors. kind ∈ {box, sphere,
    // cylinder, capsule, cone, ellipsoid, plane}; params interpreted per kind
    // (box: full x,y,z; sphere: r; cylinder/capsule/cone: r,length;
    // ellipsoid: rx,ry,rz; plane: nx,ny,nz,offset → Halfspace). Mesh has its
    // own loader-based method below. margin overrides the global clearance
    // for every pair this obstacle participates in.
    std::size_t add_obstacle(const std::string& name,
                             const std::string& kind,
                             const std::vector<double>& params,
                             const Eigen::Matrix4d& world_pose,
                             std::optional<double> margin = std::nullopt);

    std::size_t add_obstacle_box(const std::string& name,
                                 const Eigen::Vector3d& half_extents,
                                 const Eigen::Matrix4d& world_pose);
    std::size_t add_obstacle_sphere(const std::string& name,
                                    double radius,
                                    const Eigen::Matrix4d& world_pose);
    std::size_t add_obstacle_cylinder(const std::string& name,
                                      double radius,
                                      double length,
                                      const Eigen::Matrix4d& world_pose);
    std::size_t add_obstacle_mesh(const std::string& name,
                                  const std::string& mesh_path,
                                  const Eigen::Matrix4d& world_pose,
                                  const Eigen::Vector3d& mesh_scale
                                      = Eigen::Vector3d::Ones());

    // ----- Runtime geometry: frame-attached (gripper / payload) --------
    std::size_t attach_box_to_frame(const std::string& name,
                                    const Eigen::Vector3d& half_extents,
                                    const std::string& parent_frame,
                                    const Eigen::Matrix4d& placement);
    std::size_t attach_sphere_to_frame(const std::string& name,
                                       double radius,
                                       const std::string& parent_frame,
                                       const Eigen::Matrix4d& placement);
    std::size_t attach_cylinder_to_frame(const std::string& name,
                                         double radius,
                                         double length,
                                         const std::string& parent_frame,
                                         const Eigen::Matrix4d& placement);
    std::size_t attach_mesh_to_frame(const std::string& name,
                                     const std::string& mesh_path,
                                     const std::string& parent_frame,
                                     const Eigen::Matrix4d& placement,
                                     const Eigen::Vector3d& mesh_scale
                                         = Eigen::Vector3d::Ones());

    // Reposition (world pose for obstacles, placement for attached).
    void set_geometry_pose(std::size_t handle,
                           const Eigen::Matrix4d& pose);
    void set_geometry_pose_by_name(const std::string& name,
                                   const Eigen::Matrix4d& pose);

    // Returns the current world pose of a geometry, computed via the
    // checker's current FK state. Caller must have run a query (or call
    // update_placements(q)) first to refresh data_.
    Eigen::Matrix4d geometry_world_pose(const std::string& name) const;

    void remove_geometry(std::size_t handle);
    void remove_geometry_by_name(const std::string& name);

    // Re-parent (e.g., release payload from gripper into world).
    void reparent_geometry(std::size_t handle,
                           const std::string& new_parent_frame,
                           const Eigen::Matrix4d& new_placement);
    void reparent_geometry_by_name(const std::string& name,
                                   const std::string& new_parent_frame,
                                   const Eigen::Matrix4d& new_placement);

    // Refresh data_.oMf and geom_data_.oMg for diagnostic queries.
    void update_placements(const Eigen::VectorXd& q) const;

    // Introspection.
    std::size_t num_collision_pairs() const;
    std::size_t num_geometry_objects() const;
    std::vector<std::string> geometry_names() const;
    bool has_geometry(const std::string& name) const;

    // (geometry name, display name) for every geometry: URDF link geometry
    // reports its parent link's name; runtime-added geometry keeps its
    // user-supplied name. Lets callers report colliding pairs in URDF-link
    // vocabulary instead of internal geometry identifiers.
    std::vector<std::pair<std::string, std::string>> geometry_link_names() const;

    const pinocchio::GeometryModel& geom_model() const { return geom_model_; }
    pinocchio::GeometryData& geom_data() const { return geom_data_; }

private:
    enum class GeomKind { Link, World, Attached };

    void populate_default_pairs();
    void rebuild_geom_data_();
    void rebuild_name_index_();
    // Re-derive every pair's security_margin from the global clearance and
    // the per-geometry overrides.
    void apply_margins_();

    // Append a GeometryObject and add the appropriate pairs based on kind.
    std::size_t add_geometry_object_(pinocchio::GeometryObject obj,
                                     GeomKind kind);

    bool is_adjacent_joint_(pinocchio::JointIndex j1,
                            pinocchio::JointIndex j2) const;
    pinocchio::JointIndex resolve_parent_joint_(
        const std::string& parent_frame,
        Eigen::Matrix4d& placement_in_joint /* in/out */) const;

    const Robot& robot_;
    pinocchio::GeometryModel geom_model_;
    mutable pinocchio::GeometryData geom_data_;
    // Owned FK scratch for queries — see class comment (not the Robot's Data).
    mutable pinocchio::Data data_;

    // Fixed clearance applied to every collision pair's security_margin.
    double clearance_margin_ = 0.0;

    // Names -> handle for runtime-added geometry (gripper, payload,
    // world obstacles). URDF-loaded link geometry names also live here.
    std::unordered_map<std::string, std::size_t> name_to_handle_;

    // Tracks how each geometry was added so we can apply the right pair
    // policy when something new is added. Indexed by GeometryID.
    std::vector<GeomKind> kinds_;

    // Per-geometry clearance override (NaN = none). Indexed by GeometryID,
    // kept parallel to kinds_.
    std::vector<double> geom_margins_;

    // Buffer reused by check_segment and check_path.
    mutable Eigen::VectorXd seg_q_;
};

}  // namespace pinokin

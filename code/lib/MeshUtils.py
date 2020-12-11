import torch
import torch.nn as nn
import numpy as np
import BboxTools as bbt

from pytorch3d.renderer.mesh.rasterizer import Fragments
import pytorch3d.renderer.mesh.utils as utils
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
    FoVPerspectiveCameras
)
try:
    from pytorch3d.structures import Meshes, Textures
    use_textures = True
except:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.renderer import TexturesVertex as Textures

    use_textures = False


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(' ')[0])
    all_strings = ''.join(file_list[2:2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

    all_strings = ''.join(file_list[2 + n_points:])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(array_int.reshape((-1, 4))[:, 1::])


def save_off(off_file_name, vertices, faces):
    out_string = 'OFF\n'
    out_string += '%d %d 0\n' % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += '%.16f %.16f %.16f\n' % (v[0], v[1], v[2])
    for f in faces:
        out_string += '3 %d %d %d\n' % (f[0], f[1], f[2])
    with open(off_file_name, 'w') as fl:
        fl.write(out_string)
    return


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float:
        if device_ is None:
            device_ = 'cpu'
        theta = torch.ones((1, 1, 1)).to(device_) * theta
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if this is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = utils._clip_barycentric_coordinates(
            fragments.bary_coords
        )
        clipped_zbuf = utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments


def campos_to_R_T(campos, theta, device='cpu', at=((0, 0, 0),), up=((0, 1, 0), )):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


# For meshes in PASCAL3D+
def pre_process_mesh_pascal(verts):
    verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    return verts


# Calculate interpolated maps -> [n, c, h, w]
# face_memory.shape: [n_face, 3, c]
def forward_interpolate(R, T, meshes, face_memory, rasterizer, blur_radius=0):
    fragments = rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)
    # [n, h, w, 1, d]
    out_map = utils.interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_memory)
    out_map = out_map.squeeze(dim=3)
    out_map = out_map.transpose(3, 2).transpose(2, 1)
    return out_map


def vertex_memory_to_face_memory(memory_bank, faces):
    return memory_bank[faces.type(torch.long)]


def center_crop_fun(out_shape, max_shape):
    box = bbt.box_by_shape(out_shape, (max_shape[0] // 2, max_shape[1] // 2), image_boundary=max_shape)
    return lambda x: box.apply(x)


class MeshInterpolateModule(nn.Module):
    def __init__(self, vertices, faces, memory_bank, rasterizer, post_process=None, off_set_mesh=False):
        super(MeshInterpolateModule, self).__init__()

        # Convert memory feature of vertices to face
        self.face_memory = None
        self.update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple mesh at same time
        if type(vertices) == list:
            self.n_mesh = len(vertices)
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = [pre_process_mesh_pascal(t) for t in vertices]

            # Create Pytorch3D mesh
            self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        else:
            self.n_mesh = 1
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = pre_process_mesh_pascal(vertices)

            # Create Pytorch3D mesh
            self.meshes = Meshes(verts=[verts], faces=[faces], textures=None)

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, faces=None):
        if type(memory_bank) == list:
            if faces is None:
                faces = self.faces
            # Convert memory feature of vertices to face
            self.face_memory = torch.cat([vertex_memory_to_face_memory(m, f).to(m.device) for m, f in zip(memory_bank, faces)], dim=0)
        else:
            if faces is None:
                faces = self.faces
            # Convert memory feature of vertices to face
            self.face_memory = vertex_memory_to_face_memory(memory_bank, faces).to(memory_bank.device)

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super(MeshInterpolateModule, self).to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.face_memory = self.face_memory.to(device)
        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(self, campos, theta, blur_radius=0, deform_verts=None, **kwargs):
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        n_cam = campos.shape[0]
        if n_cam > 1 and self.n_mesh > 1:
            get = forward_interpolate(R, T, meshes, self.face_memory, rasterizer=self.rasterizer, blur_radius=blur_radius)
        elif n_cam > 1 and self.n_mesh == 1:
            get = forward_interpolate(R, T, meshes.extend(campos.shape[0]), self.face_memory.repeat(campos.shape[0], 1, 1).view(-1, *self.face_memory.shape[1:]), rasterizer=self.rasterizer, blur_radius=blur_radius)
        else:
            get = forward_interpolate(R, T, meshes, self.face_memory, rasterizer=self.rasterizer, blur_radius=blur_radius)

        if self.post_process is not None:
            get = self.post_process(get)
        return get


def camera_position_to_spherical_angle(camera_pose):
    distance_o = torch.sum(camera_pose ** 2, axis=1) ** .5
    azimuth_o = torch.atan(camera_pose[:, 0] / camera_pose[:, 2]) % np.pi + np.pi * (camera_pose[:, 0] < 0).type(camera_pose.dtype).to(camera_pose.device)
    elevation_o = torch.asin(camera_pose[:, 1] / distance_o)
    return distance_o, elevation_o, azimuth_o


def angel_gradient_modifier(base, grad_=None, alpha=(1.0, 1.0), center_=None):
    # alpha[0]: normal
    # alpha[1]: tangential
    if grad_ is None:
        grad_ = base.grad
        apply_to = True
    else:
        apply_to = False

    if center_ is not None:
        base_ = base.clone() - center_
    else:
        base_ = base

    with torch.no_grad():
        direction = base_ / torch.sum(base_ ** 2, dim=1) ** .5
        normal_vector = torch.sum(direction * grad_, dim=1, keepdim=True) * direction

        tangential_vector = grad_ - normal_vector
        out = normal_vector * alpha[0] + tangential_vector * alpha[1]

    if apply_to:
        base.grad = out

    return out


def decompose_pose(pose, sorts=('distance', 'elevation', 'azimuth', 'theta')):
    return pose[:, sorts.index('distance')], pose[:, sorts.index('elevation')], \
           pose[:, sorts.index('azimuth')], pose[:, sorts.index('theta')]


def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


def standard_loss_func_with_clutter(obj_s: torch.Tensor, clu_s: torch.Tensor):
    clu_s = torch.max(clu_s, dim=1)[0]
    return torch.ones(1, device=obj_s.device) - (torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s))


class MeshTrainingForwardModule(nn.Module):
    def __init__(self, path_mesh_file, render_size, feature_bank, n_points, clutter_merge_func=lambda x: normalize(torch.mean(x, dim=0), dim=0).unsqueeze(0), gradient_to_bank=False, train_mesh=False):
        super(MeshTrainingForwardModule, self).__init__()
        render_image_size = max(render_size)
        cameras = OpenGLPerspectiveCameras(fov=12.0)
        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        xvert, xface = load_off(path_mesh_file, to_torch=True)

        self.inter_module = MeshInterpolateModule(xvert, xface, feature_bank.memory[0:n_points], rasterizer,
                                             post_process=center_crop_fun(render_size, (max(render_size),) * 2), off_set_mesh=train_mesh)
        self.feature_bank = feature_bank
        self.n_points = n_points
        self.grad_to_bank = gradient_to_bank
        self.clutter_merge_func = clutter_merge_func

        if train_mesh:
            self.deform_verts = torch.nn.parameter.Parameter(torch.Tensor(*self.inter_module.meshes.verts_packed().shape))
            with torch.no_grad():
                self.deform_verts.fill_(0)
        else:
            self.register_parameter('deform_verts', None)

    def cuda(self, device=None):
        super().cuda(device)
        self.device = torch.device("cuda")
        self.inter_module.cuda(device)
        return self.inter_module.cuda(device)

    def to(self, device):
        super().to(device)
        self.inter_module.to(device)
        self.device = device
        return self.inter_module.to(device)

    def get_final_verts(self):
        if self.deform_verts is None:
            return None
        return self.inter_module.meshes.offset_verts(self.deform_verts).get_mesh_verts_faces(0)

    def save_mesh(self, mesh_file_path):
        final_verts, final_faces = self.get_final_verts()
        save_off(mesh_file_path, final_verts.detach().cpu().numpy(), final_faces.detach().cpu().numpy())

    def forward(self, forward_feature, pose, ):
        with torch.set_grad_enabled(self.grad_to_bank):
            self.inter_module.update_memory(self.feature_bank.memory[0:self.n_points])
            clutter_features = self.clutter_merge_func(self.feature_bank.memory[self.n_points::])

        pose_ = decompose_pose(pose)
        C = camera_position_from_spherical_angles(*pose_[0:3], device=forward_feature.device)
        theta = pose_[3]
        projected_feature = self.inter_module(C, theta, deform_verts=self.deform_verts)

        # [n, w, h]
        sim_fg = torch.sum(projected_feature * forward_feature, dim=1)

        # [n, clutter_num, w, h]
        sim_bg = torch.nn.functional.conv2d(forward_feature, clutter_features.unsqueeze(2).unsqueeze(3))

        return sim_fg, sim_bg

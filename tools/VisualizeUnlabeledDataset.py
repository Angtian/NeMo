import sys
sys.path.append('../code/lib')


import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import BboxTools as bbt
from lib.MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras


device = 'cuda:0'

render_image_size = (672, 672)
image_size = (256, 672)


anno_path = '../final_pred.npz'
img_path = 'path_to_dataset'

mesh_path = 'path_to_mesh'


class CustomedCrop(object):
    def __init__(self, crop_size, tar_horizontal):
        self.crop_size = crop_size
        self.tar_horizontal = tar_horizontal

    def __call__(self, im):
        size_ = im.size
        out_size = (self.tar_horizontal, int(size_[1] / size_[0] * self.tar_horizontal),)
        img = np.array(im.resize(out_size))
        crop_box = bbt.box_by_shape(self.crop_size, bbt.full(img).center, ).shift((30, 0))
        cropped_img = crop_box.apply(img)
        return cropped_img


x3d, xface = load_off(mesh_path)

faces = torch.from_numpy(xface)

# TODO: convert verts
verts = torch.from_numpy(x3d)
verts = pre_process_mesh_pascal(verts)
# cameras = OpenGLPerspectiveCameras(device=device, fov=12.0)
cameras = PerspectiveCameras(focal_length=1.0 * 3000, principal_point=((render_image_size[0]/ 2, render_image_size[1]/ 2),), image_size=(render_image_size, ), device=device)

verts_rgb = torch.ones_like(verts)[None] * torch.Tensor([1, 0.85, 0.85]).view(1, 1, 3)  # (1, V, 3)
# textures = Textures(verts_rgb=verts_rgb.to(device))
textures = Textures(verts_features=verts_rgb.to(device))
meshes = Meshes(verts=[verts], faces=[faces], textures=textures)
meshes = meshes.to(device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=render_image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0
)
# We can add a point light in front of the object.
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights, cameras=cameras),
)

trans = CustomedCrop(image_size, 430)
annos = np.load(anno_path, allow_pickle=True)




for k in list(annos.keys())[0::10]:

    img_ = Image.open(img_path + '%s.jpg' % k)
    img = trans(img_)

    print(annos[k])
    distance_pred, theta_pred, elevation_pred, azimuth_pred, t0, t1 = annos[k]
    print(distance_pred)
    # Image.fromarray(img).show()
    C = camera_position_from_spherical_angles(distance_pred, elevation_pred, azimuth_pred, degrees=False, device=device)
    R, T = campos_to_R_T(C, torch.Tensor([theta_pred]), device=device, extra_trans=torch.Tensor([[t0, t1, 0]]).to(device))

    image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T,)
    image = image[0, ..., :3].detach().squeeze().cpu().numpy()

    image = np.array((image / image.max()) * 255).astype(np.uint8)

    crop_box = bbt.box_by_shape(image_size, (render_image_size[0] // 2, render_image_size[1] // 2), image_boundary=render_image_size)

    image = crop_box.apply(image)

    mixed_image = (image * 0.6 + img * 0.4).astype(np.uint8)
    Image.fromarray(mixed_image).save('../Visuals1/%s.jpg' % k)

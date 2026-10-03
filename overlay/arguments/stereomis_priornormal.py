_base_ = './stereomis.py'
PipelineParams = dict(
    prior_normal=True,  # MoGe-2 normals (normal_moge/) supervise camera-frame rendered normals
)

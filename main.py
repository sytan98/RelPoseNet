import hydra
from relposenet.pipeline import PipelineWithAccel, PipelineWithDebug, PipelineWithIMU, PipelineWithNormal

@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    if cfg.model_mode == "accel":
        pipeline = PipelineWithAccel(cfg)
    elif cfg.model_mode == "imu":
        pipeline = PipelineWithIMU(cfg)
    elif cfg.model_mode == "normal_debug":
        pipeline = PipelineWithDebug(cfg)
    else:
        pipeline = PipelineWithNormal(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()

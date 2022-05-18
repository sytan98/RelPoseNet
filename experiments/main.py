import hydra
import sys
sys.path.append( '../' )
from seven_scenes.pipeline import SevenScenesBenchmark
from airsim.pipeline import AirSimBenchmark, AirSimWithIMUBenchmark


@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    benchmark = None
    if cfg.experiment.experiment_params.name == '7scenes':
        benchmark = SevenScenesBenchmark(cfg)
    elif cfg.experiment.experiment_params.name == 'AirsimIMU':
        benchmark = AirSimWithIMUBenchmark(cfg) 
    else:
        benchmark = AirSimBenchmark(cfg) 

    benchmark.evaluate()

if __name__ == "__main__":
    main()

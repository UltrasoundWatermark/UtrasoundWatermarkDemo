import numpy as np
import ncnn
from inference_runner import InferenceRunner


class NcnnRunner(InferenceRunner):
    def __init__(self, generator_param: str, generator_bin: str, detector_param: str, detector_bin: str):
        self.generator = ncnn.Net()
        self.generator.opt.use_vulkan_compute = True
        self.generator.opt.use_bf16_packed = False
        self.generator.opt.use_bf16_storage = False
        self.generator.opt.use_fp16_arithmetic = True
        self.generator.opt.use_fp16_packed = False
        self.generator.opt.use_fp16_storage = False
        self.generator.opt.use_int8_arithmetic = False
        self.generator.opt.use_int8_packed = False
        self.generator.opt.use_int8_storage = False
        self.generator.load_param(generator_param)
        self.generator.load_model(generator_bin)
        self.detector = ncnn.Net()
        self.detector.opt.use_vulkan_compute = True
        self.detector.opt.use_bf16_packed = True
        self.detector.opt.use_bf16_storage = True
        self.detector.opt.use_fp16_arithmetic = True
        self.detector.opt.use_fp16_packed = False
        self.detector.opt.use_fp16_storage = False
        self.detector.opt.use_int8_arithmetic = False
        self.detector.opt.use_int8_packed = False
        self.detector.opt.use_int8_storage = False
        self.detector.load_param(detector_param)
        self.detector.load_model(detector_bin)

    def run_generator(self, audio: np.ndarray, strength: float) -> np.ndarray:
        audio = audio[None, None, None, :]
        in0 = ncnn.Mat(audio)
        in1 = ncnn.Mat(np.array(np.float32(strength), ndmin=1))
        with self.generator.create_extractor() as ex:
            ex.input("in0", in0)
            _, out0 = ex.extract("out0")
            ret = np.array(out0)[0, 0, :]
            return ret

    def run_detector(self, audio: np.ndarray) -> np.float32:
        in0 = ncnn.Mat(audio)
        with self.detector.create_extractor() as ex:
            ex.input("in0", in0)
            _, out0 = ex.extract("out0")
            out = np.array(out0)
            ret = np.argmax(out, 0).mean()
            return ret

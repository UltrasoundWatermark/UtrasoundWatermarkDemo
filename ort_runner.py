import numpy as np
import onnxruntime as ort

from inference_runner import InferenceRunner


class OrtRunner(InferenceRunner):
    def __init__(self, generator_model: str, detector_model: str):
        super().__init__()
        self.generator_session = ort.InferenceSession(generator_model, providers=['CUDAExecutionProvider'])
        self.detector_session = ort.InferenceSession(detector_model, providers=['CUDAExecutionProvider'])

    def run_generator(self, audio: np.ndarray, strength: float) -> np.ndarray:
        audio = audio[None, None, :]
        input_names = self.generator_session.get_inputs()
        output_names = self.generator_session.get_outputs()
        # output = self.generator_session.run([output_names[0].name], {input_names[0].name: audio, input_names[1].name: np.array(np.float64(strength))})
        output = self.generator_session.run([output_names[0].name], {input_names[0].name: audio})
        output = output[0][0, 0, :]
        return output

    def run_detector(self, audio: np.ndarray) -> np.float32:
        input_names = self.detector_session.get_inputs()
        output_names = self.detector_session.get_outputs()
        audio = audio[None, None, :]
        output = self.detector_session.run([output_names[0].name], {input_names[0].name: audio})
        output = output[0][0, :, :]
        return np.argmax(output, 0).mean()

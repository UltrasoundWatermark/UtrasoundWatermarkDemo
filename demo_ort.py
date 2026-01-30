from demo_runner import DemoRunner
from ort_runner import OrtRunner


def main():
    INPUT_FILE = "example_input.wav"
    OUTPUT_FILE = "watermarked_ort.wav"
    impl = OrtRunner("generator.onnx", "detector.onnx")
    demo = DemoRunner(INPUT_FILE, OUTPUT_FILE, impl)
    demo.run_generator()
    demo.run_detector()


if __name__ == '__main__':
    main()

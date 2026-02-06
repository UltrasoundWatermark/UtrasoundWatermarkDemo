from demo_runner import DemoRunner
from ort_runner import OrtRunner


def main():
    INPUT_FILE = "example_input.npz"
    OUTPUT_FILE = "watermarked_ort.wav"
    impl = OrtRunner("generator.onnx", "detector.onnx")
    demo = DemoRunner(impl)
    demo.run_generator(INPUT_FILE, OUTPUT_FILE)
    demo.run_detector(OUTPUT_FILE)

if __name__ == '__main__':
    main()

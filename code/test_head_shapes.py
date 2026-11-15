import torch
from model import OCTClassifier


def main():
    # Run this test on an allocated GPU compute node.
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Run inside a GPU allocation.")

    device = torch.device("cuda")
    num_classes = (5, 5, 5, 5, 3, 6, 5, 5)

    # One synthetic OCT volume: batch, channel, three spatial dimensions.
    x = torch.ones(1, 1, 128, 128, 128, device=device)
    print(f"Input shape: {tuple(x.shape)}", flush=True)

    for width in (128, 64, 32, 16):
        model = OCTClassifier(
            num_classes=num_classes,
            layers=(3, 4, 6, 3),
            head0_hidden_dim=width,
        ).to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(x)

        assert len(outputs) == len(num_classes), "Incorrect head count."

        for i, (logits, classes) in enumerate(zip(outputs, num_classes)):
            expected = (1, classes)
            actual = tuple(logits.shape)
            assert actual == expected, (
                f"Width {width}, head {i}: expected {expected}, got {actual}"
            )
            assert torch.isfinite(logits).all().item(), (
                f"Width {width}, head {i}: NaN or infinite output."
            )

        print(f"PASS: Head 0 hidden width = {width}", flush=True)
        print([tuple(output.shape) for output in outputs], flush=True)
        del outputs, model

    print("All shape tests passed.", flush=True)


if __name__ == "__main__":
    main()
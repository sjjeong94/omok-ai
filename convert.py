import torch
import onnxruntime
import numpy as np

import models


def convert(
    model_path='./logs/model.pth',
    onnx_path='./logs/model.onnx',
):

    net = models.Tower()
    net.load_state_dict(torch.load(model_path))
    net = net.eval()

    x = torch.randn(1, 1, 15, 15)
    with torch.no_grad():
        y = net(x)

    torch.onnx.export(
        net,
        x,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        #dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(
        to_numpy(y), ort_outs[0], rtol=1e-03, atol=1e-05)


if __name__ == '__main__':
    convert()

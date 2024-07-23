import onnx
import sys

if __name__ == "__main__":
    onnx_model = onnx.load("best.onnx")
    graph = onnx_model.graph
    nodes = graph.node
    inputs = graph.input
    outputs = graph.output
    initializers = graph.initializer
    #import pdb; pdb.set_trace()

    outputs[0].name = outputs[0].name.replace("output","output0")

    for i in range(len(nodes)):
        if nodes[i].name == "Concat_348":
          print(nodes[i])
          for j in range(len(nodes[i].output)):
            print(nodes[i].output)
            if "output" in nodes[i].output[j]:
                #print(nodes[i])
                nodes[i].output[j] = nodes[i].output[j].replace("output","output0")


    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model,"out.onnx")
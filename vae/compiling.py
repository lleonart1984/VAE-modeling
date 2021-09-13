import math

from gevent.libev.watcher import stat
from sympy.codegen.cnodes import static
from torch import nn


class Compiler(object):
    def compile(self, models):
        pass


class HLSLCompiler(Compiler):
    def __init__(self):
        super(HLSLCompiler, self).__init__()

    @staticmethod
    def _flatten_layers(model):
        # case is a sequential
        if isinstance(model, nn.Sequential):
            for m in model:  # iterate over submodules
                for layer in HLSLCompiler._flatten_layers(m):
                    yield layer
            return
        yield model

    @staticmethod
    def _compile_layer_execution(layer: nn.Linear, activations, index):
        """
        Generates a code for a matrix multiplication and bias.
        n_{index+1}[i] = n_{index} x M[:,i] + W[i] i=0..output_size - 1
        """
        M = layer.weight.detach().numpy()  # dense weights
        W = layer.bias.detach().numpy()  # bias weights
        input_size = layer.in_features
        output_size = layer.out_features

        code = ""

        input_var = "n_" + str(index)
        output_var = "n_" + str(index + 1)

        activating = ""
        for a in activations:
            activating = a + "(" + activating

        for i in range(output_size):
            code += f"{output_var}[{i}]={activating}" + \
                    f"{'+'.join([input_var +'[' + str(j) + ']*' + str(M[i][j]) for j in range(input_size)])}+{W[i]}" + \
                    (")" * len(activations)) + ";\n "

        return code

    @staticmethod
    def _compile_sigmoid_activation(activation: nn.Sigmoid):
        return "float sigmoidActivation(float x){ return 1.0f / (1.0f + exp(-x)); }\n"

    @staticmethod
    def _compile_softplus_activation(activation: nn.Softplus):
        return "float softplusActivation(float x){ return log(1 + exp(x)); }\n"

    @staticmethod
    def _compile_tanh_activation(activation: nn.Tanh):
        return "float tanhActivation(float x){ return tanh(x); }\n"

    @staticmethod
    def _compile_relu_activation(activation: nn.ReLU):
        return "float reluActivation(float x){ return max(0, x); }\n"

    @staticmethod
    def _compile_leakyrelu_activation(activation: nn.LeakyReLU):
        return f"float leakyReLUActivation(float x){{ return x < 0 ? {activation.negative_slope} * x : x; }}\n"

    @staticmethod
    def _compile_activation(activation):
        compiled_function = None
        function_name = "unknown"

        if isinstance(activation, nn.Sigmoid):
            compile_function = HLSLCompiler._compile_sigmoid_activation
            function_name = "sigmoidActivation"

        if isinstance(activation, nn.Softplus):
            compile_function = HLSLCompiler._compile_softplus_activation
            function_name = "softplusActivation"

        if isinstance(activation, nn.Tanh):
            compile_function = HLSLCompiler._compile_tanh_activation
            function_name = "tanhActivation"

        if isinstance(activation, nn.ReLU):
            compile_function = HLSLCompiler._compile_relu_activation
            function_name = "reluActivation"

        if isinstance(activation, nn.LeakyReLU):
            compile_function = HLSLCompiler._compile_leakyrelu_activation
            function_name = "leakyReLUActivation"

        code = compile_function(activation) + "\n"
        return code, function_name

    @staticmethod
    def compile_model(model_name, model, activations):
        code = ""
        layers = list(HLSLCompiler._flatten_layers(model))
        index = 0
        linear_layer2compile = None
        activations2apply = []

        model_input_size = layers[0].in_features

        body_code = ""
        hidden_layer_size = []

        def compile_current_layer():
            nonlocal index
            nonlocal body_code
            if linear_layer2compile is None:
                return
            # add layer function
            body_code += HLSLCompiler._compile_layer_execution(linear_layer2compile, activations2apply, index)
            hidden_layer_size.append(linear_layer2compile.out_features)
            index += 1

        for l in layers:
            if isinstance(l, nn.Linear):  # New dense layer compile last
                compile_current_layer()
                linear_layer2compile = l
                activations2apply = []
            else:
                if not (type(l) in activations):
                    activation_code, name = HLSLCompiler._compile_activation(l)
                    activations[type(l)] = name
                    code += activation_code
                activation_name = activations[type(l)]  # get the name of the activation
                activations2apply.append(activation_name)  # append for future layer compilation
        compile_current_layer()

        # append model main function to code
        # function signature
        model_output_size = linear_layer2compile.out_features  # last layer output size
        code += f"void {model_name} (float n_0[{model_input_size}], out float n_{index}[{model_output_size}]) {{ \n"
        for i in range(0, index - 1):
            code += f"float n_{(i + 1)}[{hidden_layer_size[i]}];\n"
        code += body_code  # append input copying and layer executions
        code += "}\n\n"
        return code

    def compile(self, models):
        code = ""
        activations = {}
        for model_name, model in models.items():
            code += HLSLCompiler.compile_model(model_name, model, activations)
        return code

# Torch Model split

This script can automatically put the layers of a single model to different GPUs.

## Usage

```python
import modelsplit as ms

model = ms.DataFlow(model, output_device=0)

# The default GPU for each layer is 0
# You can change the layer by setting the layer_gpus in model
for key, value in model.layer_gpus.items():
    model.layer_gpus[key] = random.randint(0, len(model.device_ids)-1) # random

# Ensure you have updated the flow before you execute the model
# It will move the layers to each GPU you configured before
# And modify the forward function to match the GPU in each layer
model.update_flow()

# Execute the model
input = input.cuda(0)
model(input)
```

## Get Started

Only you need to do is wrap your torch model by this tool without any modification in your model. In this section, the mothods of controlling the model will be described.
```python
import modelsplit as ms

model = ms.DataFlow(your_torch_model, output_device=0)


# update the layer2gpus table
# ...

# update the flow
# Ensure you have updated the flow before you execute the model
# It will move the layers to each GPU you configured before
# And modify the forward function to match the GPU in each layer
model.update_flow()
```

### Split methods

There exists two methods to control the device for each layers in model, fine-grained split and submodule split, respectively.

* After you wrapped the model, you can control the gpu for each layer by editing its member `layer_gpus`. (e.g. `model.layer_gpus`), it's an OrderedDict type with (key, value) for name of layer and the device id number for this layer.

1. Fine-grained Split

All layer can be splitted into different GPUs. You can enable it by set the fine_grained option as True when you wrap the model.

```python
model = ms.DataFlow(your_torch_model,
                    output_device=0,
                    fine_grained=True)

# modify the device for each layer
# The following code is mean that all layers will be mapped to device 1
for key, value in model.layer_gpus.items():
    model.layer_gpus[key] = 1

model.update_flow()
```

2. Submodule Split

The default method when you wrapping the model is submodule split, or you can set . It will choose the module from `named_children()` from your torch model.

Take an example, the submodules of GoogLeNet from torchvision are shown as below. And the submodule split can control the device for these modules

```
Conv2d_1a_3x3
Conv2d_2a_3x3
Conv2d_2b_3x3
maxpool1
Conv2d_3b_1x1
Conv2d_4a_3x3
maxpool2
Mixed_5b
Mixed_5c
Mixed_5d
Mixed_6a
Mixed_6b
Mixed_6c
Mixed_6d
Mixed_6e
AuxLogits
Mixed_7a
Mixed_7b
Mixed_7c
avgpool
dropout
fc
```

### Inference only

Set the `inference_only` option as True can decrease the model size when you load into the device. It just simply disable the store of gradient.

```python
model = ms.DataFlow(your_torch_model,
                    output_device=0,
                    inference_only=True)
```

### Enable Clone

Actually, `update_flow()` will consume a little time on moving the module across different device, and modify the code in runtime.

When you set the `enable_clone` option as True, the model will be mapped to all device. And speedup the execution time of `update_flow`.

If you have the need to switch the device for each layer quickly between the model inferences, you can consider turning on this option, but one thing needs to be noted. Make sure that this model can be completely put into the device and inference will not exceed the device memory.

This option will make the model map to each device when wrapping the model by this tool, and modify the code of this model. When execute the `update_flow`, it will construct a model from these modules in different device, skip the overhead on moving modules and modifying the code.


![](https://i.imgur.com/v4RGcOE.png)

#### Costruct the module

When the `enable_clone` is True, you can construct a new module from the existed modules in devices.

```python
def construct_module(self, layer_gpus=None, module=None)
```

1. If `layer_gpus` is given, it will construct a new model form the argument layer_gpus. Otherwise, it will use the current layer_gpus in model.

2. You can create a copy of original module at first to speed up the execution time of `construct_module` when module option is set.

    ```pyyhon
    import copy

    m1 = copy.deepcopy(model.module)
    m2 = copy.deepcopy(model.module)


    m1 = model.construct_module(layer_gpus=layers1, module=m1)
    m2 = model.construct_module(layer_gpus=layers2, module=m2)

    tensor = ...
    m1(tensor)
    m2(tensor)
    ```

## TODO

- [ ]  Auto Data Pipeline

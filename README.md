# Torch Model split

This script can automatically put the layers of a single model to different GPUs.

Usage:

```python
import modelsplit as ms

model = ms.DataFlow(model)

# The default GPU for each layer is 0
# You can change the layer by setting the layer_gpus in model
for key, value in model.layer_gpus.items():
    model.layer_gpus[key] = random.randint(0, len(model.device_ids)-1) # random

# Ensure you have updated the flow before you execute the model
# It will move the layers to each GPU you configured before
# And modify the forward function to match the GPU in each layer
model.update_flow()

# Execute the model
input.to(device)
model(input)
```

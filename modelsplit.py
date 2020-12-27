from collections import OrderedDict 
import types
import ast
import inspect
import copy
import textwrap
import astunparse

import torch
from torch.nn.modules import Module
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)

class _CudaMappingVisitor(ast.NodeVisitor):
    def __init__(self, output_device=None, layer_gpus=OrderedDict()):
        super(_CudaMappingVisitor)
        self.layer_gpus = layer_gpus
        self.output_device = output_device

    def visit_Return(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        # processing return val => return val.cuda(output_device)
        # in AST
        #   Return(value=val)
        # =>
        #   Return(value=Call(func=Attribute(value=arg,
        #                                  attr='cuda',
        #                                  ctx=Load()),
        #                   args=[Num(n=output_device)]))
        value = ast.Call(func=ast.Attribute(value=node.value,
                                            attr='cuda',
                                            ctx=ast.Load()),
                         args=[ast.Num(n=self.output_device)],
                         keywords=[], starargs=None, kwargs=None)
        node.value = value
        

    #! currently, it only can deal with the layer call with only one argument
    def visit_Call(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        # processing self.layer(arg) => self.layer(arg.cuda(device_id))
        # In AST
        #   Call(func=Attribute(value=Name(id='self', ctx=Load()),
        #                       attr ='layer',
        #                       ctx  =Load()),
        #        args=[arg])
        # => 
        #   Call(func=Attribute(value=Name(id='self', ctx=Load()),
        #                       attr ='layer',
        #                       ctx  =Load()),
        #        args=[Call(func=Attribute(value=arg,
        #                                  attr='cuda',
        #                                  ctx=Load()),
        #                   args=[Num(n=device_id)]
        #             ])
        func = node.func
        if (len(node.args) == 1 and  # TODO, release the restrict of one argument 
            isinstance(func, ast.Attribute) and
            isinstance(func.ctx, ast.Load)  and
            isinstance(func.value, ast.Name)):
            value = func.value
            attr  = func.attr
            arg   = node.args[0]

            # check weather it is belong to model
            if value.id == 'self' and isinstance(value.ctx, ast.Load):
                # get the layer device id
                device_id = self.layer_gpus[attr]
                new_arg=ast.Call(func=ast.Attribute(value=arg,
                                                    attr='cuda',
                                                    ctx=ast.Load()),
                                 args=[ast.Num(n=device_id)],
                                 keywords=[], starargs=None, kwargs=None)
                # udpate args
                node.args = [new_arg]

class DataFlow(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataFlow, self).__init__()

        device_type = _get_available_device_type()

        if device_type is None:
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = _get_all_device_indices()

        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        # because inference only, so disable the gradient in model
        for param in self.module.parameters():
            param.requires_grad=False

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

        self.layer_gpus = OrderedDict()
        for name, module in self.module.named_children():
            self.layer_gpus[name] = self.output_device

        self.old_forward = copy.deepcopy(self.module.forward)

    def update_flow(self):
        self.module.forward = self.old_forward

        # update the submodule gpus
        for name, module in self.module.named_children():
            module.cuda(self.layer_gpus[name])

        # get the forward source code and convert it into AST
        source = textwrap.dedent(inspect.getsource(self.module.forward))
        tree = ast.parse(source)

        # udpate the AST
        v = _CudaMappingVisitor(layer_gpus=self.layer_gpus,
                                output_device=self.output_device)
        v.visit(tree)
        ast.fix_missing_locations(tree)

        # recompile
        code = compile(tree, filename="<ast>", mode="exec")
        namespace = self.module.forward.__globals__
        exec(code, namespace)
        self.module.forward = types.MethodType(namespace['forward'], self.module)

        # print(astunparse.unparse(tree))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)



    

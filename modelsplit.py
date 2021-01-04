from collections import OrderedDict
import types
import ast
import inspect
import copy
import textwrap

import torch
from torch.nn.modules import Module
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)

class _ChildMappingVisitor(ast.NodeVisitor):
    def __init__(self, module=None, output_device=None, layer_gpus=OrderedDict(), is_fine=False, old_functions={}):
        super(_ChildMappingVisitor)
        self.layer_gpus = layer_gpus
        self.output_device = output_device
        self.data = set()
        self.module = module
        self.is_fine = is_fine
        self.no_modify_return=False
        self.old_functions = old_functions

    def visit_Return(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        if self.no_modify_return:
            return

        # processing return val => return val.cuda(output_device)

        value = ast.Call(func=ast.Attribute(value=node.value,
                                            attr='cuda',
                                            ctx=ast.Load()),
                         args=[ast.Num(n=self.output_device)],
                         keywords=[ast.keyword(arg='non_blocking',
                                               value=ast.NameConstant(value=True))], starargs=None, kwargs=None)
        node.value = value
        
    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg != 'self':
                self.data.add(arg.arg)
        ast.NodeVisitor.generic_visit(self, node)
        self.data.clear()

    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.data.add(t.id)

    def visit_Call(self, node):
        ast.NodeVisitor.generic_visit(self, node)

        if self.is_fine:
            return 
        # processing self.layer(arg ...) => self.layer(arg.cuda(device_id) ...)

        func = node.func
        if (isinstance(func, ast.Attribute) and
            isinstance(func.ctx, ast.Load)  and
            isinstance(func.value, ast.Name)):
            value = func.value
            attr  = func.attr

            # check weather it is belong to model
            if (value.id == 'self' and
                attr in self.layer_gpus and
                isinstance(value.ctx, ast.Load)):
                # get the layer device id
                device_id = self.layer_gpus[attr]

                # upate args
                node.args = [ ast.Call(func=ast.Attribute(value=arg,
                                                          attr='cuda',
                                                          ctx=ast.Load()),
                                       args=[ast.Num(n=device_id)],
                                       keywords=[ast.keyword(arg='non_blocking',
                                                             value=ast.NameConstant(value=True))], starargs=None, kwargs=None)
                              if isinstance(arg, ast.Name) and
                              arg.id in self.data else arg for arg in node.args ]

            # attr is not in layer_gpus, traversal the function to modify
            elif value.id == 'self':
                func = getattr(self.module, attr)

                # save the func
                self.old_functions[attr] = copy.deepcopy(func)
                
                source = textwrap.dedent(inspect.getsource(func))
                tree = ast.parse(source)

                # shouldn't modify the return
                self.no_modify_return=True
                ast.NodeVisitor.generic_visit(self, tree)
                ast.fix_missing_locations(tree)
                self.no_modify_return=False

                name = func.__name__
                code = compile(tree, filename="<ast>_" + name, mode="exec")

                namespace = self.module.forward.__globals__
                exec(code, namespace)

                setattr(self.module, attr, types.MethodType(namespace[attr], self.module))

class _FineGrainedMappingVisitor(ast.NodeVisitor):
    def __init__(self, output_device=None, layer_gpus=OrderedDict(), operator_gpus=OrderedDict(), focus_operator=False):
        super(_FineGrainedMappingVisitor)
        self.layer_gpus = layer_gpus
        self.operator_gpus = operator_gpus
        self.output_device = output_device
        self.focus_operator = focus_operator
        self.instance_name = ''
        self.instance_type = ''
        self.data = set()

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            if arg.arg != 'self':
                self.data.add(arg.arg)

        for arg_name in self.data:
            device_id = self.operator_gpus[self.instance_type] \
                        if self.focus_operator \
                           else self.layer_gpus[self.instance_name]
                
            value = ast.Call(func=ast.Attribute(value=
                                                ast.Name(id=arg_name,
                                                         ctx=ast.Load()),
                                                attr='cuda',
                                                ctx=ast.Load()),
                             args=[ast.Num(n=device_id)],
                             keywords=[ast.keyword(arg='non_blocking',
                                                   value=ast.NameConstant(value=True))], starargs=None, kwargs=None)

            target = ast.Name(id=arg_name, ctx=ast.Store())
            assignment = ast.Assign(targets=[target], value=value)
            node.body.insert(0, assignment)

        ast.NodeVisitor.generic_visit(self, node)
        self.data.clear()

    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self.data.add(t.id)

        
class DataFlow(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0, inference_only=False, clear_cache=True, fine_grained=False, focus_operator=False):
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
        self.clear_cache = clear_cache
        self.fine_grained = fine_grained
        self.focus_operator = focus_operator

        # because inference only, so disable the gradient in model
        if inference_only:
            for param in self.module.parameters():
                param.requires_grad=False

        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

        self.layer_gpus = OrderedDict()
        self.operator_gpus = OrderedDict()

        if self.fine_grained:
            self.old_forwards = {}
            for n, m in self.module.named_modules():
                # terminal
                if len(m._modules) == 0:
                    self.old_forwards[n] = copy.deepcopy(m.forward)
                    self.operator_gpus[type(m).__name__] = self.output_device
                    self.layer_gpus[n] = self.output_device
            
        else:
            for n, m in self.module.named_children():
                self.layer_gpus[n] = self.output_device

        self.old_forward = copy.deepcopy(self.module.forward)
        self.old_functions = {}


    def _modify_forward(self, visitor, name, module):
        # get the forward source code and convert it into AST
        source = textwrap.dedent(inspect.getsource(module.forward))
        tree = ast.parse(source)

        # udpate the AST
        visitor.visit(tree)
        ast.fix_missing_locations(tree)
        
        # recompile
        code = compile(tree, filename="<ast>_" + name, mode="exec")
        namespace = module.forward.__globals__
        exec(code, namespace)

        return types.MethodType(namespace['forward'], module)

        
    def update_flow(self):
        self.module.forward = self.old_forward

        for attr in self.old_functions:
            setattr(self.module, attr, self.old_functions[attr])
        
        if self.fine_grained:
            for n, m in self.module.named_modules():
                # terminal
                if len(m._modules) == 0:
                    m.forward = self.old_forwards[n]
                    m.cuda(self.operator_gpus[type(m).__name__] \
                           if self.focus_operator else self.layer_gpus[n])

        else:
            # update the submodule gpus
            for n, m in self.module.named_children():
                m.cuda(self.layer_gpus[n])

        if self.clear_cache:
            torch.cuda.empty_cache()

        if self.fine_grained:
            fv = _FineGrainedMappingVisitor(layer_gpus=self.layer_gpus,
                                            operator_gpus=self.operator_gpus,
                                            output_device=self.output_device,
                                            focus_operator=self.focus_operator)

            for n, m in self.module.named_modules():
                if not n or len(m._modules) != 0:
                    continue

                fv.instance_name = n
                fv.instance_type = type(m).__name__
                m.forward = self._modify_forward(fv, n, m)

            # modify torch.cat
            namespace = self.module.forward.__globals__
            copy_cat = copy.deepcopy(namespace['torch'].cat)

            def torch_cat(arg, *args):
                arg = [x.cuda(self.output_device) for x in arg]
                return copy_cat(arg, *args)

            namespace['torch'].cat = copy.deepcopy(torch_cat)
                
        cv = _ChildMappingVisitor(module=self.module, layer_gpus=self.layer_gpus,
                                  output_device=self.output_device, is_fine=self.fine_grained,
                                  old_functions = self.old_functions)
        
        self.module.forward = self._modify_forward(cv, "main", self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

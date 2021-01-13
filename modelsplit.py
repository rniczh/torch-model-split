from collections import OrderedDict
import types
import ast
import inspect
import copy
import textwrap
import functools
import sys

import torch
from torch.nn.modules import Module
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        func_name = func.__self__.__class__.__name__
        print(f"Elapsed time: {elapsed_time:0.4f} ms [{func_name}]")
        return value
    return wrapper_timer

class _ChildMappingVisitor(ast.NodeVisitor):
    def __init__(self, module=None, output_device=None, layer_gpus=OrderedDict(), is_fine=False, old_functions={}, update_function=True, no_modify_return=False):
        super(_ChildMappingVisitor)
        self.layer_gpus = layer_gpus
        self.output_device = output_device
        self.data = set()
        self.module = module
        self.is_fine = is_fine
        self.no_modify_return=no_modify_return
        self.no_modify_call = False
        self.old_functions = old_functions
        self.modified = False
        self.update_function = update_function

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

        if self.no_modify_call or self.is_fine:
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

                self.modified = True

            # attr is not in layer_gpus, traversal the function to modify
            elif value.id == 'self' and self.update_function:
                func = getattr(self.module, attr)

                source = textwrap.dedent(inspect.getsource(func))
                tree = ast.parse(source)

                # shouldn't modify the return
                self.no_modify_return=True
                ast.NodeVisitor.generic_visit(self, tree)
                ast.fix_missing_locations(tree)
                self.no_modify_return=False


                if self.modified:
                    # save the func
                    self.old_functions[attr] = copy.deepcopy(func)
                    
                    name = func.__name__
                    code = compile(tree, filename="<ast>_" + name, mode="exec")
                    
                    namespace = self.module.forward.__globals__
                    exec(code, namespace)
                    setattr(self.module, attr, types.MethodType(namespace[attr], self.module))

                    self.modified = False


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
    def __init__(self, module, device_ids=None, output_device=None, dim=0, inference_only=False, clear_cache=True, fine_grained=False, focus_operator=False, enable_clone=False, prof_time=False):
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
        self.submodule_updated = False
        self.enable_clone = enable_clone

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
            self.old_forwards = {}
            for n, m in self.module.named_children():
                if self.enable_clone:
                    self.old_forwards[n] = copy.deepcopy(m.forward)
                self.layer_gpus[n] = self.output_device

        self.old_forward = copy.deepcopy(self.module.forward)
        self.old_functions = {}

        self._time = False

        self.clone_modules = {}
        for i in self.device_ids:
            self.clone_modules[i] = {}


        if self.enable_clone:
            if not self.fine_grained:
                for device_id in self.device_ids:
                    self.layer_gpus = OrderedDict([(n, device_id) for n in self.layer_gpus])
                    self._update_flow(prof_time=prof_time)
                    for n, m in self.module.named_children():
                        self.clone_modules[device_id][n] = copy.deepcopy(m)
                        self.clone_modules[device_id][n].cuda(device_id)
                        
            else:
                for device_id in self.device_ids:
                    self.layer_gpus = OrderedDict([(n, device_id) for n in self.layer_gpus])
                    self._update_flow(prof_time=prof_time)
                    for n, m in self.module.named_modules():
                        if len(m._modules) == 0:
                            self.clone_modules[device_id][n] = copy.deepcopy(m)
                            self.clone_modules[device_id][n].cuda(device_id)
                            


    def _modify_function(self, visitor, attr, func):
        # get the forward source code and convert it into AST
        source = textwrap.dedent(inspect.getsource(self.old_functions[attr]))
        tree = ast.parse(source)

        # udpate the AST
        visitor.visit(tree)
        ast.fix_missing_locations(tree)

        # recompile
        name = func.__name__
        code = compile(tree, filename="<ast>_" + name, mode="exec")
        namespace = self.module.forward.__globals__
        exec(code, namespace)

        return types.MethodType(namespace[attr], self.module)
    
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

    def construct_module(self):
        if self.enable_clone:
            return copy.deepcopy(self.module)
        else:
            print('You need to enable the clone by enable_clone option', file=sys.stderr)
            return None
            
    
    def update_flow(self, prof_time=False):
        if self.enable_clone:
            if self.fine_grained:
                for n, m in self.module.named_modules():
                    # terminal
                    if len(m._modules) == 0:
                        device_id = self.operator_gpus[type(m).__name__] \
                                    if self.focus_operator else self.layer_gpus[n]

                        cloned =  self.clone_modules[device_id][n]
                        m._parameters = cloned._parameters
                        m._buffers = cloned._buffers
                        m._non_persistent_buffers_set = cloned._non_persistent_buffers_set
                        m._modules = cloned._modules
                        if prof_time:
                            m.forward = timer(cloned.forward)
                        else:
                            m.forward = cloned.forward

            else:
                for n, m in self.module.named_children():
                    # if prof_time:
                    #     m.forward = timer(m.forward)
                    device_id = self.layer_gpus[n]
                    cloned =  self.clone_modules[device_id][n]

                    m._parameters = cloned._parameters
                    m._buffers = cloned._buffers
                    m._non_persistent_buffers_set = cloned._non_persistent_buffers_set
                    m._modules = cloned._modules
                    if prof_time:
                        m.forward = timer(cloned.forward)
                    else:
                        m.forward = cloned.forward
                        
                        
        else:
            self._update_flow(prof_time=prof_time)
    
    def _update_flow(self, prof_time=False):
        self.module.forward = self.old_forward

        # for attr in self.old_functions:
        #     setattr(self.module, attr, self.old_functions[attr])
        
        if self.fine_grained:
            for n, m in self.module.named_modules():
                # terminal
                if len(m._modules) == 0:
                    m.forward = self.old_forwards[n]
                    m.cuda(self.operator_gpus[type(m).__name__] \
                           if self.focus_operator else self.layer_gpus[n])

        else:
            for n, m in self.module.named_children():
                if self.enable_clone:
                    m.forward = self.old_forwards[n]
                elif prof_time:
                    m.forward = timer(m.forward)
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

                if prof_time and not self.enable_clone:
                    m.forward = timer(self._modify_forward(fv, n, m))
                else:
                    m.forward = self._modify_forward(fv, n, m)
                    
                    
        # modify torch.cat
        namespace = self.module.forward.__globals__
        copy_cat = copy.deepcopy(namespace['torch'].cat)
                
        def torch_cat(arg, *args):
            first_device_id = arg[0].device
            arg = [x.to(first_device_id) for x in arg]
            return copy_cat(arg, *args)

        namespace['torch'].cat = copy.deepcopy(torch_cat)

        cv = _ChildMappingVisitor(module=self.module, layer_gpus=self.layer_gpus,
                                  output_device=self.output_device,
                                  is_fine=self.fine_grained,
                                  old_functions = self.old_functions,
                                  update_function = False if self.fine_grained or self.submodule_updated else True,
                                  no_modify_return = True if self.submodule_updated else False)

        
        if self.submodule_updated and not self.enable_clone:
            # only update the old_functions
            for attr in self.old_functions:
                func = getattr(self.module, attr)
                setattr(self.module, attr, self._modify_function(cv, attr, func))

        
        if self.enable_clone and not self.fine_grained:
            # update each module's forward under named_children() instead
            # using fine grained visitor to update
            fv = _FineGrainedMappingVisitor(layer_gpus=self.layer_gpus,
                                            output_device=self.output_device,
                                            focus_operator=False)

            for n, m in self.module.named_children():
                fv.instance_name = n
                m.forward = self._modify_forward(fv, n, m)

        if self.enable_clone:
            cv.no_modify_call = True
        cv.no_modify_return = False
        self.module.forward = self._modify_forward(cv, "main", self.module)
        self.submodule_updated = True

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


        

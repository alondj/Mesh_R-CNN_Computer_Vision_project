from torch.nn.parallel.replicate import _broadcast_coalesced_reshape, _get_device_index,\
    _replicatable_module, _is_script_module, _init_script_module, _is_script_method, _copy_scriptmodule_methods

from copy import deepcopy
special_names = ['_backend',
                 '_parameters',
                 '_buffers',
                 '_backward_hooks',
                 '_forward_hooks',
                 '_forward_pre_hooks',
                 '_state_dict_hooks',
                 '_load_state_dict_pre_hooks',
                 '_modules']


def replicate(network, devices, detach=False):
    if not _replicatable_module(network):
        raise RuntimeError("Cannot replicate network where python modules are "
                           "childrens of ScriptModule")

    devices = list(map(lambda x: _get_device_index(x, True), devices))
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg = []
    buffers_not_rg = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx,
                             buf in enumerate(buffers_not_rg)}

    buffer_copies_rg = _broadcast_coalesced_reshape(
        buffers_rg, devices, detach=detach)
    buffer_copies_not_rg = _broadcast_coalesced_reshape(
        buffers_not_rg, devices, detach=True)

    modules = list(network.modules())
    module_copies = [[] for device in devices]
    module_indices = {}
    scriptmodule_skip_attr = {"_parameters",
                              "_buffers", "_modules", "forward", "_c"}

    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            if _is_script_module(module):
                # we have to initialize ScriptModule properly so that
                # it works with pybind11
                replica = _init_script_module()

                attribute_names = set(entry[0]
                                      for entry in module._c._get_attributes())
                keys = set(module.__dict__.keys()) - \
                    scriptmodule_skip_attr - attribute_names
                for key in keys:
                    if not _is_script_method(module.__dict__[key]):
                        replica.__dict__[key] = module.__dict__[key]
                for name, the_type, value in module._c._get_attributes():
                    if name in module._buffers.keys():
                        continue
                    replica._c._register_attribute(name, the_type, value)
            else:
                replica = module.__new__(type(module))
                replica.__dict__ = module.__dict__.copy()

                for k in module.__dict__.keys():
                    if k not in special_names:
                        replica.__dict__[k] = deepcopy(module.__dict__[k])

                replica._parameters = replica._parameters.copy()

                replica._buffers = replica._buffers.copy()
                replica._modules = replica._modules.copy()

            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            if child is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                module_idx = module_indices[child]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = module_copies[j][module_idx]
        for key, param in module._parameters.items():
            if param is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                param_idx = param_indices[param]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = param_copies[j][param_idx]
        for key, buf in module._buffers.items():
            if buf is None:
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = buffer_copies[j][buffer_idx]

    for j in range(num_replicas):
        _copy_scriptmodule_methods(modules, module_copies[j], module_indices)

    return [module_copies[j][0] for j in range(num_replicas)]

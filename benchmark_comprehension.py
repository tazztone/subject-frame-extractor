import timeit

setup = """
config_defaults = {
    f"default_{i}": i for i in range(100)
}
config_defaults.update({
    f"filter_default_{i}": i for i in range(20)
})
config_defaults.update({
    f"other_{i}": i for i in range(50)
})
valid_keys = {f"compute_{i}" for i in range(10, 30)}
"""

code_list_comp = """
defaults = {}
for metric in [k.replace("filter_default_", "") for k in config_defaults if k.startswith("filter_default_")]:
    compute_key = f"compute_{metric}"
    if compute_key in valid_keys:
        defaults[compute_key] = True
"""

code_direct_loop = """
defaults = {}
for k in config_defaults:
    if k.startswith("filter_default_"):
        compute_key = f"compute_{k[15:]}"
        if compute_key in valid_keys:
            defaults[compute_key] = True
"""

code_direct_loop_replace = """
defaults = {}
for k in config_defaults:
    if k.startswith("filter_default_"):
        compute_key = f"compute_{k.replace('filter_default_', '')}"
        if compute_key in valid_keys:
            defaults[compute_key] = True
"""

print("List comprehension:", timeit.timeit(code_list_comp, setup=setup, number=100000))
print("Direct loop with replace:", timeit.timeit(code_direct_loop_replace, setup=setup, number=100000))
print("Direct loop with slice:", timeit.timeit(code_direct_loop, setup=setup, number=100000))

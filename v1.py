import os
import shutil
import subprocess

def main():
    build("example", "example")


def build(dir, prog_name):
    out_path = os.path.join(dir, "c_out")
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path)
    
    modules = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            modules.append(transpile_module(filepath, out_path))

    bin_path = os.path.join(dir, "bin")
    shutil.rmtree(bin_path, ignore_errors=True)
    os.makedirs(bin_path)

    cmd = ["clang"]
    for module in modules:
        cmd.append(f'{out_path}/{module}.c')
    cmd.append("-o")
    cmd.append(f'{bin_path}/{prog_name}')
    print(str.join(" ", cmd))
    subprocess.run(cmd, check=True)


def transpile_module(filepath, out_path):
    with open(filepath) as f:
        src = f.read()

    modname, funcs, imports, c_includes = parse_module(src)
    write_files(modname, funcs, imports, c_includes, out_path)
    return modname


def parse_module(src):
    lines = src.splitlines()
    module_name = None
    funcs = []
    imports = []
    c_includes = []

    for line in lines:
        if line.startswith("module"):
            module_name = line.split()[1].rstrip(";")
        elif line.startswith("import"):
            imp = line.split()[1].rstrip(";")
            imports.append(imp)
        elif line.startswith("#include"):
            c_includes.append(line.strip())
        elif line.startswith("int "):
            header = line.split("{")[0].strip() + ";"
            body = line
            funcs.append((header, body))
    return module_name, funcs, imports, c_includes


def write_files(modname, funcs, imports, c_includes, out_path):
    with open(os.path.join(out_path, f"{modname}.h"), "w") as h:
        h.write(f"#ifndef {modname.upper()}_H\n#define {modname.upper()}_H\n\n")
        for header, _ in funcs:
            h.write(header + "\n")
        h.write(f"\n#endif\n")

    with open(os.path.join(out_path, f"{modname}.c"), "w") as c:
        c.write(f'#include "{modname}.h"\n\n')

        for c_incl in c_includes:
            c.write(c_incl + "\n")
        c.write("\n")
        
        for imp in imports:
            c.write(f'#include "{imp}.h"\n')
        c.write("\n")
        
        for _, body in funcs:
            c.write(body + "\n\n")


if __name__ == "__main__":
    main()

import os
import shutil
import subprocess
import sys

verbose = False

def main():
    if len(sys.argv) < 4:
        print_usage()
        return

    i = 1
    prog_name = ""
    prog_name_i = 1
    global verbose
    for arg in sys.argv[1:]:
        if arg == "-v" or arg == "--verbose":
            verbose = True
        elif arg == "-o" or arg == "--out":
            if i + 1 >= len(sys.argv):
                print_usage()
                return 
            prog_name = sys.argv[i + 1]
            prog_name_i = i + 1
        i += 1

    build_dir = ""
    i = 1
    for arg in sys.argv[1:]:
        if not arg.startswith("-") and i != prog_name_i:
            build_dir = arg
            break
        i += 1

    print(f'Building {build_dir} for "{prog_name}". Verbose={verbose}')

    build(build_dir, prog_name)

def print_usage():
    print("Usage: python3 s0.py -v ./build/dir -o prog_name")

def build(dir, prog_name):
    out_path = os.path.join(dir, "c_out")
    shutil.rmtree(out_path, ignore_errors=True)
    modules = process_dir(dir, out_path)

    print(f'\nTranspiled modules: {", ".join(module.name for module in modules)}\n')

    bin_path = os.path.join(dir, "bin")
    shutil.rmtree(bin_path, ignore_errors=True)
    clang_compile(modules, out_path, bin_path, prog_name)

def process_dir(dir, out_path):
    os.makedirs(out_path)
    modules = []
    for filename in os.listdir(dir):
        filepath = os.path.join(dir, filename)
        if os.path.isfile(filepath):
            module = process_file(filepath, out_path)
            modules.append(module)
    return modules

def clang_compile(modules, out_path, bin_path, prog_name):
    os.makedirs(bin_path)
    cmd = ["clang"]
    for module in modules:
        cmd.append(os.path.join(out_path, f'{module.name}.c'))
    cmd.append("-o")
    exe_name = os.path.join(bin_path, prog_name)
    cmd.append(exe_name)
    print(f'{str.join(" ", cmd)}\n')
    subprocess.run(cmd, check=True)
    return exe_name

def process_file(path, out_path):
    if verbose:
        print(f'\n\nProcessing file: {path}\n')
    src = ""
    with open(path, "r") as f:
        src = f.read()
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    if verbose:
        print(f'Tokens: {tokens}')
    module = parse_module(tokens, src)
    h_src, c_src = generate_for_module(module, src)
    write_module_to_file(out_path, module, h_src, c_src)
    return module

def parse_module(tokens, src):
    parser = Parser(tokens)

    modname = ""
    imports = []
    c_includes = []
    funcs = []
    structs = []
    extracted_ranges = []
    
    while True:
        tok = parser.advance()
        if tok.kind == 'EOF':
            break

        if tok.kind == 'MODULE':
            modname = parser.expect('IDENT').value
            parser.expect('SEMICOLON');
            # include following white space after struct defintion in extracted range
            p = parser.pos
            next_tok = parser.advance()
            parser.pos = p
            extracted_ranges.append((tok.start, next_tok.start))

        elif tok.kind == 'IMPORT':
            imp = parser.expect('IDENT').value
            parser.expect('SEMICOLON')
            imports.append(imp)
            # include following white space after struct defintion in extracted range
            p = parser.pos
            next_tok = parser.advance()
            parser.pos = p
            extracted_ranges.append((tok.start, next_tok.start))

        elif tok.kind == 'HASH':
            parser.expect('INCLUDE')
            str_inc = parser.expect('STRING', optional=True)
            if str_inc:
                c_includes.append((str_inc.value, '"'))
            else:
                parser.expect('LT')
                include_strs = []
                while parser.peek().kind != 'GT':
                    include_strs.append(parser.advance().value)
                parser.expect('GT')
                c_includes.append((''.join(include_strs), '<'))
            # include following white space after struct defintion in extracted range
            p = parser.pos
            next_tok = parser.advance()
            parser.pos = p
            extracted_ranges.append((tok.start, next_tok.start))


        elif tok.kind == 'STRUCT':
            struct_name = parser.expect('IDENT').value
            struct_tokens = []
            parser.expect('LBRACE')
            while parser.peek().kind != 'RBRACE':
                struct_tokens.append(parser.advance())
            structs.append(CStruct(struct_name, struct_tokens))
            parser.expect('RBRACE')
            parser.expect('SEMICOLON')
            # include following white space after struct defintion in extracted range
            p = parser.pos
            next_tok = parser.advance()
            parser.pos = p
            extracted_ranges.append((tok.start, next_tok.start))

        elif parser.check_function(tok):
            decl_tokens = parser.parse_func_signature(tok)
            # skip function body
            parser.expect('LBRACE')
            parser.find_matching_brace()
            funcs.append(CFunc(decl_tokens))

    module = Module(modname, imports, c_includes, funcs, structs, extracted_ranges)

    if verbose:
        print(f"modname: {modname}")
        print(f"imports: {imports}")
        print(f"c_includes: {c_includes}")
        struct_strs = list(f"{s.name}: {token_list_to_src_range(s.body_tokens, src)}" for s in structs)
        print(f"structs: {struct_strs}")
        func_strs = list(f"{token_list_to_src_range(f.decl_tokens, src)}" for f in funcs)
        print(f"funcs: {func_strs}")
        extracted_ranges_strs = list(f'({start}, {end}): "{src[start:end]}"' for start, end in extracted_ranges)
        print(f"extracted_ranges: {extracted_ranges_strs}")

    return module

class Module:
    def __init__(self, name, imports, c_includes, funcs, structs, extracted_ranges):
        self.name = name
        self.imports = imports
        self.c_includes = c_includes
        self.funcs = funcs
        self.structs = structs
        self.extracted_ranges = extracted_ranges

def generate_for_module(module, src):
    # emit c header
    h_src = ""
    h_src += "#pragma once\n"
    h_src += "\n"

    #include c includes
    any = False
    for c_inc in module.c_includes:
        h_src += f'#include {'"' if c_inc[1] == '"' else '<'}{c_inc[0]}{'"' if c_inc[1] == '"' else '>'}\n'
        any = True
    if any:
        h_src += "\n"

    #include modules
    any = False
    for imp in module.imports:
        h_src += f'#include "{imp}.h"\n'
        any = True
    if any:
        h_src += "\n"

    # struct forward declarations
    any = False
    for struct in module.structs:
        h_src += f'typedef struct {struct.name} {struct.name};\n'
        any = True
    if any:
        h_src += "\n"

    # struct definitions
    any = False
    for struct in module.structs:
        h_src += f'struct {struct.name}\n{{\n'
        h_src += '    '
        h_src += token_list_to_src_range(struct.body_tokens, src)
        h_src += '\n};\n'
        any = True
    if any:
        h_src += "\n"

    # func forward declarations
    for func in module.funcs:
        h_src += token_list_to_src_range(func.decl_tokens, src)
        h_src += ';\n'

    # emit c source
    c_src = ""
    with open(f"example_s0/c_out/{module.name}.c", "w") as f:
        c_src += f'#include "{module.name}.h"\n\n'

        #include c includes
        any = False
        for c_inc in module.c_includes:
            c_src += f'#include {'"' if c_inc[1] == '"' else '<'}{c_inc[0]}{'"' if c_inc[1] == '"' else '>'}\n'
            any = True
        if any:
            c_src += "\n"

        #include modules
        any = False
        for imp in module.imports:
            c_src += f'#include "{imp}.h"\n'
            any = True
        if any:
            c_src += "\n"

        c_src += copy_src_with_skips(src, module.extracted_ranges)

    return h_src, c_src

def write_module_to_file(out_path, module, h_src, c_src):
    with open(os.path.join(out_path, f'{module.name}.h'), "w") as f:
        f.write(h_src)

    with open(os.path.join(out_path, f'{module.name}.c'), "w") as f:
        f.write(c_src)

def copy_src_with_skips(src, skip_ranges):
    out = []
    i = 0
    for start, end in skip_ranges:
        if i < start:
            out.append(src[i:start])
        i = end
    if i < len(src):
        out.append(src[i:])
    return ''.join(out)

def token_list_to_src_range(tokens, src):
    return src[tokens[0].start:tokens[-1].end]

class CStruct:
    def __init__(self, name, body_tokens):
        self.name = name
        self.body_tokens = body_tokens

    def __repr__(self):
        return f"Struct({self.name}, {self.body_tokens})"
    
class CFunc:
    def __init__(self, decl_tokens):
        self.decl_tokens = decl_tokens

    def __repr__(self):
        return f"Func({self.decl_tokens})"

class Token:
    def __init__(self, kind, value, start, end):
        self.kind = kind
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, {self.start}, {self.end})"

class Lexer:
    def __init__(self, src):
        self.src = src
        self.pos = 0
        self.len = len(src)

    def peek(self):
        return self.src[self.pos] if self.pos < self.len else '\0'

    def advance(self):
        ch = self.peek()
        self.pos += 1
        return ch

    def skip_whitespace(self):
        while self.peek() in ' \r\t\n':
            self.advance()

    def match_keyword_or_indent(self, first, start_pos):
        ident = first
        while self.peek().isalnum() or self.peek() == '_':
            ident += self.advance()
        KEYWORDS = {
            'module',
            'import',
            'include',
            'struct'
        }
        if ident in KEYWORDS:
            return Token(ident.upper(), ident, start_pos, self.pos)
        return Token('IDENT', ident, start_pos, self.pos)

    def match_number(self, first, start_pos):
        num = first
        while self.peek().isdigit():
            num += self.advance()
        return Token('NUMBER', num, start_pos, self.pos)

    def read_string(self):
        result = ''
        start_pos = self.pos
        while True:
            ch = self.advance()
            if ch == '"':
                break
            result += ch
        return Token('STRING', result, start_pos, self.pos)

    def next_token(self):
        self.skip_whitespace()

        start_pos = self.pos
        ch = self.advance()

        if ch.isalpha() or ch == '_':
            return self.match_keyword_or_indent(ch, start_pos)

        if ch.isdigit():
            return self.match_number(ch, start_pos)

        if ch == '"':
            return self.read_string()

        SINGLE_CH_TOKENS = {
            '(': 'LPAREN',
            ')': 'RPAREN',
            '{': 'LBRACE',
            '}': 'RBRACE',
            ',': 'COMMA',
            ';': 'SEMICOLON',
            '+': 'PLUS',
            '=': 'EQUAL',
            '<': 'LT',
            '>': 'GT',
            '.': 'PERIOD',
            '#': 'HASH',
            '*': 'STAR'
        }

        if ch in SINGLE_CH_TOKENS:
            return Token(SINGLE_CH_TOKENS[ch], ch, start_pos, self.pos)

        if ch == '\0':
            return Token('EOF', '', start_pos, start_pos)

        raise SyntaxError(f'Unexpected character: {ch}')

    def tokenize(self):
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.kind == 'EOF':
                break
        return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos]

    def peek2(self):
        return self.tokens[self.pos + 1]

    def advance(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, kind, optional=False):
        token = self.peek()
        if token.kind != kind and not optional:
            raise SyntaxError(f"Expected {kind}, got {token.kind}");
        if token.kind == kind:
            self.advance()
            return token
        else:
            return None
        
    def find_matching_paren(self):
        old_pos = self.pos
        depth = 1
        while depth > 0:
            tok = self.peek()
            if tok.kind == 'LPAREN':
                depth += 1
            elif tok.kind == 'RPAREN':
                depth -= 1
            elif tok.kind == 'EOF':
                self.pos = old_pos
                return False
            self.advance()
        return True

    def find_matching_brace(self):
        old_pos = self.pos
        depth = 1
        while depth > 0:
            tok = self.peek()
            if tok.kind == 'LBRACE':
                depth += 1
            elif tok.kind == 'RBRACE':
                depth -= 1
            elif tok.kind == 'EOF':
                self.pos = old_pos
                return False
            self.advance()

    def check_function(self, first):
        if first.kind !='IDENT' and first.kind != 'STAR':
            return False
        old_pos = self.pos
        while self.peek().kind == 'IDENT' or self.peek().kind == 'STAR':
            tok = self.advance()
        if self.advance().kind != 'LPAREN':
            self.pos = old_pos
            return False
        if not self.find_matching_paren():
            self.pos = old_pos
            return False
        if self.advance().kind != 'LBRACE':
            self.pos = old_pos
            return False
        self.pos = old_pos
        return True

    def parse_func_signature(self, first):
        sign_toks = [first]
        while self.peek().kind != 'LBRACE':
            sign_toks.append(self.advance())
        return sign_toks

if __name__ == "__main__":
    main()
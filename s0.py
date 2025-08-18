def main():
    with open("example_s0/blah.s0") as f:
        src = f.read()
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    # print(tokens)
    parser = Parser(tokens)

    # parse
    modname = ""
    imports = []
    funcs = []
    structs = []
    
    while True:
        tok = parser.advance()
        if tok.kind == 'EOF':
            break

        if tok.kind == 'MODULE':
            modname = parser.expect('IDENT').value
            parser.expect('SEMICOLON');
        elif tok.kind == 'IMPORT':
            imp = parser.expect('IDENT').value
            parser.expect('SEMICOLON')
            imports.append(imp)
        elif tok.kind == 'STRUCT':
            struct_name = parser.expect('IDENT').value
            struct_tokens = []
            parser.expect('LBRACE')
            while parser.peek().kind != 'RBRACE':
                struct_tokens.append(parser.advance())
            structs.append(CStruct(struct_name, struct_tokens))
            parser.expect('RBRACE')
            parser.expect('SEMICOLON')
        elif parser.check_function(tok):
            decl_tokens = parser.parse_func_signature(tok)
            # skip function body
            parser.expect('LBRACE')
            parser.find_matching_brace()
            funcs.append(CFunc(decl_tokens))

    print(f"modname: {modname}\n")
    print(f"imports: {imports}\n")
    print(f"structs: {structs}\n")
    print(f"funcs: {funcs}")

    # emit c header
    with open("example_s0/c_out/blah.h", "w") as f:
        f.write("#pragma once\n")
        f.write("\n")

        #include modules
        for imp in imports:
            f.write(f'#include "{imp}.h"\n')
        f.write("\n")

        # struct forward declarations
        for struct in structs:
            f.write(f'typedef struct {struct.name} {struct.name};\n')
        f.write("\n")

        # struct definitions
        for struct in structs:
            f.write(f'struct {struct.name}\n{{\n')
            new_line = True
            for tok in struct.body_tokens:
                if new_line:
                    f.write('    ')
                    new_line = False
                if tok.kind == 'SEMICOLON':
                    f.write(f'{tok.value}\n')
                    new_line = True
                else:
                    f.write(f'{tok.value} ')
            f.write('};\n\n')

        # func forward declarations
        for func in funcs:
            for tok in func.decl_tokens:
                f.write(f'{tok.value} ')
            f.write(';\n')



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
    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"

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

    def match_keyword_or_indent(self, first):
        ident = first
        while self.peek().isalnum() or self.peek() == '_':
            ident += self.advance()
        KEYWORDS = {
            'module',
            'import',
            'struct'
        }
        if ident in KEYWORDS:
            return Token(ident.upper(), ident)
        return Token('IDENT', ident)

    def match_number(self, first):
        num = first
        while self.peek().isdigit():
            num += self.advance()
        return Token('NUMBER', num)

    def read_string(self):
        result = ''
        while True:
            ch = self.advance()
            if ch == '"':
                break
            result += ch
        return Token('STRING', result)

    def next_token(self):
        self.skip_whitespace()
        ch = self.advance()

        if ch.isalpha() or ch == '_':
            return self.match_keyword_or_indent(ch)

        if ch.isdigit():
            return self.match_number(ch)

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
            return Token(SINGLE_CH_TOKENS[ch], ch)

        if ch == '\0':
            return Token('EOF', '')

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
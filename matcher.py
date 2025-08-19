def main():
    # lexer_test()
    # matcher_test()
    matcher_test0()

def lexer_test():
    src = ""
    with open("example_s0/blah.s0") as file:
        src = file.read()
        print(len(src))
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    print(tokens)

def matcher_test0():
    src = "abc def();"
    lexer = Lexer(src)
    tokens = lexer.tokenize()

    pattern = seq(
        ident(),
        ident(),
        token("("),
        token(")"),
        token(";")
    )

    m = match_next(pattern, tokens, 0)
    print(m)

def matcher_test():
    src = ""
    with open("example_s0/blah.s0") as file:
        src = file.read()
        print(len(src))
    lexer = Lexer(src)
    tokens = lexer.tokenize()
    print(tokens)

    func_def = seq(
        capture("keywords", repeat(choice(token("inline"), token("static")))),
        capture("ret_type", ident()),
        capture("name", ident()),
        token("("),
        capture("params", skip_to_matching("(", ")")),
        token(")"),
        token("{"),
        capture("body", skip_to_matching("{", "}")),
        token("}")
    )

    m = match_next(func_def, tokens, 0)
    func = FuncDef(m["keywords"], m["ret_type"], m["name"], m["params"], m["body"])
    print(func)

class FuncDef:
    def __init__(self, keywords, ret_type, name, params, body):
        self.keywords = keywords
        self.ret_type = ret_type
        self.name = name
        self.params = params
        self.body = body

# Matcher

def match_next(pattern, tokens, pos):
    kind = pattern[0]

    if kind == "token":
        value = pattern[1]
        if pos < len(tokens) and tokens[pos].value == value:
            return pos + 1, {}

    elif kind == "ident":
        if pos < len(tokens) and tokens[pos].kind == 'IDENT':
            return pos + 1, {}

    elif kind == "seq":
        p = pos
        caps = {}
        for subpat in pattern[1]:
            res = match_next(subpat, tokens, p)
            if not res:
                return None
            p, subcaps = res
            caps.update(subcaps)
        return p, caps

    elif kind == "repeat":
        p = pos
        caps = {}
        while True:
            res = match_next(pattern[1], tokens, p)
            if not res:
                break
            p, subcaps = res
            caps.update(subcaps)
        return p, caps

    elif kind == "choice":
        for subpat in pattern[1]:
            res = match_next(subpat, tokens, pos)
            if res:
                return res
        return None

    elif kind == "capture":
        name, subpat = pattern[1], pattern[2]
        start = pos
        res = match_next(subpat, tokens, pos)
        if not res:
            return None
        end, subcaps = res
        subcaps[name] = tokens[start:end]

    elif kind == "skip_to_matching":
        left = pattern[1]
        right = pattern[2]
        depth = 0
        p = pos
        while p < len(tokens):
            tok = tokens[p]
            if tok.value == left:
                depth += 1
            elif tok.value == right:
                depth -= 1
                if depth == 0:
                    return p + 1, {}
            p += 1
        return None

    else:
        raise Exception("Unknown pattern")

def seq(*args):
    return ("seq", list(args))

def capture(name, pattern):
    return ("capture", name, pattern)

def repeat(pattern):
    return ("repeat", pattern)

def choice(*args):
    return ("choice", list(args))

def ident():
    return ("ident", )

def token(value):
    return ("token", value)

def skip_to_matching(left, right):
    return ("skip_to_matching", left, right)

# Lexer

class Token:
    def __init__(self, kind, value, start, end):
        self.kind = kind
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r}, {self.start}, {self.end})"

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
        start_pos = self.pos
        result = ''
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
            return Token(ch, ch, start_pos, self.pos)

        if ch == '\0':
            return Token('EOF', '', start_pos, self.pos)

        raise SyntaxError(f'Unexpected character: {ch}')

    def tokenize(self):
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.kind == 'EOF':
                break
        return tokens

if __name__ == "__main__":
    main()
class Token(object):
    def __init__(self, symbol, spec):
        self.symbol = symbol
        self.spec = spec
        if spec == 'const':
            self.value = int(symbol)
        elif spec == 'OCT':
            self.value = oct(int(symbol, base=8))
        elif spec == 'HEX':
            self.value = hex(int(symbol, base=16))
        elif spec == 'float':
            self.value = float(symbol)
        elif spec == 'IDN':
            self.value = symbol
        else:
            self.value = '_'

    def get_spec(self):
        return self.spec

    def get_value(self):
        return self.value

    def __repr__(self):
        return "{:}\t\t< {:} , {:} >".format(self.symbol, self.spec.upper(), self.value)

    def __str__(self):
        return "{:}\t\t< {:} , {:} >".format(self.symbol, self.spec.upper(), self.value)
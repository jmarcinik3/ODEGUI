import sympy as sp

if __name__ == "__main__":
    x, y, a, b = sp.symbols('x y a b')
    
    z = x + y
    
    f1 = x + y
    print(f1.free_symbols, f1.atoms(sp.Symbol))
    
    f2 = f1.subs(y, a+b)
    print(f2.free_symbols, f2.atoms(sp.Symbol))
    
    f3 = x + f2*f1
    print(f3.free_symbols, f3.atoms(sp.Function))
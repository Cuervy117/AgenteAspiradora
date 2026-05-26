from sympy import symbols, Not, Or, And
from sympy.logic.inference import satisfiable


class Literal:
    """
    Un 'Literal' es una proposición positiva o negativa.
    """

    def __init__(self, name, positive=True):
        self.name = name          
        self.positive = positive  

    def negate(self):
        return Literal(self.name, not self.positive)

    def to_sympy(self):
        """
        Traducción a símbolos de sympy.
        """
        sym = symbols(self.name)
        return sym if self.positive else Not(sym)

    def __eq__(self, other):
        return self.name == other.name and self.positive == other.positive

    def __hash__(self):
        return hash((self.name, self.positive))

    def __repr__(self):
        return f"{'' if self.positive else '~'}{self.name}"


class KnowledgeBase:
    """
    La 'Base de Conocimientos' (KnowledgeBase).
    """

    def __init__(self):
        self.clauses = []               # La lista donde guardamos las pistas
        self._kb_expr_cache = None      # Una memoria rápida para no pensar lo mismo dos veces
        self._cache_dirty = True        # Nos avisa si hay pistas nuevas que no hemos memorizado
        self._entails_cache = {}        # Caché de deducciones pasadas


    def add_fact(self, literal):
        """
        Anota un hecho.
        """
        fact = literal.to_sympy()
        if fact not in self.clauses:
            self.clauses.append(fact)
            self._cache_dirty = True

    def add_rule(self, sympy_expr):
        """
        Anota una regla lógica compleja.
        """
        if sympy_expr not in self.clauses:
            self.clauses.append(sympy_expr)
            self._cache_dirty = True

    def update_fact(self, literal):
        """
        Actualización de la KB.
        """
        fact = literal.to_sympy()
        neg_fact = literal.negate().to_sympy()
        
        if neg_fact in self.clauses: 
            self.clauses.remove(neg_fact) 
            self._cache_dirty = True
            
        if fact not in self.clauses:
            self.clauses.append(fact) # Escribimos el nuevo hecho.
            self._cache_dirty = True

    def entails(self, query_literal):
        """
        Inferencia lógica.
        """
        query_expr = query_literal.to_sympy()

        # Obtenemos el estado. Falso o Verdadero.
        if query_expr in self.clauses:
            estado = "Verdadero" if query_literal.positive else "Falso"
            return True, f"Usando memoria directa, sabemos explícitamente que {query_literal.name} es {estado}."
        
        # Verificamos si ya se resolvio.
        if query_expr in self._entails_cache:
            return self._entails_cache[query_expr]

        # Inferencia por refutacion
        neg_query_expr = Not(query_expr) 

        if not self.clauses:
            return False, "" # No existen datos.

        # Actualiza la caché si hay nuevas pistas
        if self._cache_dirty: 
            self._kb_expr_cache = And(*self.clauses)
            self._entails_cache.clear() 
            self._cache_dirty = False
        
        kb_expr = self._kb_expr_cache

        # Teorema de Refutación
        is_satisfiable = satisfiable(And(kb_expr, neg_query_expr))

        if is_satisfiable is False:
            # Se comprueba que la pregunta es verdadera
            estado = "Verdadero" if query_literal.positive else "Falso"
            explicacion = f"Usando matemáticas, descubrimos que {query_literal.name} es {estado} sin ninguna duda."
            self._entails_cache[query_expr] = (True, explicacion)
            return True, explicacion
        else:
            # No se pudo comprobar que la pregunta es verdadera.
            self._entails_cache[query_expr] = (False, "")
            return False, ""

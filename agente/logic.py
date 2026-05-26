"""
=============================================================================
  logic.py — El Cerebro Matemático (Motor Lógico)
=============================================================================
  Imagina que este archivo es el cuaderno de notas de Sherlock Holmes.
  Aquí se guardan las "pistas" que la aspiradora va descubriendo en su camino,
  y se hacen deducciones lógicas.

  Por ejemplo, si la aspiradora sabe que:
    1. Si choca, es porque hay una pared.
    2. ¡Acaba de chocar!
  Entonces deduce matemáticamente: "¡Debe haber una pared aquí!"

  Utiliza la magia de la "Lógica Proposicional". En lugar de adivinar,
  resuelve ecuaciones matemáticas complejas (usando una herramienta llamada
  SymPy) para demostrar sin dejar lugar a dudas si una casilla es segura o no.
=============================================================================
"""

from sympy import symbols, Not, Or, And
from sympy.logic.inference import satisfiable


class Literal:
    """
    Un 'Literal' es simplemente un Hecho Positivo o Negativo.
    
    Por ejemplo:
    - Hecho Positivo: "Hace Sol" (O_3_4 -> Hay obstáculo en 3,4)
    - Hecho Negativo: "NO hace Sol" (~O_3_4 -> NO hay obstáculo en 3,4)
    """

    def __init__(self, name, positive=True):
        self.name = name          # El nombre del hecho (ej. "O_3_4")
        self.positive = positive  # ¿Es verdad (True) o es mentira (False)?

    def negate(self):
        """
        Le da la vuelta al hecho. Si era "Sí", lo vuelve "No". 
        Si era "No", lo vuelve "Sí".
        """
        return Literal(self.name, not self.positive)

    def to_sympy(self):
        """
        Traduce nuestro hecho al idioma que la librería matemática SymPy 
        puede entender para resolver sus ecuaciones.
        """
        sym = symbols(self.name)
        return sym if self.positive else Not(sym)

    def __eq__(self, other):
        # Para que Python sepa si dos hechos son exactamente iguales
        return self.name == other.name and self.positive == other.positive

    def __hash__(self):
        return hash((self.name, self.positive))

    def __repr__(self):
        return f"{'' if self.positive else '~'}{self.name}"


class KnowledgeBase:
    """
    La 'Base de Conocimientos' (KnowledgeBase) es el Cuaderno de Pistas.
    Aquí se guardan todos los hechos (Literales) que sabemos que son 100% ciertos.
    """

    def __init__(self):
        self.clauses = []               # La lista donde guardamos las pistas
        self._kb_expr_cache = None      # Una memoria rápida para no pensar lo mismo dos veces
        self._cache_dirty = True        # Nos avisa si hay pistas nuevas que no hemos memorizado
        self._entails_cache = {}        # Caché de deducciones pasadas


    def add_fact(self, literal):
        """
        Anota una nueva pista innegable en el cuaderno.
        
        Args:
            literal (Literal): El hecho que descubrimos (ej. ¡Hay una pared aquí!).
        """
        fact = literal.to_sympy()
        if fact not in self.clauses:
            self.clauses.append(fact)
            self._cache_dirty = True # Hay pistas nuevas, hay que actualizar la memoria!

    def add_rule(self, sympy_expr):
        """
        Anota una regla lógica compleja (ej. P_1_1 <=> C_1_2 | C_2_1).
        """
        if sympy_expr not in self.clauses:
            self.clauses.append(sympy_expr)
            self._cache_dirty = True

    def update_fact(self, literal):
        """
        Cambia de opinión sobre un hecho si descubre que estaba equivocada.
        
        Por ejemplo, si el cuaderno decía "NO hay pared" pero de pronto
        encontramos una, tenemos que borrar el "NO hay pared" antes de escribir 
        el "SÍ hay pared". Si no lo borramos, el cuaderno se volvería loco.
        """
        fact = literal.to_sympy()
        neg_fact = literal.negate().to_sympy()
        
        if neg_fact in self.clauses: # Si había algo contrario a lo que vamos a anotar...
            self.clauses.remove(neg_fact) # Lo borramos con goma de borrar.
            self._cache_dirty = True
            
        if fact not in self.clauses:
            self.clauses.append(fact) # Escribimos el nuevo hecho.
            self._cache_dirty = True

    def entails(self, query_literal):
        """
        ¡LA FUNCIÓN MÁS INTELIGENTE DEL PROGRAMA!
        Esta es la Lupa de Detective. Le hacemos una pregunta al cuaderno y él 
        usa matemáticas para darnos la respuesta absoluta.

        ¿Cómo funciona el truco (La Prueba por Contradicción)?
        Imagínate que queremos probar que "El cielo es azul".
        1. Fingimos que es falso: "El cielo es ROJO".
        2. Metemos esa mentira junto con todo lo que sabemos de la física.
        3. El motor matemático explota (da Error/Contradicción).
        4. Como la mentira causó un error lógico, concluimos que la verdad SÍ 
           es que "El cielo es azul". ¡Genial!
        
        Args:
            query_literal (Literal): La pregunta (ej. ¿Es segura la casilla 3,4?)
            
        Returns:
            (EsVerdad?, "Por qué es verdad")
        """
        query_expr = query_literal.to_sympy()

        # Optimización 1: Early Exit si ya está en las pistas explícitas (O(1))
        if query_expr in self.clauses:
            estado = "Verdadero" if query_literal.positive else "Falso"
            return True, f"Usando memoria directa, sabemos explícitamente que {query_literal.name} es {estado}."

        # Optimización 2: Caché de consultas complejas
        if query_expr in self._entails_cache:
            return self._entails_cache[query_expr]

        neg_query_expr = Not(query_expr) # Creamos "la mentira"

        if not self.clauses:
            return False, "" # Si no sabemos nada del mundo, no podemos deducir nada.

        if self._cache_dirty: # Si hay pistas nuevas, juntamos todas las pistas en una sola súper-ecuación
            self._kb_expr_cache = And(*self.clauses)
            self._entails_cache.clear() # Limpiamos la caché porque el mundo cambió
            self._cache_dirty = False
        
        kb_expr = self._kb_expr_cache

        # Teorema de Refutación: Juntamos las pistas reales con "la mentira"
        # satisfiable() averigua si es matemáticamente posible que ambas existan juntas.
        is_satisfiable = satisfiable(And(kb_expr, neg_query_expr))

        if is_satisfiable is False:
            # ¡Explotó! Fue imposible. Por lo tanto, ¡nuestra pregunta original era Verdadera!
            estado = "Verdadero" if query_literal.positive else "Falso"
            explicacion = f"Usando matemáticas, descubrimos que {query_literal.name} es {estado} sin ninguna duda."
            self._entails_cache[query_expr] = (True, explicacion)
            return True, explicacion
        else:
            # No explotó. Significa que aún no tenemos suficientes pistas para estar 100% seguros.
            self._entails_cache[query_expr] = (False, "")
            return False, ""

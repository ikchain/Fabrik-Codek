#!/usr/bin/env python3
"""Generate evaluation benchmark cases for Fabrik model quality assessment.

This script creates 50+ categorized test cases to objectively measure
model performance across different task types.
"""

import json
from datetime import date
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval-benchmark" / "cases"


def create_case(id: str, category: str, difficulty: str, language: str,
                prompt: str, context: str, must_include: list,
                must_not_include: list, ideal_response: str,
                correctness: float = 0.5, completeness: float = 0.3,
                clarity: float = 0.2, tags: list = None, source: str = "manual") -> dict:
    """Create a benchmark case following the schema."""
    return {
        "id": id,
        "category": category,
        "difficulty": difficulty,
        "language": language,
        "input": {
            "prompt": prompt,
            "context": context
        },
        "expected_behavior": {
            "must_include": must_include,
            "must_not_include": must_not_include,
            "ideal_response": ideal_response
        },
        "evaluation_criteria": {
            "correctness_weight": correctness,
            "completeness_weight": completeness,
            "clarity_weight": clarity
        },
        "metadata": {
            "source": source,
            "created_at": date.today().isoformat(),
            "tags": tags or []
        }
    }


def generate_code_review_cases() -> list:
    """Generate code review benchmark cases."""
    cases = []

    # review-001: SQL Injection vulnerability
    cases.append(create_case(
        id="review-001",
        category="code-review",
        difficulty="easy",
        language="python",
        prompt="Revisa este código y encuentra problemas de seguridad",
        context='''def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)''',
        must_include=["SQL injection", "parameterized", "prepared statement"],
        must_not_include=["el código está bien", "no hay problemas"],
        ideal_response="Este código tiene una vulnerabilidad de SQL injection. El user_id se concatena directamente en la query sin sanitizar. Debe usar queries parametrizadas: db.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        tags=["security", "sql", "owasp"]
    ))

    # review-002: Missing error handling
    cases.append(create_case(
        id="review-002",
        category="code-review",
        difficulty="easy",
        language="python",
        prompt="¿Qué problemas tiene este código?",
        context='''def read_config():
    with open("config.json") as f:
        return json.load(f)''',
        must_include=["FileNotFoundError", "JSONDecodeError", "try", "except"],
        must_not_include=["perfecto", "no hay errores"],
        ideal_response="El código no maneja excepciones. Puede fallar si el archivo no existe (FileNotFoundError) o si el JSON es inválido (JSONDecodeError). Debería usar try/except y manejar estos casos.",
        tags=["error-handling", "exceptions"]
    ))

    # review-003: Race condition
    cases.append(create_case(
        id="review-003",
        category="code-review",
        difficulty="hard",
        language="python",
        prompt="Analiza este código para problemas de concurrencia",
        context='''class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        current = self.value
        self.value = current + 1''',
        must_include=["race condition", "thread", "lock", "atomic"],
        must_not_include=["thread-safe", "no hay problema"],
        ideal_response="Este código tiene una race condition. Si múltiples threads llaman increment() simultáneamente, pueden leer el mismo valor y sobrescribirse. Solución: usar threading.Lock() o atomic operations.",
        correctness=0.6,
        completeness=0.25,
        clarity=0.15,
        tags=["concurrency", "threading", "race-condition"]
    ))

    # review-004: Memory leak potential
    cases.append(create_case(
        id="review-004",
        category="code-review",
        difficulty="medium",
        language="python",
        prompt="Revisa este código por posibles memory leaks",
        context='''class DataProcessor:
    _cache = {}  # Class-level cache

    def process(self, data_id, data):
        result = expensive_computation(data)
        DataProcessor._cache[data_id] = result
        return result''',
        must_include=["memory leak", "cache", "crece", "límite", "LRU"],
        must_not_include=["está bien", "no hay leak"],
        ideal_response="El cache a nivel de clase crece indefinidamente sin límite. Esto causa memory leak en aplicaciones de larga duración. Solución: usar functools.lru_cache con maxsize, o implementar limpieza periódica del cache.",
        tags=["memory", "cache", "performance"]
    ))

    # review-005: Hardcoded secrets
    cases.append(create_case(
        id="review-005",
        category="code-review",
        difficulty="easy",
        language="python",
        prompt="¿Hay problemas de seguridad en este código?",
        context='''def connect_to_db():
    return psycopg2.connect(
        host="prod-db.example.com",
        password="SuperSecret123!",
        database="production"
    )''',
        must_include=["hardcoded", "secret", "environment variable", "vault"],
        must_not_include=["seguro", "está bien"],
        ideal_response="Las credenciales están hardcodeadas en el código. Esto es un riesgo de seguridad grave. Deben usar variables de entorno (os.environ) o un secret manager como HashiCorp Vault.",
        tags=["security", "secrets", "credentials"]
    ))

    # review-006: N+1 query problem
    cases.append(create_case(
        id="review-006",
        category="code-review",
        difficulty="medium",
        language="python",
        prompt="Analiza el rendimiento de este código",
        context='''def get_orders_with_products():
    orders = Order.query.all()
    result = []
    for order in orders:
        products = Product.query.filter_by(order_id=order.id).all()
        result.append({"order": order, "products": products})
    return result''',
        must_include=["N+1", "query", "join", "eager loading", "rendimiento"],
        must_not_include=["eficiente", "óptimo"],
        ideal_response="Este código tiene el problema N+1: ejecuta 1 query para orders + N queries para products. Con 1000 orders = 1001 queries. Solución: usar JOIN o eager loading (joinedload en SQLAlchemy).",
        correctness=0.5,
        completeness=0.3,
        clarity=0.2,
        tags=["performance", "database", "n+1"]
    ))

    # review-007: XSS vulnerability
    cases.append(create_case(
        id="review-007",
        category="code-review",
        difficulty="easy",
        language="typescript",
        prompt="Revisa este componente React por vulnerabilidades",
        context='''function Comment({ text }) {
    return <div dangerouslySetInnerHTML={{ __html: text }} />;
}''',
        must_include=["XSS", "dangerouslySetInnerHTML", "sanitize", "DOMPurify"],
        must_not_include=["seguro", "no hay problema"],
        ideal_response="Este código es vulnerable a XSS. dangerouslySetInnerHTML renderiza HTML sin sanitizar. Si 'text' viene del usuario, pueden inyectar scripts maliciosos. Solución: usar DOMPurify.sanitize(text) o evitar innerHTML.",
        tags=["security", "xss", "react"]
    ))

    # review-008: Async/await error handling
    cases.append(create_case(
        id="review-008",
        category="code-review",
        difficulty="medium",
        language="typescript",
        prompt="¿Qué problemas tiene este código async?",
        context='''async function fetchAllData() {
    const users = await fetch('/api/users');
    const orders = await fetch('/api/orders');
    const products = await fetch('/api/products');
    return { users, orders, products };
}''',
        must_include=["Promise.all", "paralelo", "secuencial", "rendimiento"],
        must_not_include=["óptimo", "correcto"],
        ideal_response="Las llamadas son secuenciales cuando podrían ser paralelas. Cada await espera a que termine el anterior. Solución: usar Promise.all([fetch('/api/users'), fetch('/api/orders'), fetch('/api/products')]) para ejecutar en paralelo.",
        tags=["async", "performance", "promises"]
    ))

    # review-009: Terraform security group
    cases.append(create_case(
        id="review-009",
        category="code-review",
        difficulty="medium",
        language="mixed",
        prompt="Revisa esta configuración de Terraform",
        context='''resource "aws_security_group" "web" {
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}''',
        must_include=["0.0.0.0/0", "todos los puertos", "restringir", "seguridad"],
        must_not_include=["correcto", "seguro"],
        ideal_response="Esta configuración abre TODOS los puertos TCP a TODO internet (0.0.0.0/0). Es extremadamente inseguro. Debe restringir: 1) Solo puertos necesarios (80, 443), 2) Solo IPs conocidas o usar VPC.",
        tags=["terraform", "security", "aws"]
    ))

    # review-010: Missing input validation
    cases.append(create_case(
        id="review-010",
        category="code-review",
        difficulty="easy",
        language="python",
        prompt="Revisa este endpoint de API",
        context='''@app.post("/transfer")
def transfer(from_account: str, to_account: str, amount: float):
    db.execute(f"UPDATE accounts SET balance = balance - {amount} WHERE id = '{from_account}'")
    db.execute(f"UPDATE accounts SET balance = balance + {amount} WHERE id = '{to_account}'")
    return {"status": "ok"}''',
        must_include=["validación", "SQL injection", "transacción", "negativo", "rollback"],
        must_not_include=["correcto", "seguro"],
        ideal_response="Múltiples problemas: 1) SQL injection en ambas queries, 2) No valida que amount sea positivo, 3) No usa transacción (si falla la segunda query, el dinero desaparece), 4) No verifica saldo suficiente.",
        correctness=0.6,
        completeness=0.3,
        clarity=0.1,
        tags=["security", "validation", "transactions"]
    ))

    return cases


def generate_debugging_cases() -> list:
    """Generate debugging benchmark cases."""
    cases = []

    # debug-001: Off-by-one error
    cases.append(create_case(
        id="debug-001",
        category="debugging",
        difficulty="easy",
        language="python",
        prompt="Este código debería imprimir números del 1 al 10, pero no funciona correctamente. ¿Por qué?",
        context='''for i in range(1, 10):
    print(i)''',
        must_include=["range", "exclusivo", "11", "range(1, 11)"],
        must_not_include=["está correcto"],
        ideal_response="range(1, 10) genera números del 1 al 9 porque el límite superior es exclusivo. Para incluir el 10, debe ser range(1, 11).",
        tags=["range", "off-by-one"]
    ))

    # debug-002: Mutable default argument
    cases.append(create_case(
        id="debug-002",
        category="debugging",
        difficulty="medium",
        language="python",
        prompt="Esta función debería crear una lista nueva cada vez, pero las listas se acumulan. ¿Qué pasa?",
        context='''def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] - ¿Por qué?''',
        must_include=["mutable", "default", "None", "argumento por defecto"],
        must_not_include=["está bien"],
        ideal_response="Los argumentos por defecto mutables (como []) se crean UNA vez cuando se define la función, no en cada llamada. Todas las llamadas comparten la misma lista. Solución: usar None como default y crear la lista dentro: if items is None: items = []",
        tags=["mutable-default", "gotcha", "python"]
    ))

    # debug-003: Closure in loop
    cases.append(create_case(
        id="debug-003",
        category="debugging",
        difficulty="hard",
        language="python",
        prompt="Esperaba que imprimiera 0,1,2,3,4 pero imprime 4,4,4,4,4. ¿Por qué?",
        context='''functions = []
for i in range(5):
    functions.append(lambda: print(i))

for f in functions:
    f()''',
        must_include=["closure", "variable", "captura", "referencia"],
        must_not_include=["correcto"],
        ideal_response="El lambda captura la REFERENCIA a 'i', no su valor. Cuando se ejecutan los lambdas, i ya vale 4 (último valor del loop). Solución: capturar el valor con un argumento default: lambda i=i: print(i)",
        correctness=0.5,
        completeness=0.3,
        clarity=0.2,
        tags=["closure", "lambda", "scope"]
    ))

    # debug-004: Integer division
    cases.append(create_case(
        id="debug-004",
        category="debugging",
        difficulty="easy",
        language="python",
        prompt="El cálculo de porcentaje está dando 0 en vez de 0.5. ¿Por qué?",
        context='''completed = 5
total = 10
percentage = completed / total
print(percentage)  # Esperaba 0.5 pero obtengo 0''',
        must_include=["Python 2", "división entera", "float", "/"],
        must_not_include=[],
        ideal_response="En Python 2, la división entre enteros es división entera (5/10=0). Soluciones: 1) Usar Python 3 donde / es división real, 2) Convertir a float: float(completed)/total, 3) Usar from __future__ import division.",
        tags=["division", "python2", "types"]
    ))

    # debug-005: Shallow copy issue
    cases.append(create_case(
        id="debug-005",
        category="debugging",
        difficulty="medium",
        language="python",
        prompt="Modifico la copia pero el original también cambia. ¿Qué está mal?",
        context='''original = [[1, 2], [3, 4]]
copy = original.copy()
copy[0][0] = 99
print(original)  # [[99, 2], [3, 4]] - ¿Por qué cambió?''',
        must_include=["shallow copy", "deep copy", "referencia", "copy.deepcopy"],
        must_not_include=["bug de Python"],
        ideal_response="list.copy() hace shallow copy: copia la lista externa pero las listas internas son referencias a las mismas. Modificar copy[0][0] modifica el mismo objeto que original[0][0]. Solución: usar copy.deepcopy(original).",
        tags=["copy", "reference", "mutable"]
    ))

    # debug-006: Async not awaited
    cases.append(create_case(
        id="debug-006",
        category="debugging",
        difficulty="easy",
        language="python",
        prompt="Esta función async no devuelve datos, solo un objeto coroutine. ¿Por qué?",
        context='''async def fetch_data():
    return await some_api_call()

def main():
    data = fetch_data()
    print(data)  # <coroutine object fetch_data at 0x...>''',
        must_include=["await", "coroutine", "async", "asyncio.run"],
        must_not_include=["correcto"],
        ideal_response="Las funciones async devuelven un coroutine que debe ser awaited. En main() falta await, pero main() no es async. Solución: hacer main() async y usar await, o usar asyncio.run(fetch_data()).",
        tags=["async", "await", "coroutine"]
    ))

    # debug-007: TypeScript null check
    cases.append(create_case(
        id="debug-007",
        category="debugging",
        difficulty="medium",
        language="typescript",
        prompt="TypeScript da error 'Object is possibly undefined' aunque verifico con if. ¿Por qué?",
        context='''interface User { name: string; address?: { city: string } }

function getCity(user: User): string {
    if (user.address) {
        return user.address.city;  // Error aquí
    }
    return "Unknown";
}''',
        must_include=["narrowing", "optional chaining", "strictNullChecks"],
        must_not_include=["bug de TypeScript"],
        ideal_response="El código debería funcionar con type narrowing. Si hay error, puede ser: 1) Versión vieja de TS, 2) El check está en otro scope. Alternativas: optional chaining (user.address?.city) o non-null assertion (user.address!.city si estás seguro).",
        tags=["typescript", "null", "types"]
    ))

    # debug-008: Docker container exit
    cases.append(create_case(
        id="debug-008",
        category="debugging",
        difficulty="medium",
        language="bash",
        prompt="Mi contenedor Docker arranca y se detiene inmediatamente. ¿Por qué?",
        context='''FROM python:3.9
COPY app.py /app/
WORKDIR /app
CMD python app.py''',
        must_include=["foreground", "proceso", "daemon", "exec form"],
        must_not_include=["está bien"],
        ideal_response="Docker mantiene el contenedor mientras el proceso principal corra. Posibles causas: 1) app.py termina rápido, 2) CMD en shell form puede tener problemas de señales. Soluciones: 1) Verificar que app.py tenga un loop o servidor, 2) Usar exec form: CMD [\"python\", \"app.py\"], 3) Ver logs: docker logs <container>",
        tags=["docker", "container", "process"]
    ))

    # debug-009: Git merge conflict
    cases.append(create_case(
        id="debug-009",
        category="debugging",
        difficulty="easy",
        language="mixed",
        prompt="Tengo conflictos de merge y no sé cómo resolverlos. El archivo muestra estos marcadores:",
        context='''<<<<<<< HEAD
const API_URL = "https://prod.api.com";
=======
const API_URL = "https://staging.api.com";
>>>>>>> feature-branch''',
        must_include=["HEAD", "feature-branch", "elegir", "eliminar marcadores"],
        must_not_include=["automático"],
        ideal_response="Los marcadores muestran: entre <<<<<<< HEAD y ======= está tu versión actual, entre ======= y >>>>>>> está la versión de feature-branch. Debes: 1) Elegir qué código mantener (o combinar ambos), 2) Eliminar los marcadores (<<<<, ====, >>>>), 3) git add y git commit.",
        tags=["git", "merge", "conflict"]
    ))

    # debug-010: React useEffect infinite loop
    cases.append(create_case(
        id="debug-010",
        category="debugging",
        difficulty="medium",
        language="typescript",
        prompt="Mi componente React hace llamadas infinitas a la API. ¿Por qué?",
        context='''function UserList() {
    const [users, setUsers] = useState([]);

    useEffect(() => {
        fetch('/api/users')
            .then(res => res.json())
            .then(data => setUsers(data));
    });

    return <ul>{users.map(u => <li>{u.name}</li>)}</ul>;
}''',
        must_include=["dependency array", "[]", "useEffect", "re-render"],
        must_not_include=["bug de React"],
        ideal_response="useEffect sin dependency array se ejecuta en CADA render. setUsers causa re-render, que ejecuta useEffect de nuevo = loop infinito. Solución: agregar [] como segundo argumento para ejecutar solo al montar: useEffect(() => {...}, [])",
        tags=["react", "useEffect", "hooks"]
    ))

    return cases


def generate_explanation_cases() -> list:
    """Generate code explanation benchmark cases."""
    cases = []

    # explain-001: Decorator pattern
    cases.append(create_case(
        id="explain-001",
        category="explanation",
        difficulty="medium",
        language="python",
        prompt="Explica qué hace este código y cómo funciona",
        context='''def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
            return wrapper
        return decorator
    return retry''',
        must_include=["decorator", "retry", "intentos", "excepción", "wrapper"],
        must_not_include=["simple", "básico"],
        ideal_response="Es un decorator con parámetros que implementa retry logic. Cuando decoras una función con @retry(3), si la función lanza excepción, se reintenta hasta 3 veces. Solo propaga la excepción si todos los intentos fallan.",
        tags=["decorator", "patterns", "error-handling"]
    ))

    # explain-002: Context manager
    cases.append(create_case(
        id="explain-002",
        category="explanation",
        difficulty="easy",
        language="python",
        prompt="¿Qué es un context manager y para qué sirve 'with'?",
        context='''with open("file.txt") as f:
    content = f.read()''',
        must_include=["__enter__", "__exit__", "cleanup", "automático", "recursos"],
        must_not_include=["magia"],
        ideal_response="Un context manager gestiona recursos automáticamente. 'with' llama __enter__ al inicio y __exit__ al final (incluso si hay excepciones). En este caso, garantiza que el archivo se cierre sin importar qué pase dentro del bloque.",
        tags=["context-manager", "with", "resources"]
    ))

    # explain-003: Generator expression
    cases.append(create_case(
        id="explain-003",
        category="explanation",
        difficulty="medium",
        language="python",
        prompt="¿Cuál es la diferencia entre estas dos líneas?",
        context='''lista = [x**2 for x in range(1000000)]
generador = (x**2 for x in range(1000000))''',
        must_include=["memoria", "lazy", "generador", "lista", "yield"],
        must_not_include=["iguales", "lo mismo"],
        ideal_response="La lista comprehension crea TODA la lista en memoria (usa ~8MB). El generator expression es lazy: calcula valores bajo demanda, usando memoria constante. Para iteración única, el generador es más eficiente.",
        tags=["generator", "memory", "performance"]
    ))

    # explain-004: Dependency injection
    cases.append(create_case(
        id="explain-004",
        category="explanation",
        difficulty="medium",
        language="python",
        prompt="¿Por qué este diseño es mejor que instanciar la dependencia directamente?",
        context='''class OrderService:
    def __init__(self, payment_gateway: PaymentGateway):
        self.payment = payment_gateway

    def process(self, order):
        self.payment.charge(order.total)

# Uso:
service = OrderService(StripeGateway())''',
        must_include=["dependency injection", "testing", "mock", "desacoplado", "intercambiar"],
        must_not_include=["innecesario", "sobre-ingeniería"],
        ideal_response="Es Dependency Injection: la dependencia se pasa desde fuera en vez de crearse dentro. Ventajas: 1) Facilita testing (puedes pasar un mock), 2) Desacopla OrderService de una implementación específica, 3) Permite cambiar gateways sin modificar OrderService.",
        tags=["di", "patterns", "testing"]
    ))

    # explain-005: Event loop
    cases.append(create_case(
        id="explain-005",
        category="explanation",
        difficulty="hard",
        language="python",
        prompt="Explica cómo funciona async/await en Python",
        context='''async def main():
    task1 = asyncio.create_task(fetch_url("url1"))
    task2 = asyncio.create_task(fetch_url("url2"))
    result1 = await task1
    result2 = await task2

asyncio.run(main())''',
        must_include=["event loop", "concurrencia", "no paralelo", "I/O", "coroutine"],
        must_not_include=["hilos", "threads paralelos"],
        ideal_response="async/await permite concurrencia cooperativa. El event loop ejecuta coroutines: cuando una hace await (espera I/O), el loop ejecuta otra. create_task() programa las tareas, await espera su resultado. No es paralelismo real (un solo thread), pero eficiente para I/O.",
        correctness=0.5,
        completeness=0.35,
        clarity=0.15,
        tags=["async", "event-loop", "concurrency"]
    ))

    # explain-006: React hooks lifecycle
    cases.append(create_case(
        id="explain-006",
        category="explanation",
        difficulty="medium",
        language="typescript",
        prompt="Explica cuándo se ejecuta cada parte de este useEffect",
        context='''useEffect(() => {
    console.log("Effect runs");
    const subscription = api.subscribe();

    return () => {
        console.log("Cleanup runs");
        subscription.unsubscribe();
    };
}, [userId]);''',
        must_include=["mount", "cleanup", "dependencias", "userId", "unmount"],
        must_not_include=["siempre", "cada render"],
        ideal_response="El effect corre: 1) Al montar el componente, 2) Cuando userId cambia. El cleanup (return function) corre: 1) Antes de re-ejecutar el effect (cuando userId cambia), 2) Al desmontar el componente. Esto previene memory leaks de subscripciones.",
        tags=["react", "useEffect", "lifecycle"]
    ))

    # explain-007: SQL window function
    cases.append(create_case(
        id="explain-007",
        category="explanation",
        difficulty="hard",
        language="sql",
        prompt="¿Qué hace esta query y cómo funciona ROW_NUMBER?",
        context='''SELECT
    employee_id,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rank
FROM employees;''',
        must_include=["window function", "PARTITION", "ranking", "departamento", "ORDER BY"],
        must_not_include=["GROUP BY"],
        ideal_response="ROW_NUMBER() es una window function que asigna números secuenciales. PARTITION BY department reinicia el contador por cada departamento. ORDER BY salary DESC ordena de mayor a menor salario. Resultado: cada empleado tiene su ranking de salario dentro de su departamento.",
        tags=["sql", "window-function", "analytics"]
    ))

    # explain-008: Kubernetes deployment
    cases.append(create_case(
        id="explain-008",
        category="explanation",
        difficulty="medium",
        language="mixed",
        prompt="Explica qué hace esta configuración de Kubernetes",
        context='''apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    spec:
      containers:
      - name: web
        image: myapp:v1
        resources:
          limits:
            memory: "128Mi"
            cpu: "500m"''',
        must_include=["replicas", "pods", "contenedor", "recursos", "límites"],
        must_not_include=["difícil", "complejo"],
        ideal_response="Este Deployment crea 3 réplicas (pods) de la aplicación. Cada pod corre el contenedor myapp:v1 con límites de recursos (128MB RAM, 0.5 CPU). El selector app:web permite que Services encuentren estos pods. Kubernetes mantiene siempre 3 réplicas corriendo.",
        tags=["kubernetes", "deployment", "devops"]
    ))

    # explain-009: TypeScript generics
    cases.append(create_case(
        id="explain-009",
        category="explanation",
        difficulty="medium",
        language="typescript",
        prompt="¿Qué significa <T> y para qué sirve en este código?",
        context='''function identity<T>(arg: T): T {
    return arg;
}

const num = identity<number>(42);
const str = identity("hello");''',
        must_include=["generic", "tipo", "reutilizable", "type parameter", "inferencia"],
        must_not_include=["any"],
        ideal_response="<T> es un type parameter (generic). Permite que la función trabaje con cualquier tipo manteniendo type safety. En identity<number>(42), T es number. En identity(\"hello\"), TypeScript infiere T como string. Es más seguro que usar 'any' porque preserva el tipo.",
        tags=["typescript", "generics", "types"]
    ))

    # explain-010: Docker multi-stage build
    cases.append(create_case(
        id="explain-010",
        category="explanation",
        difficulty="medium",
        language="bash",
        prompt="¿Por qué se usa 'FROM' dos veces en este Dockerfile?",
        context='''FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html''',
        must_include=["multi-stage", "imagen", "tamaño", "build", "producción"],
        must_not_include=["error", "incorrecto"],
        ideal_response="Es un multi-stage build. Stage 1 (builder): usa Node para compilar la app. Stage 2: solo copia el resultado compilado a una imagen nginx ligera. Beneficio: la imagen final NO incluye Node, npm, ni node_modules - solo nginx y los archivos estáticos. Reduce tamaño dramáticamente (de ~1GB a ~50MB).",
        tags=["docker", "multi-stage", "optimization"]
    ))

    return cases


def generate_refactoring_cases() -> list:
    """Generate refactoring benchmark cases."""
    cases = []

    # refactor-001: Extract method
    cases.append(create_case(
        id="refactor-001",
        category="refactoring",
        difficulty="easy",
        language="python",
        prompt="Este código es difícil de leer. ¿Cómo lo mejorarías?",
        context='''def process_order(order):
    # Validate
    if not order.items:
        raise ValueError("Empty order")
    if order.total < 0:
        raise ValueError("Invalid total")
    for item in order.items:
        if item.quantity <= 0:
            raise ValueError("Invalid quantity")

    # Calculate discounts
    discount = 0
    if order.total > 100:
        discount = order.total * 0.1
    if order.customer.is_premium:
        discount += order.total * 0.05

    # Process payment
    final_total = order.total - discount
    payment_result = payment_gateway.charge(order.customer, final_total)

    return payment_result''',
        must_include=["extraer", "función", "método", "validate", "calculate", "responsabilidad"],
        must_not_include=["está bien", "no cambiar"],
        ideal_response="Refactorizar extrayendo métodos: 1) validate_order(order) para las validaciones, 2) calculate_discount(order) para el descuento, 3) La función principal queda: validate_order(order); discount = calculate_discount(order); return process_payment(order, discount). Cada función tiene una responsabilidad.",
        tags=["extract-method", "clean-code", "srp"]
    ))

    # refactor-002: Replace conditional with polymorphism
    cases.append(create_case(
        id="refactor-002",
        category="refactoring",
        difficulty="hard",
        language="python",
        prompt="Este código tiene muchos if/elif. ¿Cómo lo mejorarías?",
        context='''def calculate_shipping(order):
    if order.shipping_type == "standard":
        return 5.99
    elif order.shipping_type == "express":
        return 15.99
    elif order.shipping_type == "overnight":
        return 25.99
    elif order.shipping_type == "international":
        base = 30.0
        return base + (order.weight * 2.5)
    else:
        raise ValueError("Unknown shipping type")''',
        must_include=["polimorfismo", "clase", "strategy", "diccionario", "extensible"],
        must_not_include=["está bien"],
        ideal_response="Usar Strategy pattern o diccionario. Opción 1: clases ShippingStrategy con método calculate(). Opción 2: diccionario de funciones {\"standard\": lambda o: 5.99, ...}. Beneficios: agregar nuevo tipo no requiere modificar la función, cada lógica está aislada.",
        correctness=0.5,
        completeness=0.3,
        clarity=0.2,
        tags=["polymorphism", "strategy", "patterns"]
    ))

    # refactor-003: Remove duplication
    cases.append(create_case(
        id="refactor-003",
        category="refactoring",
        difficulty="easy",
        language="typescript",
        prompt="Hay código duplicado aquí. ¿Cómo lo refactorizarías?",
        context='''async function getUser(id: string) {
    try {
        const response = await fetch(`/api/users/${id}`);
        if (!response.ok) throw new Error('Failed');
        return await response.json();
    } catch (error) {
        console.error('Error fetching user:', error);
        throw error;
    }
}

async function getProduct(id: string) {
    try {
        const response = await fetch(`/api/products/${id}`);
        if (!response.ok) throw new Error('Failed');
        return await response.json();
    } catch (error) {
        console.error('Error fetching product:', error);
        throw error;
    }
}''',
        must_include=["genérico", "parametrizar", "DRY", "reutilizar"],
        must_not_include=["está bien", "no duplicado"],
        ideal_response="Crear función genérica: async function fetchResource<T>(endpoint: string): Promise<T>. Las funciones específicas llaman a esta: getUser = (id) => fetchResource(`/api/users/${id}`). Elimina duplicación y centraliza manejo de errores.",
        tags=["dry", "generics", "abstraction"]
    ))

    # refactor-004: Simplify boolean logic
    cases.append(create_case(
        id="refactor-004",
        category="refactoring",
        difficulty="easy",
        language="python",
        prompt="Esta condición es confusa. ¿Cómo la simplificarías?",
        context='''def can_access(user, resource):
    if user.is_admin == True:
        return True
    else:
        if user.role == "editor":
            if resource.is_public == True or resource.owner_id == user.id:
                return True
            else:
                return False
        else:
            if resource.is_public == True:
                return True
            else:
                return False''',
        must_include=["simplificar", "return directo", "or", "and"],
        must_not_include=["correcto", "claro"],
        ideal_response="Simplificar a: return (user.is_admin or resource.is_public or (user.role == 'editor' and resource.owner_id == user.id)). Evitar: == True (redundante), else return False (innecesario si retornas True antes), anidación excesiva.",
        tags=["boolean", "simplify", "clean-code"]
    ))

    # refactor-005: Use dataclass
    cases.append(create_case(
        id="refactor-005",
        category="refactoring",
        difficulty="easy",
        language="python",
        prompt="Esta clase tiene mucho boilerplate. ¿Cómo la mejorarías?",
        context='''class User:
    def __init__(self, id, name, email, age):
        self.id = id
        self.name = name
        self.email = email
        self.age = age

    def __eq__(self, other):
        return (self.id == other.id and self.name == other.name
                and self.email == other.email and self.age == other.age)

    def __repr__(self):
        return f"User(id={self.id}, name={self.name}, email={self.email}, age={self.age})"''',
        must_include=["dataclass", "@dataclass", "automático", "boilerplate"],
        must_not_include=["necesario", "correcto"],
        ideal_response="Usar @dataclass: from dataclasses import dataclass; @dataclass class User: id: int; name: str; email: str; age: int. Genera automáticamente __init__, __eq__, __repr__. Opciones: frozen=True para inmutabilidad, field() para defaults.",
        tags=["dataclass", "python", "boilerplate"]
    ))

    # refactor-006: Early return
    cases.append(create_case(
        id="refactor-006",
        category="refactoring",
        difficulty="easy",
        language="python",
        prompt="Esta función tiene mucha anidación. ¿Cómo reducirla?",
        context='''def process_payment(order):
    result = None
    if order is not None:
        if order.is_valid():
            if order.total > 0:
                if order.customer.has_payment_method():
                    result = payment_gateway.charge(order)
                else:
                    result = "No payment method"
            else:
                result = "Invalid total"
        else:
            result = "Invalid order"
    else:
        result = "No order"
    return result''',
        must_include=["early return", "guard clause", "anidación", "invertir"],
        must_not_include=["correcto", "claro"],
        ideal_response="Usar early returns/guard clauses: if order is None: return 'No order'; if not order.is_valid(): return 'Invalid order'; etc. La lógica principal queda sin anidación al final. Más legible y fácil de seguir.",
        tags=["early-return", "guard-clause", "clean-code"]
    ))

    # refactor-007: Extract configuration
    cases.append(create_case(
        id="refactor-007",
        category="refactoring",
        difficulty="medium",
        language="python",
        prompt="Los valores hardcodeados dificultan el mantenimiento. ¿Cómo mejorar?",
        context='''def send_notification(user, message):
    smtp = smtplib.SMTP("smtp.gmail.com", 587)
    smtp.login("app@company.com", "secretpassword123")
    smtp.send(user.email, message)

    if len(message) < 160:
        twilio = Twilio("AC123456", "auth_token_here")
        twilio.send("+1234567890", user.phone, message)''',
        must_include=["configuración", "environment", "constantes", "settings"],
        must_not_include=["está bien"],
        ideal_response="Extraer a configuración: 1) Credenciales a variables de entorno o secrets manager, 2) Valores como SMTP_HOST, SMS_MAX_LENGTH a archivo de config o constantes. 3) Inyectar clientes ya configurados. Nunca hardcodear credenciales.",
        tags=["configuration", "secrets", "maintainability"]
    ))

    # refactor-008: Decompose complex query
    cases.append(create_case(
        id="refactor-008",
        category="refactoring",
        difficulty="medium",
        language="sql",
        prompt="Esta query es difícil de entender. ¿Cómo la harías más clara?",
        context='''SELECT * FROM orders o
JOIN customers c ON o.customer_id = c.id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.status = 'pending'
  AND o.created_at > DATE_SUB(NOW(), INTERVAL 30 DAY)
  AND c.country IN ('US', 'CA', 'MX')
  AND p.category = 'electronics'
  AND (o.total > 100 OR c.is_premium = 1)
ORDER BY o.total DESC
LIMIT 50;''',
        must_include=["CTE", "WITH", "subconsulta", "nombres", "legible"],
        must_not_include=["óptima", "perfecta"],
        ideal_response="Usar CTEs para separar lógica: WITH recent_orders AS (SELECT ... WHERE created_at > ...), north_american_customers AS (...), electronics_items AS (...) SELECT ... FROM recent_orders JOIN .... Cada CTE tiene nombre descriptivo y responsabilidad clara.",
        tags=["sql", "cte", "readability"]
    ))

    # refactor-009: Replace magic numbers
    cases.append(create_case(
        id="refactor-009",
        category="refactoring",
        difficulty="easy",
        language="python",
        prompt="Este código tiene 'números mágicos'. ¿Cómo mejorarlo?",
        context='''def calculate_score(attempts, time_seconds):
    base_score = 1000
    if attempts > 3:
        base_score -= (attempts - 3) * 50
    if time_seconds > 300:
        base_score -= (time_seconds - 300) * 2
    if base_score < 0:
        base_score = 0
    return base_score''',
        must_include=["constantes", "nombre", "MAX_", "significado"],
        must_not_include=["claro", "entendible"],
        ideal_response="Reemplazar con constantes nombradas: MAX_BASE_SCORE = 1000, FREE_ATTEMPTS = 3, POINTS_PER_EXTRA_ATTEMPT = 50, TIME_LIMIT_SECONDS = 300, POINTS_PER_EXTRA_SECOND = 2, MIN_SCORE = 0. El código se auto-documenta y los valores son fáciles de ajustar.",
        tags=["magic-numbers", "constants", "readability"]
    ))

    # refactor-010: Async/await conversion
    cases.append(create_case(
        id="refactor-010",
        category="refactoring",
        difficulty="medium",
        language="typescript",
        prompt="Este código con Promises es difícil de seguir. ¿Cómo mejorarlo?",
        context='''function getUserOrders(userId: string) {
    return getUser(userId)
        .then(user => {
            return getOrders(user.id)
                .then(orders => {
                    return Promise.all(orders.map(order =>
                        getOrderDetails(order.id)
                    ));
                })
                .then(details => {
                    return { user, details };
                });
        })
        .catch(error => {
            console.error(error);
            throw error;
        });
}''',
        must_include=["async", "await", "try/catch", "secuencial", "legible"],
        must_not_include=["igual", "correcto"],
        ideal_response="Convertir a async/await: async function getUserOrders(userId) { try { const user = await getUser(userId); const orders = await getOrders(user.id); const details = await Promise.all(orders.map(o => getOrderDetails(o.id))); return { user, details }; } catch (error) { ... } }. Más legible, flujo lineal.",
        tags=["async-await", "promises", "refactoring"]
    ))

    return cases


def generate_generation_cases() -> list:
    """Generate code generation benchmark cases."""
    cases = []

    # gen-001: Simple function
    cases.append(create_case(
        id="gen-001",
        category="generation",
        difficulty="easy",
        language="python",
        prompt="Escribe una función que calcule el factorial de un número",
        context="",
        must_include=["def", "factorial", "return", "recursivo", "iterativo"],
        must_not_include=["import math"],
        ideal_response="def factorial(n): if n <= 1: return 1; return n * factorial(n-1). O iterativo: result = 1; for i in range(2, n+1): result *= i; return result",
        tags=["function", "math", "recursion"]
    ))

    # gen-002: Class design
    cases.append(create_case(
        id="gen-002",
        category="generation",
        difficulty="medium",
        language="python",
        prompt="Crea una clase BankAccount con depósito, retiro y balance",
        context="",
        must_include=["class", "__init__", "balance", "deposit", "withdraw", "validación"],
        must_not_include=["global"],
        ideal_response="class BankAccount: def __init__(self, initial=0): self._balance = initial; def deposit(self, amount): if amount > 0: self._balance += amount; def withdraw(self, amount): if 0 < amount <= self._balance: self._balance -= amount; return True; return False; @property def balance(self): return self._balance",
        tags=["class", "oop", "encapsulation"]
    ))

    # gen-003: API endpoint
    cases.append(create_case(
        id="gen-003",
        category="generation",
        difficulty="medium",
        language="python",
        prompt="Crea un endpoint FastAPI para crear usuarios con validación",
        context="Campos: name (str, requerido), email (str, email válido), age (int, >= 18)",
        must_include=["@app.post", "Pydantic", "BaseModel", "EmailStr", "Field", "HTTPException"],
        must_not_include=["sin validación"],
        ideal_response="from pydantic import BaseModel, EmailStr, Field; class UserCreate(BaseModel): name: str = Field(min_length=1); email: EmailStr; age: int = Field(ge=18); @app.post('/users') async def create_user(user: UserCreate): ...",
        tags=["fastapi", "api", "validation"]
    ))

    # gen-004: SQL query
    cases.append(create_case(
        id="gen-004",
        category="generation",
        difficulty="medium",
        language="sql",
        prompt="Escribe una query para obtener los 5 productos más vendidos por categoría",
        context="Tablas: products(id, name, category_id), order_items(product_id, quantity), categories(id, name)",
        must_include=["JOIN", "GROUP BY", "SUM", "ROW_NUMBER", "PARTITION BY", "WHERE"],
        must_not_include=["SELECT *"],
        ideal_response="WITH ranked AS (SELECT p.name, c.name as category, SUM(oi.quantity) as total, ROW_NUMBER() OVER (PARTITION BY p.category_id ORDER BY SUM(oi.quantity) DESC) as rn FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN categories c ON p.category_id = c.id GROUP BY p.id, p.name, c.name, p.category_id) SELECT * FROM ranked WHERE rn <= 5",
        tags=["sql", "window-function", "analytics"]
    ))

    # gen-005: React component
    cases.append(create_case(
        id="gen-005",
        category="generation",
        difficulty="medium",
        language="typescript",
        prompt="Crea un componente React de formulario de login con validación",
        context="Campos: email y password. Mostrar errores de validación.",
        must_include=["useState", "handleSubmit", "onChange", "email", "password", "error"],
        must_not_include=["any"],
        ideal_response="function LoginForm() { const [email, setEmail] = useState(''); const [password, setPassword] = useState(''); const [errors, setErrors] = useState<{email?: string, password?: string}>({}); const handleSubmit = (e) => { e.preventDefault(); const newErrors = {}; if (!email.includes('@')) newErrors.email = 'Invalid email'; if (password.length < 8) newErrors.password = 'Min 8 chars'; ... }; return <form onSubmit={handleSubmit}>...</form> }",
        tags=["react", "forms", "validation"]
    ))

    # gen-006: Unit test
    cases.append(create_case(
        id="gen-006",
        category="generation",
        difficulty="easy",
        language="python",
        prompt="Escribe tests unitarios para una función que suma dos números",
        context="def add(a: int, b: int) -> int: return a + b",
        must_include=["def test_", "assert", "positivos", "negativos", "cero"],
        must_not_include=["pass"],
        ideal_response="def test_add_positive_numbers(): assert add(2, 3) == 5; def test_add_negative_numbers(): assert add(-1, -2) == -3; def test_add_zero(): assert add(0, 5) == 5; def test_add_mixed(): assert add(-1, 5) == 4",
        tags=["testing", "pytest", "unit-test"]
    ))

    # gen-007: Docker compose
    cases.append(create_case(
        id="gen-007",
        category="generation",
        difficulty="medium",
        language="mixed",
        prompt="Crea un docker-compose para una app web con PostgreSQL y Redis",
        context="La app web escucha en puerto 8000, necesita variables de entorno para conectar a DB y cache",
        must_include=["services", "depends_on", "environment", "volumes", "ports", "networks"],
        must_not_include=["version: '2'"],
        ideal_response="services: web: build: .; ports: ['8000:8000']; environment: [DATABASE_URL, REDIS_URL]; depends_on: [db, redis]; db: image: postgres:15; volumes: [postgres_data:/var/lib/postgresql/data]; environment: [POSTGRES_PASSWORD]; redis: image: redis:7; volumes: postgres_data:",
        tags=["docker", "compose", "infrastructure"]
    ))

    # gen-008: Terraform resource
    cases.append(create_case(
        id="gen-008",
        category="generation",
        difficulty="medium",
        language="mixed",
        prompt="Crea un recurso Terraform para un bucket S3 privado con versionado",
        context="Nombre del bucket: my-secure-bucket. Debe bloquear acceso público.",
        must_include=["aws_s3_bucket", "versioning", "block_public", "enabled", "true"],
        must_not_include=["acl = \"public\""],
        ideal_response="resource 'aws_s3_bucket' 'secure' { bucket = 'my-secure-bucket' }; resource 'aws_s3_bucket_versioning' 'secure' { bucket = aws_s3_bucket.secure.id; versioning_configuration { status = 'Enabled' } }; resource 'aws_s3_bucket_public_access_block' 'secure' { bucket = aws_s3_bucket.secure.id; block_public_acls = true; block_public_policy = true; ignore_public_acls = true; restrict_public_buckets = true }",
        tags=["terraform", "aws", "s3"]
    ))

    # gen-009: Bash script
    cases.append(create_case(
        id="gen-009",
        category="generation",
        difficulty="easy",
        language="bash",
        prompt="Escribe un script bash que haga backup de un directorio con fecha",
        context="El script recibe el directorio como argumento. El backup debe ir a /backups/",
        must_include=["$1", "date", "tar", "if", "-d", "echo"],
        must_not_include=["rm -rf"],
        ideal_response="#!/bin/bash; if [ -z \"$1\" ]; then echo 'Usage: backup.sh <dir>'; exit 1; fi; if [ ! -d \"$1\" ]; then echo 'Directory not found'; exit 1; fi; BACKUP_NAME=\"backup_$(basename $1)_$(date +%Y%m%d_%H%M%S).tar.gz\"; tar -czf \"/backups/$BACKUP_NAME\" \"$1\"; echo \"Backup created: $BACKUP_NAME\"",
        tags=["bash", "backup", "scripting"]
    ))

    # gen-010: GraphQL schema
    cases.append(create_case(
        id="gen-010",
        category="generation",
        difficulty="medium",
        language="mixed",
        prompt="Define un schema GraphQL para un blog con posts y comentarios",
        context="Posts tienen: id, title, content, author, createdAt, comments. Comentarios tienen: id, text, author, createdAt",
        must_include=["type", "Query", "Mutation", "ID!", "String!", "[Comment]"],
        must_not_include=["REST"],
        ideal_response="type Post { id: ID!; title: String!; content: String!; author: String!; createdAt: String!; comments: [Comment!]! }; type Comment { id: ID!; text: String!; author: String!; createdAt: String!; postId: ID! }; type Query { posts: [Post!]!; post(id: ID!): Post }; type Mutation { createPost(title: String!, content: String!, author: String!): Post; addComment(postId: ID!, text: String!, author: String!): Comment }",
        tags=["graphql", "schema", "api"]
    ))

    return cases


def main():
    """Generate all benchmark cases and save to files."""
    print("Generating Fabrik Eval Benchmark cases...")

    # Generate all cases
    all_cases = []

    print("  - Code Review cases...")
    code_review = generate_code_review_cases()
    all_cases.extend(code_review)

    print("  - Debugging cases...")
    debugging = generate_debugging_cases()
    all_cases.extend(debugging)

    print("  - Explanation cases...")
    explanation = generate_explanation_cases()
    all_cases.extend(explanation)

    print("  - Refactoring cases...")
    refactoring = generate_refactoring_cases()
    all_cases.extend(refactoring)

    print("  - Generation cases...")
    generation = generate_generation_cases()
    all_cases.extend(generation)

    # Save individual cases by category
    BENCHMARK_PATH.mkdir(parents=True, exist_ok=True)

    categories = {
        "code-review": code_review,
        "debugging": debugging,
        "explanation": explanation,
        "refactoring": refactoring,
        "generation": generation
    }

    for category, cases in categories.items():
        category_file = BENCHMARK_PATH / f"{category}.json"
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(cases)} cases to {category_file.name}")

    # Save combined benchmark
    combined_file = BENCHMARK_PATH / "all_cases.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*50}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"Total cases: {len(all_cases)}")
    print(f"\nBy category:")
    for category, cases in categories.items():
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for case in cases:
            difficulties[case["difficulty"]] += 1
        print(f"  {category}: {len(cases)} (easy:{difficulties['easy']}, medium:{difficulties['medium']}, hard:{difficulties['hard']})")

    print(f"\nFiles saved to: {BENCHMARK_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

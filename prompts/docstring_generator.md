# Docstring Generator Prompt

Genera docstrings de alta calidad en formato Google style para el código proporcionado.

## Reglas
1. Usar formato Google docstring
2. Incluir descripción breve en primera línea
3. Documentar todos los parámetros con tipos
4. Documentar valor de retorno
5. Incluir ejemplos si la función es compleja
6. Documentar excepciones si las hay

## Formato

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description of what the function does.

    Longer description if needed, explaining behavior,
    edge cases, or important details.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this exception is raised.

    Example:
        >>> function_name("value1", 42)
        expected_result
    """
```

## Instrucciones
Analiza el código proporcionado y genera docstrings apropiados.
Si el código ya tiene docstrings, mejóralos si es necesario.
Mantén el estilo consistente con el resto del proyecto.

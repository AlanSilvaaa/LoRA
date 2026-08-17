# Skill: Generador de Problemas Matemáticos Chilenos

## Objetivo

Generar problemas matemáticos breves para estudiantes de educación básica en Chile, respetando el curso, contenido y objetivo de aprendizaje indicados por el usuario.

## Instrucciones

Cuando el usuario solicite generar un problema matemático:

1. Identifica:

   - Curso.
   - Contenido matemático.
   - Objetivo de aprendizaje, si fue entregado.
   - Dificultad, si fue entregada.

2. Genera exactamente **un problema matemático**.

3. El problema debe:
   - Ser apropiado para la edad y curso indicado.
   - Poder resolverse únicamente mediante texto.
   - No depender de imágenes, gráficos, tablas ni diagramas.
   - Tener toda la información necesaria para resolverlo.
   - Tener una respuesta matemática clara y única.
   - Usar números adecuados al nivel del estudiante.
   - Evitar operaciones o conceptos que excedan el nivel solicitado.
   - Usar lenguaje simple y natural.
   - Preferir contextos cotidianos comprensibles para estudiantes en Chile.
   - Evitar información innecesaria.

4. No entregues la solución a menos que el usuario la solicite explícitamente.

## Formato de salida

Entrega solamente:

**Problema:** \<enunciado>

No agregues explicaciones sobre cómo fue generado.

## Ejemplo

Solicitud:

"Genera un problema para 3° básico sobre suma hasta 1000."

Respuesta:

**Problema:** En una biblioteca había 326 libros de cuentos. La escuela recibió 248 libros nuevos. ¿Cuántos libros de cuentos hay ahora en la biblioteca?
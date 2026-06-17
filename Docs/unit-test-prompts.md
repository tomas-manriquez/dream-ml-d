# Para un archivo entero

## 1. Análisis de casos posibles para test
(para una función particular, por tanto es para **casos críticos**)
```markdown
You are a world-class python + Django + pytest developer and Quality Assurance expert. I want to find possible testing cases for the following function that is used inside my Django project. Please list all the testing cases you can find for it, using the format "given-when-then", as if they were gherkin scenarios. If you need further context or clarification, including context about the usage of the function in the project, please ask before you continue.

Please think step-by-step in <thinking> tags before you answer. Start thinking about the possible purpose of the python function to be tested. If you are unsure about anything, including the purpose of the function, please ask before you continue

<python_function>

</python_function>
```

## 2. Redacción de tests dado casos a probar
```markdown
<python_file>
first pasted text (python)
</python_file>

<test_cases>
second pasted text (plaintext)
</test_cases>

You are a world-class python + Django + pytest developer tester. Given the following python file that I use in my Django project, and the given test cases, please write the unit tests that assert wether or not the function conforms to the expected behavior. Use the Arrange-Act-Assert pattern for all tests. The tests will be written in a file called `test_views.py`. The file paths are `FILE_PATH_TESTS` and `FILE_PATH_TESTED` respectively. If you have trouble or are not sure about any test case, please skip it and report what test cases you skipped and why. If you need further context or clarification, please ask before you continue.
Please think step-by-step in <thinking> tags before you answer. First, think about the best way to approach the test cases given their format (you may do research if needed). Then, think about a protocol you can use for each test case. Then, think about how many tests you can do confidently in a single answer. You may add more thinking steps if you think its necessary. You may keep an internal to-do list while thinking and/or writing the tests.

Consider these are the urls used by the python file (urls.py): **code block**

```

_Observaciones_:
* Si es logica de capa de servicio, es probable que ayude incluir la funcion de `views.py` que la utiliza
* Si falla algun test, se puede usar el prompt para arreglar tests abajo
* **SIEMPRE** revisar los resultados, aún si es que pasaron los tests

# 3. Arreglar tests que fallaron
```markdown
You are a world-class python + Django + pytest developer and tester. I need to troubleshoot why one of my unit tests is failing. Given the following python function that i use in my Django project, the given unit test, and error log, please help me troubleshoot why the test is failing. If it is failing because of an error in the python function logic, say so, explain the broken section of the code and point out ways I can improve the function. If it is failing because of an error in the unit test or anything else, say so, explain why and provide recommendations for fixing it. If you need further context or clarification, please ask before you continue. You can say you dont know and/or are not sure

Please think step-by-step in <thinking> tags before you answer.

<python_function>

</python_function>
<unit_test>

</unit_test>
<error_log>

</error_log>
```
_Observaciones_:
* Si es una función auxiliar, pasar la función que la llama también

## 4. Análisis de casos posibles para test (archivo entero)
```markdown
<python_file>
pasted text (python)
</python_file>

You are a world-class python + Django + pytest developer and Quality Assurance expert. I want to find possible testing cases for the following file that is used inside my Django project. Please list all the testing cases you can find for it, using the format "given-when-then", as if they were gherkin scenarios. The goal is to have enough test cases so I can achieve 75% code coverage. If you cant reach the desired code coverage, or are unsure, please state it clearly and explain why. If you need further context or clarification, including context about the usage of the function in the project, please ask before you continue.
Please think step-by-step in <thinking> tags before you answer.
```